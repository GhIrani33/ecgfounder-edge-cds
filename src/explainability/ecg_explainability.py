# -*- coding: utf-8 -*-
"""
ecg_explainability.py
Segment-level Grad-CAM + Lead Influence + Clinical Heuristic Evaluator (+ Temperature Calibration)

- Loads Net1D (ECGFounder) and fine-tuned checkpoint
- Applies optional temperature scaling (from file or CLI)
- Produces:
  * Full-record Grad-CAM overlays
  * Segment-level top-k windows (2.0–2.5s) with Grad-CAM
  * Lead Influence Scores (|grad wrt input|)
  * Clinical heuristic scores (alignment) for STEMI/AMI, VT, AF, AVB
  * Sanity checks (random/zero/consistency)

python D:\Project\ECG\ecg_explainability.py --model_dir "D:\Project\ECG\ECGFounder" --checkpoint "D:\Project\ECG\ECGFounder\posttrain_all71\checkpoint_reg_e029.pth" --labels_file "D:\Project\ECG\ECGFounder\posttrain_all71\labels_all71.json" --ecg_samples "D:\Project\ECG\Dataset\PTB-XL\records500" --out_dir "D:\Project\ECG\ECGFounder\explainability_results" --device cuda --run_sanity_checks --seg_window_sec 2.0 --seg_stride_ratio 0.5 --seg_topk 3 --calibration_file "D:\Project\ECG\ECGFounder\posttrain_all71\calibration_temp.json" --min_conf_for_explain 0.3 --max_samples 30

"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wfdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def sigmoid_with_temp(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    return torch.sigmoid(logits / max(T, 1e-6))

def softclip(x, a=1e-8):
    return np.maximum(x, a)

class GradCAM1D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        def fwd_hook(module, inp, out):
            self.activations = out.detach()
        def bwd_hook(module, gin, gout):
            self.gradients = gout[0].detach()
        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)
    def generate_cam(self, input_tensor: torch.Tensor, target_class: int,
                     smooth_sigma: float = 10.0) -> np.ndarray:
        self.model.eval()
        out = self.model(input_tensor)
        if out.dim() == 2: out = out[0]
        self.model.zero_grad()
        out[target_class].backward()
        grads = self.gradients[0]
        acts  = self.activations[0]
        weights = grads.mean(dim=1)
        cam = (weights.unsqueeze(1) * acts).sum(dim=0)
        cam = F.relu(cam).cpu().numpy()
        L_feat = len(cam); L = input_tensor.shape[-1]
        if L_feat < L:
            cam = np.interp(np.linspace(0, L_feat-1, L), np.arange(L_feat), cam)
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (softclip(cam.max() - cam.min()))
        if smooth_sigma and smooth_sigma > 0:
            cam = gaussian_filter1d(cam, sigma=smooth_sigma)
        return cam

class IntegratedGradients:
    def __init__(self, model):
        self.model = model
    def generate_attributions(self, input_tensor: torch.Tensor, target_class: int, steps: int = 50) -> np.ndarray:
        self.model.eval()
        baseline = torch.zeros_like(input_tensor)
        scaled_inputs = [baseline + (float(i)/steps)*(input_tensor - baseline) for i in range(steps+1)]
        scaled_inputs = torch.cat(scaled_inputs, dim=0)
        scaled_inputs.requires_grad_(True)
        outputs = self.model(scaled_inputs)
        target_outputs = outputs[:, target_class]
        grads = torch.autograd.grad(
            outputs=target_outputs,
            inputs=scaled_inputs,
            grad_outputs=torch.ones_like(target_outputs),
            create_graph=False
        )[0]
        avg_grads = grads.mean(dim=0)
        ig = (input_tensor[0] - baseline[0]) * avg_grads
        return ig.detach().cpu().numpy()

def load_model_and_checkpoint(model_dir: str, checkpoint_path: str, n_classes: int, device: torch.device):
    sys.path.insert(0, str(Path(model_dir)))
    import net1d
    model = net1d.Net1D(
        in_channels=12,
        base_filters=64,
        ratio=1,
        filter_list=[64,160,160,400,400,1024,1024],
        m_blocks_list=[2,2,2,3,3,4,4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        n_classes=n_classes,
        use_bn=True,
        use_do=True,
        return_features=False,
        verbose=False
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    target_layer = model.stage_list[-1].block_list[-1].conv3
    return model, target_layer

def preprocess_ecg(ecg_path: str, target_length: int = 5000) -> np.ndarray:
    sig, fields = wfdb.rdsamp(ecg_path)
    x = sig.T.astype(np.float32)
    L = x.shape[1]
    if L >= target_length: x = x[:, :target_length]
    else: x = np.pad(x, ((0,0),(0, target_length - L)), mode='constant')
    mu, std = x.mean(), x.std() + 1e-8
    x = (x - mu) / std
    return x

def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(x).unsqueeze(0).to(device)

def detect_r_peaks(lead_signal: np.ndarray, sr: int = 500) -> np.ndarray:
    sig = gaussian_filter1d(lead_signal, sigma=2)
    peaks, _ = find_peaks(sig, distance=int(0.25*sr))
    return peaks

def estimate_qrs_width(lead_signal: np.ndarray, peaks: np.ndarray, sr: int = 500) -> float:
    sig = gaussian_filter1d(lead_signal, sigma=2)
    widths = []
    for p in peaks:
        left = max(0, p - int(0.12*sr))
        right = min(len(sig)-1, p + int(0.12*sr))
        seg = sig[left:right]
        thr = 0.5 * np.max(np.abs(seg)) if np.max(np.abs(seg)) > 0 else 0.0
        mask = np.where(np.abs(seg) >= thr)[0]
        if len(mask) > 1:
            w = (mask[-1] - mask[0]) / sr
            widths.append(w)
    return float(np.mean(widths)) if widths else 0.0

def rr_irregularity(peaks: np.ndarray, sr: int = 500) -> float:
    if len(peaks) < 3: return 0.0
    rr = np.diff(peaks) / sr
    if rr.mean() <= 1e-6: return 0.0
    return float(np.std(rr) / (np.mean(rr) + 1e-6))

def st_focus_score(cam: np.ndarray, peaks: np.ndarray, sr: int = 500, st_ms: Tuple[int,int]=(80,160)) -> float:
    if len(peaks) == 0: return 0.0
    L = len(cam); a = int(st_ms[0]/1000*sr); b = int(st_ms[1]/1000*sr)
    mask = np.zeros(L, dtype=bool)
    for p in peaks:
        s = min(max(0, p + a), L-1)
        e = min(max(0, p + b), L-1)
        if s < e: mask[s:e] = True
    inside = cam[mask].sum()
    total = cam.sum() + 1e-8
    return float(inside / total)

def plot_gradcam_overlay(ecg_signal: np.ndarray, cam_map: np.ndarray, lead_names: List[str],
                         title: str, save_path: str, sample_rate: int = 500):
    fig, axes = plt.subplots(12, 1, figsize=(16, 22), sharex=True)
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.995)
    t = np.arange(ecg_signal.shape[1]) / sample_rate
    for i, ax in enumerate(axes):
        ax.plot(t, ecg_signal[i], 'k-', linewidth=1.0, zorder=3)
        cam_norm = cam_map / (cam_map.max() + 1e-8)
        extent = [t[0], t[-1], ecg_signal[i].min()-0.2, ecg_signal[i].max()+0.2]
        cam_2d = np.tile(cam_norm, (20, 1))
        im = ax.imshow(cam_2d, aspect='auto', extent=extent, cmap='YlOrRd', alpha=0.5, origin='lower',
                       vmin=0, vmax=1, zorder=1)
        ax.set_ylabel(lead_names[i], fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.3, alpha=0.5)
    axes[-1].set_xlabel('Time (s)', fontsize=13, fontweight='bold')
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', pad=0.01, fraction=0.015)
    cbar.set_label('Model Attention (Grad-CAM)', fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

def sliding_windows(signal_len: int, window_len: int, stride: int) -> List[Tuple[int,int]]:
    if signal_len <= window_len: return [(0, signal_len)]
    starts = list(range(0, signal_len - window_len + 1, stride))
    return [(s, s + window_len) for s in starts]

def export_segments_and_influence(model, device, labels: List[str], ecg_paths: List[str],
                                  out_dir: str, window_sec: float = 2.0, stride_ratio: float = 0.5,
                                  sr: int = 500, topk: int = 3, T: float = 1.0):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    seg_len = int(window_sec * sr); stride = int(seg_len * stride_ratio)
    seg_rows = []; infl_rows = []
    for path in ecg_paths:
        x = preprocess_ecg(path)
        xt = to_tensor(x, device)
        with torch.no_grad():
            probs_full = sigmoid_with_temp(model(xt), T=T)[0].cpu().numpy()
        pc = int(np.argmax(probs_full))
        windows = sliding_windows(x.shape[1], seg_len, stride)
        scored = []
        for (s, e) in windows:
            xw = to_tensor(x[:, s:e], device)
            with torch.no_grad():
                pw = sigmoid_with_temp(model(xw), T=T)[0].cpu().numpy()
            scored.append((s, e, pw[pc], pw))
        scored.sort(key=lambda t: t[2], reverse=True)
        for s, e, pwpc, pw in scored[:topk]:
            seg_rows.append({
                "path": path, "start_sample": s, "end_sample": e,
                "top_class": labels[pc], "top_prob": float(pwpc),
                "probs_json": json.dumps({labels[j]: float(pw[j]) for j in range(len(labels))})
            })
        xt.requires_grad_(True)
        logits = model(xt)
        cls_logit = logits[0, pc]
        model.zero_grad()
        cls_logit.backward()
        grad = xt.grad.detach().abs().mean(dim=2)[0].cpu().numpy()
        for lead_idx in range(12):
            infl_rows.append({
                "path": path, "class": labels[pc],
                "lead_index": lead_idx, "lead_influence": float(grad[lead_idx])
            })
    seg_csv = os.path.join(out_dir, "explain_segments_topk.csv")
    infl_csv = os.path.join(out_dir, "lead_influence_scores.csv")
    pd.DataFrame(seg_rows).to_csv(seg_csv, index=False)
    pd.DataFrame(infl_rows).to_csv(infl_csv, index=False)
    print(f"Saved: {seg_csv}\nSaved: {infl_csv}")

LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
GROUPS = {
    "anterior": [6,7,8,9],
    "lateral":  [0,4,10,11],
    "inferior": [1,2,5],
}
AMI_ALIASES = {"STEMI","AMI","ASMI","ALMI","ILMI","IMI","IPLMI","INJAS","INJAL","INJIL","INJIN","INJLA","STE_"}
VT_ALIASES  = {"VT","VTA","VTAC","TVT"}
AF_ALIASES  = {"AF","AFIB","AFLT"}
AVB_ALIASES = {"1AVB","2AVB","3AVB"}

def lead_group_alignment(influence: np.ndarray, group: str) -> float:
    idxs = GROUPS[group]
    total = influence.sum() + 1e-8
    return float(influence[idxs].sum() / total)

def clinical_eval_record(model, device, labels: List[str], path: str, T: float = 1.0) -> Dict:
    x = preprocess_ecg(path)
    xt = to_tensor(x, device)
    with torch.no_grad():
        probs = sigmoid_with_temp(model(xt), T=T)[0].cpu().numpy()
    top_idx = int(np.argmax(probs)); top_label = labels[top_idx]; top_prob = float(probs[top_idx])
    target_layer = model.stage_list[-1].block_list[-1].conv3
    cam_engine = GradCAM1D(model, target_layer)
    cam = cam_engine.generate_cam(xt, top_idx, smooth_sigma=8.0)
    lead_II = x[1]; lead_V2 = x[7]
    peaks_II = detect_r_peaks(lead_II, sr=500)
    peaks_V2 = detect_r_peaks(lead_V2, sr=500)
    rr_cv = max(rr_irregularity(peaks_II, sr=500), rr_irregularity(peaks_V2, sr=500))
    qrs_w = max(estimate_qrs_width(lead_II, peaks_II, sr=500), estimate_qrs_width(lead_V2, peaks_V2, sr=500))
    st_score = max(st_focus_score(cam, peaks_II, 500), st_focus_score(cam, peaks_V2, 500))
    xt.requires_grad_(True)
    logits = model(xt)
    cls_logit = logits[0, top_idx]
    model.zero_grad(); cls_logit.backward()
    infl = xt.grad.detach().abs().mean(dim=2)[0].cpu().numpy()
    verdicts = []
    details = {}
    if top_label in AMI_ALIASES:
        ant = lead_group_alignment(infl, "anterior")
        lat = lead_group_alignment(infl, "lateral")
        inf = lead_group_alignment(infl, "inferior")
        details["ami_alignment"] = {"anterior": ant, "lateral": lat, "inferior": inf, "st_focus": st_score}
        verdicts.append("AMI_alignment_ok" if st_score >= 0.2 and max(ant,lat,inf) >= 0.3 else "AMI_alignment_weak")
    if top_label in VT_ALIASES:
        vt_cam_qrs_focus = float(np.mean([cam[max(p-30,0):min(p+30,len(cam))].mean() for p in peaks_V2]) if len(peaks_V2)>0 else 0.0)
        details["vt_width_sec"] = qrs_w
        details["vt_cam_qrs_focus"] = vt_cam_qrs_focus
        verdicts.append("VT_alignment_ok" if qrs_w >= 0.12 and vt_cam_qrs_focus >= 0.3 else "VT_alignment_weak")
    if top_label in AF_ALIASES:
        details["af_rr_cv"] = rr_cv
        verdicts.append("AF_alignment_ok" if rr_cv >= 0.15 else "AF_alignment_weak")
    if top_label in AVB_ALIASES:
        if len(peaks_II)>1:
            intervals = []
            for i in range(len(peaks_II)-1):
                mid = (peaks_II[i]+peaks_II[i+1])//2
                intervals.append(cam[peaks_II[i]:mid].mean())
            mean_between = float(np.mean(intervals)) if intervals else 0.0
        else:
            mean_between = 0.0
        details["avb_between_rr_cam"] = mean_between
        verdicts.append("AVB_alignment_ok" if mean_between >= 0.3 else "AVB_alignment_weak")
    report = {
        "path": path,
        "top_label": top_label,
        "top_prob": top_prob,
        "rr_cv": rr_cv,
        "qrs_width_sec": qrs_w,
        "st_focus": st_score,
        "lead_influence": {LEAD_NAMES[i]: float(infl[i]) for i in range(12)},
        "verdicts": verdicts,
        "details": details
    }
    return report

def run_sanity_checks(model, device, out_dir, T: float = 1.0):
    sanity_dir = os.path.join(out_dir, "sanity_checks"); Path(sanity_dir).mkdir(parents=True, exist_ok=True)
    results = {}
    target_layer = model.stage_list[-1].block_list[-1].conv3
    gradcam = GradCAM1D(model, target_layer)
    rand = torch.randn(1,12,5000).to(device)
    cam_r = gradcam.generate_cam(rand, target_class=0)
    results["random_variance"] = float(np.var(cam_r))
    zero = torch.zeros(1,12,5000).to(device)
    cam_z = gradcam.generate_cam(zero, target_class=0)
    results["zero_variance"] = float(np.var(cam_z))
    cams = [gradcam.generate_cam(rand, target_class=0) for _ in range(5)]
    results["consistency_std"] = float(np.std(np.stack(cams), axis=0).mean())
    with open(os.path.join(sanity_dir, "sanity_check_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Sanity checks:", results)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--labels_file", required=True)
    parser.add_argument("--ecg_samples", required=True, help="Folder under PTB-XL records500")
    parser.add_argument("--out_dir", default="explainability_results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", type=int, default=30)
    parser.add_argument("--run_sanity_checks", action="store_true", default=True)
    parser.add_argument("--run_ig", action="store_true", default=False)
    parser.add_argument("--seg_window_sec", type=float, default=2.0)
    parser.add_argument("--seg_stride_ratio", type=float, default=0.5)
    parser.add_argument("--seg_topk", type=int, default=3)
    parser.add_argument("--min_conf_for_explain", type=float, default=0.3)
    parser.add_argument("--calibration_file", type=str, default=None, help="Path to calibration_temp.json with {'temperature': T}")
    parser.add_argument("--temperature", type=float, default=1.0, help="Override temperature if no file")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open(args.labels_file, 'r', encoding='utf-8') as f:
        labels = json.load(f)['labels']

    T = args.temperature
    if args.calibration_file and os.path.exists(args.calibration_file):
        with open(args.calibration_file,'r') as f:
            T = float(json.load(f).get("temperature", T))
    print(f"Using temperature T={T:.3f}")

    ecg_files = sorted([p for p in Path(args.ecg_samples).rglob("*.hea")])[:args.max_samples]
    ecg_paths = [str(p).replace(".hea", "") for p in ecg_files]
    print(f"Found {len(ecg_paths)} ECGs")

    model, target_layer = load_model_and_checkpoint(args.model_dir, args.checkpoint, len(labels), device)
    gradcam = GradCAM1D(model, target_layer)
    ig = IntegratedGradients(model) if args.run_ig else None

    if args.run_sanity_checks:
        run_sanity_checks(model, device, args.out_dir, T=T)

    lead_names = LEAD_NAMES
    clinical_reports = []
    for path in ecg_paths:
        try:
            x = preprocess_ecg(path)
            xt = to_tensor(x, device)
            with torch.no_grad():
                out = model(xt)
                probs = sigmoid_with_temp(out, T=T)[0].cpu().numpy()
            top5_idx = np.argsort(probs)[-5:][::-1]
            top5_labels = [labels[i] for i in top5_idx]
            top5_probs  = [float(probs[i]) for i in top5_idx]
            print(f"{Path(path).name}: {list(zip(top5_labels, [round(p,3) for p in top5_probs]))}")
            explain_targets = [(top5_labels[0], int(top5_idx[0]))]
            for lab in ["STEMI","AMI","ASMI","ALMI","ILMI","IMI","VT","AF","3AVB","1AVB","LAFB","RBBB","LBBB","STE_"]:
                if lab in labels:
                    idx = labels.index(lab)
                    if probs[idx] >= args.min_conf_for_explain and not any(t[1]==idx for t in explain_targets):
                        explain_targets.append((lab, idx))
            for diag, idx_cls in explain_targets:
                cam = gradcam.generate_cam(xt, idx_cls, smooth_sigma=8.0)
                plot_gradcam_overlay(x, cam, lead_names, f"{diag} (full, T={T:.2f})",
                                     os.path.join(args.out_dir, f"{Path(path).stem}_{diag}_gradcam_full.png"))
                if ig is not None:
                    ig_attr = ig.generate_attributions(xt, idx_cls)
                    # Optional: save IG visualization similarly if needed
            export_segments_and_influence(
                model=model, device=device, labels=labels, ecg_paths=[path],
                out_dir=args.out_dir, window_sec=args.seg_window_sec,
                stride_ratio=args.seg_stride_ratio, sr=500, topk=args.seg_topk, T=T
            )
            report = clinical_eval_record(model, device, labels, path, T=T)
            clinical_reports.append(report)
            with open(os.path.join(args.out_dir, f"{Path(path).stem}_clinical_report.json"), "w") as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            print(f"Error on {path}: {e}")
            continue
    with open(os.path.join(args.out_dir, "clinical_reports_summary.json"), "w") as f:
        json.dump({"reports": clinical_reports}, f, indent=2)

if __name__ == "__main__":
    main()
