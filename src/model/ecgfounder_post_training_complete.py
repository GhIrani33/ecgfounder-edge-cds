# -*- coding: utf-8 -*-
"""
ecgfounder_post_training_complete.py
Robust post-validation pipeline for ECGFounder on PTB-XL:
- Imports Net1D from net1d.py located near the checkpoint automatically
- Builds PTB-XL multilabel tasks (all71/diag44/sub23/rhythm12) with standard strat_fold split
- Baseline eval, Initialization (linear probing with frozen backbone), Regularization (full finetune with cosine + dropout)
- Saves detailed logs, per-epoch metrics, and checkpoints
"""

import os
import sys
import ast
import json
import time
import math
import copy
import argparse
import logging
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import wfdb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score, average_precision_score

# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False, warn_only=True)

# ----------------------------
# Logging
# ----------------------------
def setup_logger(out_dir: str, level=logging.INFO):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(out_dir, f"run_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()],
    )
    logging.info(f"Logging to {log_file}")
    return log_file

# ----------------------------
# Config loader (input length, rate, leads)
# ----------------------------
def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    input_size = int(cfg.get("input_size", 5000))
    sr = cfg.get("sampling_rate", "500Hz")
    if isinstance(sr, str) and sr.endswith("Hz"):
        sr = int(sr.replace("Hz", ""))
    lead_num = cfg.get("lead_num", "12/1")
    if isinstance(lead_num, str):
        lead_num = int(lead_num.split("/")[0])
    return input_size, sr, lead_num

# ----------------------------
# PTB-XL utilities
# ----------------------------
def parse_scp_codes(s: str) -> Dict[str, float]:
    try:
        d = ast.literal_eval(s)
        if isinstance(d, dict):
            return d
    except Exception:
        pass
    return {}

def build_label_sets(ptbxl_root: str, expected: Dict[str, int]) -> Dict[str, List[str]]:
    scp_df = pd.read_csv(os.path.join(ptbxl_root, "scp_statements.csv"))
    label_sets = {}

    # rhythm12
    if "rhythm" in scp_df.columns:
        rhythm = scp_df.loc[scp_df["rhythm"] == 1, "Unnamed: 0"].astype(str).tolist()
    else:
        rhythm = []
    label_sets["rhythm12"] = sorted(rhythm)

    # diag44 and subclass23
    diag_mask = scp_df["diagnostic"] == 1 if "diagnostic" in scp_df.columns else pd.Series([False]*len(scp_df))
    diag_statements = scp_df.loc[diag_mask, "Unnamed: 0"].astype(str).tolist()
    label_sets["diag44"] = sorted(diag_statements)
    if "diagnostic_subclass" in scp_df.columns:
        diag_sub = scp_df.loc[diag_mask, "diagnostic_subclass"].dropna().astype(str).unique().tolist()
    else:
        diag_sub = []
    label_sets["sub23"] = sorted(diag_sub)

    # all71: fallback union with optional frequency threshold if provided
    if "freq" in scp_df.columns:
        common = scp_df.loc[scp_df["freq"] >= 30, "Unnamed: 0"].astype(str).tolist()
    else:
        cols = [c for c in ["diagnostic","rhythm","form","infarct","axis","conduction"] if c in scp_df.columns]
        mask = None
        for c in cols:
            mask = (scp_df[c] == 1) if mask is None else (mask | (scp_df[c] == 1))
        if mask is None:
            common = scp_df["Unnamed: 0"].astype(str).tolist()
        else:
            common = scp_df.loc[mask, "Unnamed: 0"].astype(str).tolist()
    label_sets["all71"] = sorted(common)

    # Adjust sizes to expected when overshooting
    for k, target in expected.items():
        if k in label_sets and target is not None and len(label_sets[k]) > target:
            label_sets[k] = label_sets[k][:target]
    return label_sets

class PTBXL_Multilabel(Dataset):
    def __init__(self, root: str, split: str, task: str, label_map: Dict[str, List[str]],
                 input_size: int = 5000, sr: int = 500, zscore_per_lead: bool = False):
        self.root = root
        self.split = split  # train/val/test
        self.task = task
        self.label_map = label_map
        self.labels = label_map[task]  # List[str]
        self.input_size = input_size
        self.sr = sr
        self.zscore_per_lead = zscore_per_lead

        self.db = pd.read_csv(os.path.join(root, "ptbxl_database.csv"))
        self.db["scp_codes"] = self.db["scp_codes"].apply(parse_scp_codes)

        # Standard PTB-XL split via strat_fold
        if "strat_fold" not in self.db.columns:
            raise RuntimeError("ptbxl_database.csv missing 'strat_fold' column")
        if split == "train":
            folds = list(range(1, 9))
        elif split == "val":
            folds = [9]
        else:
            folds = [10]
        self.df = self.db[self.db["strat_fold"].isin(folds)].copy().reset_index(drop=True)

        # Build label matrix
        scp_df = pd.read_csv(os.path.join(root, "scp_statements.csv"))
        statement_to_subclass = dict(zip(scp_df["Unnamed: 0"], scp_df.get("diagnostic_subclass", pd.Series([np.nan]*len(scp_df)))))
        Y = []
        for _, row in self.df.iterrows():
            scp = row["scp_codes"]
            if self.task in ["all71", "diag44", "rhythm12"]:
                labels_present = [k for k in scp.keys() if k in self.labels]
            elif self.task == "sub23":
                present_sub = set()
                for k in scp.keys():
                    sub = statement_to_subclass.get(k, np.nan)
                    if isinstance(sub, str) and sub in self.labels:
                        present_sub.add(sub)
                labels_present = list(present_sub)
            else:
                raise ValueError(f"Unknown task: {self.task}")
            y = np.zeros(len(self.labels), dtype=np.float32)
            for lab in labels_present:
                y[self.labels.index(lab)] = 1.0
            Y.append(y)
        self.Y = np.stack(Y, axis=0).astype(np.float32)

        # Resolve high-resolution 500Hz file paths
        if "filename_hr" in self.df.columns:
            self.paths = self.df["filename_hr"].tolist()
        else:
            raise RuntimeError("ptbxl_database.csv missing 'filename_hr' column; please ensure PTB-XL 500Hz paths are present")

    def __len__(self):
        return len(self.df)

    def _zscore(self, x: np.ndarray):
        if self.zscore_per_lead:
            mu = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True) + 1e-8
            return (x - mu) / std
        mu = x.mean()
        std = x.std() + 1e-8
        return (x - mu) / std

    def __getitem__(self, idx):
        rel = self.paths[idx]
        rec_path = os.path.join(self.root, rel)
        sig, meta = wfdb.rdsamp(rec_path)
        x = sig.T.astype(np.float32)  # (leads, length)
        # Trim/pad to input_size
        L = x.shape[1]
        if L >= self.input_size:
            x = x[:, :self.input_size]
        else:
            pad = self.input_size - L
            x = np.pad(x, ((0,0), (0, pad)), mode='constant')
        x = self._zscore(x)
        return torch.from_numpy(x), torch.from_numpy(self.Y[idx])

# ----------------------------
# Model import and setup
# ----------------------------
def import_model_class(model_dir: Optional[str] = None):
    # Add model_dir (where net1d.py resides) to sys.path and import Net1D
    if model_dir is not None:
        sys.path.insert(0, str(Path(model_dir)))
    try:
        import net1d  # noqa
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"Cannot import net1d from model_dir={model_dir}; ensure net1d.py is accessible") from e
    if hasattr(net1d, "Net1D"):
        return net1d.Net1D
    # Fallback: try to find a class with .first_conv and .stage_list attributes
    for name in dir(net1d):
        obj = getattr(net1d, name)
        if isinstance(obj, type) and hasattr(obj, "__init__"):
            return obj
    raise ImportError("No suitable model class found in net1d.py")

def build_ecgfounder_net1d(ModelClass, in_channels: int, pretrain_classes: int = 150):
    """
    Instantiate Net1D with the canonical ECGFounder backbone hyperparameters so checkpoint weights align.
    """
    # According to net1d.py main, these are the canonical args:
    kwargs = dict(
        in_channels=in_channels,
        base_filters=64,
        ratio=1,
        filter_list=[64,160,160,400,400,1024,1024],
        m_blocks_list=[2,2,2,3,3,4,4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        n_classes=pretrain_classes,
        use_bn=True,
        use_do=True,
        return_features=False,
        verbose=False,
    )
    return ModelClass(**kwargs)

def load_pretrained_checkpoint(model: nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected

def replace_classification_head(model: nn.Module, num_classes: int) -> nn.Module:
    # Replace the last nn.Linear with new out_features=num_classes
    last_linear_name = None
    last_linear = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_linear_name = name
            last_linear = module
    if last_linear is None:
        raise RuntimeError("No Linear head found to replace")
    in_features = last_linear.in_features
    new_head = nn.Linear(in_features, num_classes)
    # Set attribute on parent
    parts = last_linear_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_head)
    return model

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.floor()
        return x.div(keep_prob) * random_tensor

def try_set_dropout(model: nn.Module, p: float):
    # Set all nn.Dropout rates if present
    def set_p(m):
        if isinstance(m, nn.Dropout):
            m.p = p
    model.apply(set_p)

# ----------------------------
# Metrics
# ----------------------------
@torch.no_grad()
def evaluate(model, loader, device, task_labels: List[str]) -> Dict[str, float]:
    model.eval()
    Ys = []
    Ps = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)  # [B, C, L]
        yb = yb.to(device, non_blocking=True).float()  # [B, K]
        logits = model(xb)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        Ps.append(probs)
        Ys.append(yb.cpu().numpy())
    Y = np.concatenate(Ys, axis=0)  # [N, K]
    P = np.concatenate(Ps, axis=0)  # [N, K]

    # Per-class metrics with robust nan handling for classes without positives/negatives
    per_class_auroc = []
    per_class_auprc = []
    for k in range(Y.shape[1]):
        yk = Y[:, k]
        pk = P[:, k]
        if yk.max() == yk.min():
            auroc_k = np.nan
            auprc_k = np.nan
        else:
            try:
                auroc_k = roc_auc_score(yk, pk)
            except Exception:
                auroc_k = np.nan
            try:
                auprc_k = average_precision_score(yk, pk)
            except Exception:
                auprc_k = np.nan
        per_class_auroc.append(auroc_k)
        per_class_auprc.append(auprc_k)
    macro_auroc = float(np.nanmean(per_class_auroc))
    macro_auprc = float(np.nanmean(per_class_auprc))

    return {
        "macro_auroc": macro_auroc,
        "macro_auprc": macro_auprc,
        "per_class_auroc": {lab: (None if np.isnan(v) else float(v)) for lab, v in zip(task_labels, per_class_auroc)},
        "per_class_auprc": {lab: (None if np.isnan(v) else float(v)) for lab, v in zip(task_labels, per_class_auprc)},
    }

# ----------------------------
# Training
# ----------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, grad_accum=1, max_norm=None):
    model.train()
    running_loss = 0.0
    n = 0
    optimizer.zero_grad(set_to_none=True)
    for step, (xb, yb) in enumerate(loader):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).float()
        logits = model(xb)
        loss = criterion(logits, yb)
        (loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0:
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        running_loss += loss.item() * xb.size(0)
        n += xb.size(0)
    return running_loss / max(n, 1)

def get_pos_weight(Y: np.ndarray):
    pos = Y.sum(axis=0)
    neg = Y.shape[0] - pos
    pw = (neg / np.maximum(pos, 1.0)).astype(np.float32)
    return torch.from_numpy(pw)

def save_checkpoint(out_dir, tag, model, optimizer, scheduler, epoch, val_metrics):
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "val_metrics": val_metrics,
    }
    path = os.path.join(out_dir, f"checkpoint_{tag}_e{epoch:03d}.pth")
    torch.save(state, path)
    logging.info(f"Saved checkpoint: {path}")
    return path

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptbxl_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task", type=str, default="all71", choices=["all71", "diag44", "sub23", "rhythm12"])
    parser.add_argument("--out_dir", type=str, default="outputs_posttrain")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--zscore_per_lead", action="store_true")
    parser.add_argument("--model_dir", type=str, default=None, help="Optional path to directory containing net1d.py; by default inferred from checkpoint")

    # Initialization stage (linear probing)
    parser.add_argument("--init_epochs", type=int, default=30)
    parser.add_argument("--init_lr", type=float, default=5e-3)
    parser.add_argument("--init_wd", type=float, default=0.1)

    # Regularization stage (full finetune)
    parser.add_argument("--reg_epochs", type=int, default=30)
    parser.add_argument("--reg_lr", type=float, default=5e-4)
    parser.add_argument("--reg_min_lr", type=float, default=1e-5)
    parser.add_argument("--reg_wd", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--droppath", type=float, default=0.5)  # advisory; requires source patch in net1d

    args = parser.parse_args()
    set_seed(args.seed)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    setup_logger(args.out_dir)

    # Load config
    input_size, sr, lead_num = load_config(args.config)
    logging.info(f"Config -> input_size={input_size}, sr={sr}, leads={lead_num}")

    # Build labels and datasets
    expected_sizes = {"all71": 71, "diag44": 44, "sub23": 23, "rhythm12": 12}
    label_sets = build_label_sets(args.ptbxl_root, expected=expected_sizes)
    labels = label_sets[args.task]
    logging.info(f"Task={args.task}, #labels={len(labels)}; labels(head)={labels[:10]}")

    ds_train = PTBXL_Multilabel(args.ptbxl_root, "train", args.task, label_sets, input_size, sr, args.zscore_per_lead)
    ds_val = PTBXL_Multilabel(args.ptbxl_root, "val", args.task, label_sets, input_size, sr, args.zscore_per_lead)
    ds_test = PTBXL_Multilabel(args.ptbxl_root, "test", args.task, label_sets, input_size, sr, args.zscore_per_lead)

    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Infer model_dir from checkpoint if not provided (…/ECGFounder/checkpoint/ -> …/ECGFounder)
    if args.model_dir is None:
        ckpt_dir = Path(args.checkpoint).resolve().parent
        args.model_dir = str(ckpt_dir.parent)
    logging.info(f"Model dir inferred: {args.model_dir}")

    # Import and build model
    ModelClass = import_model_class(args.model_dir)
    model = build_ecgfounder_net1d(ModelClass, in_channels=lead_num, pretrain_classes=150)

    # Load pretrained checkpoint (strict=False to allow head size mismatch)
    missing, unexpected = load_pretrained_checkpoint(model, args.checkpoint)
    logging.info(f"Checkpoint loaded: missing={len(missing)} keys, unexpected={len(unexpected)} keys")

    # Replace classification head for this PTB-XL task
    model = replace_classification_head(model, num_classes=len(labels))

    # Prepare device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Optionally reduce Dropout p to paper setting
    try_set_dropout(model, args.dropout)

    # Baseline eval (frozen model with new head; for transparency, freeze all and eval random head)
    for p in model.parameters():
        p.requires_grad = False
    base_val = evaluate(model, dl_val, device, labels)
    base_test = evaluate(model, dl_test, device, labels)
    logging.info(f"BASELINE | val AUROC={base_val['macro_auroc']:.4f} AUPRC={base_val['macro_auprc']:.4f}")
    logging.info(f"BASELINE | test AUROC={base_test['macro_auroc']:.4f} AUPRC={base_test['macro_auprc']:.4f}")
    with open(os.path.join(args.out_dir, f"metrics_baseline_val.json"), "w", encoding="utf-8") as f:
        json.dump(base_val, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, f"metrics_baseline_test.json"), "w", encoding="utf-8") as f:
        json.dump(base_test, f, ensure_ascii=False, indent=2)

    # ----------------------------
    # Stage 1: Initialization (linear probing on head)
    # ----------------------------
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze last Linear only
    head = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            head = module
    for p in head.parameters():
        p.requires_grad = True

    pos_weight = get_pos_weight(ds_train.Y).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(head.parameters(), lr=args.init_lr, weight_decay=args.init_wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, verbose=True)

    best_val = -np.inf
    best_path = None
    for epoch in range(1, args.init_epochs + 1):
        loss = train_one_epoch(model, dl_train, optimizer, criterion, device)
        val_metrics = evaluate(model, dl_val, device, labels)
        scheduler.step(val_metrics["macro_auroc"])
        logging.info(f"[INIT] epoch {epoch:03d} | loss={loss:.5f} | val AUROC={val_metrics['macro_auroc']:.4f} AUPRC={val_metrics['macro_auprc']:.4f}")
        with open(os.path.join(args.out_dir, f"metrics_init_e{epoch:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(val_metrics, f, ensure_ascii=False, indent=2)
        if val_metrics["macro_auroc"] > best_val:
            best_val = val_metrics["macro_auroc"]
            best_path = save_checkpoint(args.out_dir, "init", model, optimizer, scheduler, epoch, val_metrics)
    logging.info(f"[INIT] best val AUROC={best_val:.4f} | ckpt={best_path}")

    # ----------------------------
    # Stage 2: Regularization (full fine-tuning)
    # Note: For exact DropPath on residual add, patch net1d.BasicBlock forward: out = DropPath(out) + identity
    # ----------------------------
    for p in model.parameters():
        p.requires_grad = True
    try_set_dropout(model, args.dropout)

    optimizer = optim.AdamW(model.parameters(), lr=args.reg_lr, weight_decay=args.reg_wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.reg_epochs, eta_min=args.reg_min_lr)

    best_val = -np.inf
    best_path = None
    for epoch in range(1, args.reg_epochs + 1):
        loss = train_one_epoch(model, dl_train, optimizer, criterion, device)
        val_metrics = evaluate(model, dl_val, device, labels)
        scheduler.step()
        logging.info(f"[REG ] epoch {epoch:03d} | loss={loss:.5f} | val AUROC={val_metrics['macro_auroc']:.4f} AUPRC={val_metrics['macro_auprc']:.4f}")
        with open(os.path.join(args.out_dir, f"metrics_reg_e{epoch:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(val_metrics, f, ensure_ascii=False, indent=2)
        if val_metrics["macro_auroc"] > best_val:
            best_val = val_metrics["macro_auroc"]
            best_path = save_checkpoint(args.out_dir, "reg", model, optimizer, scheduler, epoch, val_metrics)
    logging.info(f"[REG ] best val AUROC={best_val:.4f} | ckpt={best_path}")

    # Final test
    if best_path and os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=False)
    test_metrics = evaluate(model, dl_test, device, labels)
    with open(os.path.join(args.out_dir, f"metrics_test_final.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)
    logging.info(f"[TEST] AUROC={test_metrics['macro_auroc']:.4f} AUPRC={test_metrics['macro_auprc']:.4f}")
    logging.info("Done.")

if __name__ == "__main__":
    main()