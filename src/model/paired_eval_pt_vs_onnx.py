

from pathlib import Path
import json, numpy as np, torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from scipy.stats import pearsonr, spearmanr
import onnxruntime
from onnxruntime import InferenceSession

# مسیرها
BASE = Path(r"D:\Project\ECG\ECGFounder")
CKPT_71 = BASE / r"posttrain_all71\checkpoint_reg_e029.pth"      # 71-class ckpt
LABELS_JSON = BASE / r"posttrain_all71\labels_all71.json"        # {"task":"all71","labels":[...]}
ONNX_PATH = BASE / r"onnx_models\ecg_founder_all71.onnx"         
CFG = BASE / r"checkpoint\config.json"
NET1D = BASE / "net1d.py"

# {"task":"all71","labels":[...]}
def load_label_list(p):
    with open(p,"r") as f: data=json.load(f)
    if isinstance(data,dict) and "labels" in data and isinstance(data["labels"],list):
        codes=list(data["labels"]); return codes,{c:i for i,c in enumerate(codes)}
    raise ValueError("labels_all71.json must contain {'labels':[...]}")


def export_onnx_from_ckpt(ckpt_path: Path, net1d_path: Path, cfg_path: Path, out_path: Path):
   
    from pytorch_to_onnx_converter import import_net1d, infer_n_classes, build_model, load_weights, export_to_onnx
    net1d = import_net1d(net1d_path)
    n_classes = infer_n_classes(ckpt_path) 
    model = build_model(net1d, n_classes)
    load_weights(model, ckpt_path)
    with open(cfg_path,"r") as f: input_size=int(json.load(f).get("input_size",5000))
    export_to_onnx(model, out_path, input_size=input_size, opset=14)
    return n_classes


import pandas as pd, wfdb
def zscore(x):
    mu=x.mean(axis=1,keepdims=True); sd=x.std(axis=1,keepdims=True)+1e-8; return (x-mu)/sd
def make_test_df(db_csv): 
    df=pd.read_csv(db_csv); return df[df["strat_fold"]==10].copy()
def iter_test(records_root: Path, test_df: pd.DataFrame, code_to_idx: dict, input_size=5000, bs=16):
    Xb,Yb=[],[]
    for _,row in test_df.iterrows():
        rel=row["filename_hr"]; sig,_=wfdb.rdsamp((records_root.parent/rel).as_posix())
        x=sig.T.astype(np.float32)
        if x.shape[1]>input_size: x=x[:,:input_size]
        elif x.shape[1]<input_size: x=np.pad(x,((0,0),(0,input_size-x.shape[1])),"edge")
        x=zscore(x); y=np.zeros(len(code_to_idx),np.float32)
        for k in (row["scp_codes"] if isinstance(row["scp_codes"],dict) else {}): 
            if k in code_to_idx: y[code_to_idx[k]]=1.0
        Xb.append(x); Yb.append(y)
        if len(Xb)==bs:
            yield np.stack(Xb,0), np.stack(Yb,0); Xb,Yb=[],[]
    if Xb: yield np.stack(Xb,0), np.stack(Yb,0)

def main():

    with open(CFG,"r") as f: input_size=int(json.load(f).get("input_size",5000))
    labels, code_to_idx = load_label_list(LABELS_JSON)


    need_export = True
    if ONNX_PATH.exists():
        try:
            sess = InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
            out = sess.get_outputs()[0]; onnx_dim = out.shape[1] if (len(out.shape)==2 and isinstance(out.shape[1],int)) else None
            if onnx_dim == len(labels): need_export = False
        except Exception: need_export = True
    if need_export:
        nc = export_onnx_from_ckpt(CKPT_71, NET1D, CFG, ONNX_PATH)
        if nc != len(labels):
            raise RuntimeError(f"Checkpoint classes {nc} != labels {len(labels)}; provide matching ckpt/labels.")


    sess = InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name; out_name = sess.get_outputs()[0].name


    db_csv = Path(r"D:\Project\ECG\Dataset\PTB-XL\ptbxl_database.csv")
    test_df = make_test_df(db_csv)
    # parse scp_codes به dict
    import ast
    test_df["scp_codes"] = test_df["scp_codes"].apply(lambda s: ast.literal_eval(s) if isinstance(s,str) else {})


    from pytorch_to_onnx_converter import import_net1d, build_model, load_weights, infer_n_classes
    net1d = import_net1d(NET1D); n_classes = infer_n_classes(CKPT_71)
    model = build_model(net1d, n_classes); load_weights(model, CKPT_71); model.eval()


    all_pt, all_ort, all_y = [], [], []
    records_root = Path(r"D:\Project\ECG\Dataset\PTB-XL\records500")
    with torch.no_grad():
        for X,Y in iter_test(records_root, test_df, code_to_idx, input_size=input_size, bs=16):
            pt = model(torch.from_numpy(X).float()).cpu().numpy()
            ort = sess.run([out_name], {in_name: X.astype(np.float32)})[0]
            all_pt.append(pt); all_ort.append(ort); all_y.append(Y)
    pt_logits = np.concatenate(all_pt,0); ort_logits=np.concatenate(all_ort,0); Y=np.concatenate(all_y,0)
    pt_probs = 1/(1+np.exp(-pt_logits)); ort_probs = 1/(1+np.exp(-ort_logits))


    pr = float(pearsonr(pt_logits.ravel(), ort_logits.ravel())[0]); sr = float(spearmanr(pt_logits.ravel(), ort_logits.ravel())[0])
    def ECE(P,Y,n_bins=15):
        edges=np.linspace(0,1,n_bins+1); e=[]
        for c in range(P.shape[1]):
            pc=P[:,c]; yc=Y[:,c]; s=0.0
            for i in range(n_bins):
                lo,hi=edges[i],edges[i+1]; m=(pc>=lo)&(pc<(hi if i<n_bins-1 else hi))
                if m.sum()==0: continue
                s+= m.mean()*abs((yc[m]>0.5).mean()-pc[m].mean())
            e.append(s)
        return float(np.mean(e))
    ece_pt=ECE(pt_probs,Y); ece_ort=ECE(ort_probs,Y)
    brier_pt=float(np.mean((pt_probs-Y)**2)); brier_ort=float(np.mean((ort_probs-Y)**2))
    def per_class(P,Y):
        C=Y.shape[1]; auroc=np.zeros(C); auprc=np.zeros(C)
        for c in range(C):
            yy=Y[:,c]; pp=P[:,c]; valid=(yy.max()!=yy.min())
            auroc[c]=roc_auc_score(yy,pp) if valid else np.nan
            auprc[c]=average_precision_score(yy,pp) if valid else np.nan
        return auroc, auprc
    auroc_pt, auprc_pt = per_class(pt_probs,Y); auroc_ort, auprc_ort = per_class(ort_probs,Y)
    macro = {"auroc_pt":float(np.nanmean(auroc_pt)),"auroc_ort":float(np.nanmean(auroc_ort)),
             "auprc_pt":float(np.nanmean(auprc_pt)),"auprc_ort":float(np.nanmean(auprc_ort))}
    micro = {"auroc_pt":float(roc_auc_score(Y.ravel(), pt_probs.ravel())),
             "auroc_ort":float(roc_auc_score(Y.ravel(), ort_probs.ravel())),
             "auprc_pt":float(average_precision_score(Y.ravel(), pt_probs.ravel())),
             "auprc_ort":float(average_precision_score(Y.ravel(), ort_probs.ravel()))}
    f1_pt=float(f1_score(Y.ravel(), (pt_probs.ravel()>=0.5).astype(int), average="macro", zero_division=0))
    f1_ort=float(f1_score(Y.ravel(), (ort_probs.ravel()>=0.5).astype(int), average="macro", zero_division=0))
    print("\n=== Paired Results (all71) ===")
    print(f"Logit corr: Pearson={pr:.4f} | Spearman={sr:.4f}")
    print(f"Macro AUROC: PT={macro['auroc_pt']:.4f} | ORT={macro['auroc_ort']:.4f} | Δ={macro['auroc_ort']-macro['auroc_pt']:+.4f}")
    print(f"Macro AUPRC: PT={macro['auprc_pt']:.4f} | ORT={macro['auprc_ort']:.4f} | Δ={macro['auprc_ort']-macro['auprc_pt']:+.4f}")
    print(f"Micro AUROC: PT={micro['auroc_pt']:.4f} | ORT={micro['auroc_ort']:.4f} | Δ={micro['auroc_ort']-micro['auroc_pt']:+.4f}")
    print(f"Micro AUPRC: PT={micro['auprc_pt']:.4f} | ORT={micro['auprc_ort']:.4f} | Δ={micro['auprc_ort']-micro['auprc_pt']:+.4f}")
    print(f"F1@0.5: PT={f1_pt:.4f} | ORT={f1_ort:.4f} | Δ={f1_ort-f1_pt:+.4f}")
    print(f"Calibration: ECE PT={ece_pt:.4f} | ORT={ece_ort:.4f} | Brier PT={brier_pt:.4f} | ORT={brier_ort:.4f}")

if __name__ == "__main__":
    main()
