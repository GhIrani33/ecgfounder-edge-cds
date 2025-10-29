# -*- coding: utf-8 -*-
"""
pytorch_to_onnx_converter.py 

⚠️ CRITICAL PREREQUISITE for all_close=True:
   Before running this script, patch net1d.py as follows:

1️⃣ Line 142 (class Swish):
   OLD: return x * torch.sigmoid(x)
   NEW: return torch.nn.functional.silu(x)

2️⃣ Line 271 (inside BasicBlock.forward, SE gating):
   OLD: out = torch.einsum('abc,ab->abc', out, se)
   NEW: out = out * se.unsqueeze(-1)

These two changes ensure PyTorch and ONNX use identical computational paths,
eliminating numerical differences caused by operator mapping.

- Export: opset=14, do_constant_folding=False
- Validate: fixed input, expects all_close=True (rtol=1e-5, atol=1e-5)
"""

import os
import sys
import json
from pathlib import Path
import warnings
import numpy as np
import torch
import onnx

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except Exception:
    ORT_AVAILABLE = False

warnings.filterwarnings("ignore")


def load_config(config_path: Path):
    """Load model configuration from JSON."""
    if config_path.exists():
        with open(config_path, "r") as f:
            cfg = json.load(f)
        return {
            "input_size": int(cfg.get("input_size", 5000)),
            "sampling_rate": cfg.get("sampling_rate", "500Hz"),
            "time": cfg.get("time", "10s"),
            "lead_num": cfg.get("lead_num", "12/1"),
        }
    return {"input_size": 5000, "sampling_rate": "500Hz", "time": "10s", "lead_num": "12/1"}


def infer_n_classes(ckpt_path: Path) -> int:
    """Infer number of classes from checkpoint dense layer."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if "dense.weight" in sd:
        return int(sd["dense.weight"].shape[0])
    raise RuntimeError("Unable to infer n_classes (dense.weight not found)")


def import_net1d(net1d_path: Path):
    """Import net1d module dynamically."""
    sys.path.insert(0, str(net1d_path.parent))
    import importlib
    net1d = importlib.import_module("net1d")
    if not hasattr(net1d, "Net1D"):
        raise AttributeError("net1d.py has no class 'Net1D'")
    return net1d


def build_model(net1d, n_classes: int):
    """Build ECGFounder Net1D model with standard 12-lead configuration."""
    model = net1d.Net1D(
        in_channels=12,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        n_classes=n_classes,
        use_bn=True,
        use_do=True,
        return_features=False,
        verbose=False,
    )
    # Disable stochastic depth if available
    if hasattr(net1d, "set_drop_path_rate"):
        try:
            net1d.set_drop_path_rate(0.0)
        except Exception:
            pass
    return model


def load_weights(model: torch.nn.Module, ckpt_path: Path):
    """Load pretrained weights from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected


def export_to_onnx(model, out_path: Path, input_size: int, opset: int = 14):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model in eval mode
        out_path: Output ONNX file path
        input_size: Sequence length (default 5000 for 10s @ 500Hz)
        opset: ONNX opset version (14 for Raspberry Pi 4 compatibility)
    """
    model.eval()
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    np.random.seed(0)

    dummy = torch.randn(1, 12, input_size, dtype=torch.float32)
    input_names = ["ecg_signal"]
    output_names = ["predictions"]
    dynamic_axes = {"ecg_signal": {0: "batch_size"}, "predictions": {0: "batch_size"}}

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=False,  # Critical: prevents algebraic reordering
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False,
    )
    
    # Validate ONNX graph
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    return out_path


def validate_exact(pytorch_model, onnx_path: Path, input_size: int, rtol=1e-5, atol=1e-5):
    """
    Validate numerical parity between PyTorch and ONNX Runtime.
    
    Returns:
        dict with keys: max_abs, mean_abs, all_close, rtol, atol
    """
    if not ORT_AVAILABLE:
        return {"skipped": True}
    
    torch.manual_seed(1234)
    np.random.seed(1234)
    x = torch.randn(1, 12, input_size, dtype=torch.float32)
    
    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        y_pt = pytorch_model(x).cpu().numpy()
    
    # ONNX Runtime inference
    sess = ort.InferenceSession(str(onnx_path))
    inp = sess.get_inputs()[0].name
    out = sess.get_outputs()[0].name
    y_ort = sess.run([out], {inp: x.numpy()})[0]
    
    # Compute differences
    diff = np.abs(y_pt - y_ort)
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "all_close": bool(np.allclose(y_pt, y_ort, rtol=rtol, atol=atol)),
        "rtol": rtol,
        "atol": atol,
    }


def main():
    print("\n" + "=" * 70)
    print("🚀 ECGFounder PyTorch → ONNX Conversion (Q1-ready)")
    print("=" * 70)
    
    # Configuration
    BASE_DIR = Path(r"D:\Project\ECG\ECGFounder")
    CKPT = BASE_DIR / "checkpoint" / "12_lead_ECGFounder.pth"
    CFG = BASE_DIR / "checkpoint" / "config.json"
    NET1D = BASE_DIR / "net1d.py"
    OUT_DIR = BASE_DIR / "onnx_models"
    OUT_DIR.mkdir(exist_ok=True)
    OUT_ONNX = OUT_DIR / "ecg_founder_12lead.onnx"
    RESULTS_JSON = OUT_DIR / "validation_results.json"
    
    # Load config
    cfg = load_config(CFG)
    input_size = int(cfg["input_size"])
    print(f"\n✓ Config: {cfg}")
    
    # Build model
    net1d = import_net1d(NET1D)
    n_classes = infer_n_classes(CKPT)
    print(f"✓ Classes: {n_classes}")
    
    model = build_model(net1d, n_classes)
    missing, unexpected = load_weights(model, CKPT)
    print(f"✓ Weights loaded (missing={len(missing)}, unexpected={len(unexpected)})")
    
    # Export to ONNX
    print("\n🔄 Exporting ONNX (opset=14, no constant folding)...")
    export_to_onnx(model, OUT_ONNX, input_size=input_size, opset=14)
    size_mb = os.path.getsize(OUT_ONNX) / (1024 * 1024)
    print(f"✓ ONNX saved: {OUT_ONNX} ({size_mb:.2f} MB)")
    
    # Validate exactness
    print("\n🔍 Validating exactness...")
    val = validate_exact(model, OUT_ONNX, input_size=input_size)
    
    # Save results
    with open(RESULTS_JSON, "w") as f:
        json.dump({
            "onnx_path": str(OUT_ONNX),
            "file_size_mb": size_mb,
            "validation": val,
        }, f, indent=2)
    
    if val.get("skipped"):
        print("⚠ ONNX Runtime not installed; validation skipped.")
    else:
        print(f"  - all_close: {val['all_close']} (rtol={val['rtol']}, atol={val['atol']})")
        print(f"  - max_abs: {val['max_abs']:.3e}")
        print(f"  - mean_abs: {val['mean_abs']:.3e}")
        
        if not val["all_close"]:
            print("\n⚠️ WARNING: Numerical parity NOT achieved!")
            print("   Please ensure net1d.py has been patched as documented in the header.")
            print("   Expected changes:")
            print("   1) Line 142: Swish forward → torch.nn.functional.silu(x)")
            print("   2) Line 271: einsum → out * se.unsqueeze(-1)")
    
    print("\n✅ Done.")


if __name__ == "__main__":
    main()
