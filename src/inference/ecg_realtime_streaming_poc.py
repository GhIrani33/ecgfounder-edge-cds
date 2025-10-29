# -*- coding: utf-8 -*-
"""
ecg_realtime_streaming_poc.py - FIXED VERSION

Purpose:
- Simulate real-time ECG streaming from ADS1298 device
- Process multiple ECG windows sequentially (like continuous acquisition)
- Demonstrate complete pipeline: load → preprocess → infer → report
- Save results for UI visualization

Key Changes:
1. ✅ Use ecg_founder_all71.onnx (71-class fine-tuned model)
2. ✅ Import ecg_preprocessing_pipeline.py (consistent preprocessing)
3. ✅ Accurate latency reporting (preprocessing + inference)
4. ✅ Save processed ECG windows for UI
5. ✅ Better error handling and logging

Usage:
python ecg_realtime_streaming_poc.py `
  --num_windows 50 `
  --save_windows `
  --output "D:\Project\ECG\realtime_poc_report.json"

Author: PhD Research Project  
Date: October 2025
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
import onnxruntime as ort

# Import preprocessing pipeline
from ecg_preprocessing_pipeline import ECGPreprocessor

# ============================================================================
# Configuration
# ============================================================================
PTB_XL_ROOT = Path(r"D:\Project\ECG\Dataset\PTB-XL")
RECORDS_DIR = PTB_XL_ROOT / "records500"
DATABASE_CSV = PTB_XL_ROOT / "ptbxl_database.csv"

ONNX_MODEL = Path(r"D:\Project\ECG\ECGFounder\onnx_models\ecg_founder_all71.onnx")
LABELS_JSON = Path(r"D:\Project\ECG\ECGFounder\posttrain_all71\labels_all71.json")

DEFAULT_OUTPUT = Path("realtime_poc_report.json")
DEFAULT_WINDOWS_DIR = Path("realtime_poc_windows_npy")


# ============================================================================
# Main POC
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Real-time ECG Streaming POC")
    parser.add_argument("--num_windows", type=int, default=50, help="Number of ECG windows to process")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output JSON report path")
    parser.add_argument("--save_windows", action="store_true", help="Save processed ECG windows as .npy")
    args = parser.parse_args()
    
    output_json = Path(args.output)
    windows_dir = output_json.parent / (output_json.stem + "_windows_npy")
    
    print("\n" + "=" * 70)
    print("REAL-TIME ECG STREAMING PROOF-OF-CONCEPT")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {ONNX_MODEL.name}")
    print(f"  Windows to process: {args.num_windows}")
    print(f"  Output: {output_json}")
    print(f"  Save windows: {args.save_windows}")
    print()
    
    # Verify files
    assert ONNX_MODEL.exists(), f"ONNX model not found: {ONNX_MODEL}"
    assert LABELS_JSON.exists(), f"Labels not found: {LABELS_JSON}"
    assert DATABASE_CSV.exists(), f"Database not found: {DATABASE_CSV}"
    
    # Create windows directory
    if args.save_windows:
        windows_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Windows directory: {windows_dir}")
    
    # Load labels
    with open(LABELS_JSON, 'r') as f:
        labels_data = json.load(f)
    label_list = labels_data['labels']
    print(f"✓ Loaded {len(label_list)} diagnostic labels")
    
    # Load ONNX session
    print("Loading ONNX model...")
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(ONNX_MODEL), sess_opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"✓ ONNX model loaded")
    
    # Initialize preprocessor
    preprocessor = ECGPreprocessor(
        use_bandpass=True,
        bandpass_low_hz=0.5,
        bandpass_high_hz=50.0,
        use_notch=True,
        qc_mode='lenient'
    )
    print("✓ Preprocessing pipeline initialized")
    
    # Load test records
    print(f"\nLoading PTB-XL test records...")
    db = pd.read_csv(DATABASE_CSV)
    test_records = db[db.strat_fold == 10].head(args.num_windows)
    print(f"✓ Loaded {len(test_records)} test records\n")
    
    # Lead names
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # ========================================================================
    # STREAMING SIMULATION
    # ========================================================================
    print("=" * 70)
    print("🚀 Simulating Real-Time ECG Streaming")
    print("=" * 70)
    print()
    
    results = []
    preprocess_times = []
    inference_times = []
    qc_failures = 0
    
    for idx, (_, row) in enumerate(test_records.iterrows()):
        try:
            # Load ECG (simulates ADS1298 acquisition)
            filename = row['filename_hr']
            rec_path = PTB_XL_ROOT / filename
            
            sig, fields = wfdb.rdsamp(str(rec_path))
            raw_ecg = sig.T.astype(np.float32)  # (12, n_samples)
            fs = fields['fs']
            
            rec_id = filename.split('/')[-1]
            
            # ====== PREPROCESSING ======
            t0_preprocess = time.perf_counter()
            processed_ecg, metadata = preprocessor.process(raw_ecg, lead_names, fs)
            t1_preprocess = time.perf_counter()
            preprocess_time_ms = (t1_preprocess - t0_preprocess) * 1000
            
            # Check QC
            if not metadata['qc_pass']:
                qc_failures += 1
                print(f"[{idx+1}/{args.num_windows}] {rec_id} ⚠️ QC FAILED - skipping")
                continue
            
            # ====== INFERENCE ======
            x = processed_ecg[np.newaxis, :, :].astype(np.float32)
            t0_inference = time.perf_counter()
            logits = session.run([output_name], {input_name: x})[0][0]
            t1_inference = time.perf_counter()
            inference_time_ms = (t1_inference - t0_inference) * 1000
            
            # Convert logits to probabilities
            probs = 1.0 / (1.0 + np.exp(-logits))
            
            # Get top-5 diagnoses
            top5_indices = np.argsort(probs)[::-1][:5]
            top5_diagnoses = [
                {
                    "label": label_list[i],
                    "probability": float(probs[i]),
                    "logit": float(logits[i])
                }
                for i in top5_indices
            ]
            
            # Save window if requested
            window_path = None
            if args.save_windows:
                window_filename = f"{rec_id}.npy"
                window_path = windows_dir / window_filename
                np.save(window_path, processed_ecg)
            
            # Store result
            result = {
                "window_id": idx + 1,
                "record_id": rec_id,
                "latency_ms": {
                    "preprocessing": round(preprocess_time_ms, 2),
                    "inference": round(inference_time_ms, 2),
                    "total": round(preprocess_time_ms + inference_time_ms, 2)
                },
                "top5_diagnoses": top5_diagnoses,
                "qc_metadata": {
                    "qc_pass": metadata['qc_pass'],
                    "warnings": metadata.get('qc_warnings', []),
                    "powerline_snr_db": metadata.get('powerline_snr_db', None)
                }
            }
            
            if window_path:
                result["ecg_window_npy"] = str(window_path)
            
            results.append(result)
            preprocess_times.append(preprocess_time_ms)
            inference_times.append(inference_time_ms)
            
            # Progress
            if (idx + 1) % 10 == 0:
                print(f"[{idx+1}/{args.num_windows}] Processed {len(results)} windows (QC failures: {qc_failures})...")
        
        except Exception as e:
            print(f"[{idx+1}/{args.num_windows}] {rec_id} ❌ ERROR: {e}")
            continue
    
    print(f"\n✓ Streaming complete: {len(results)}/{args.num_windows} windows processed\n")
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    if len(preprocess_times) == 0:
        print("❌ ERROR: No windows were successfully processed!")
        return
    
    preprocess_times = np.array(preprocess_times)
    inference_times = np.array(inference_times)
    total_times = preprocess_times + inference_times
    
    report = {
        "experiment": "Real-Time ECG Streaming Proof-of-Concept",
        "configuration": {
            "model": str(ONNX_MODEL),
            "num_windows_requested": args.num_windows,
            "num_windows_processed": len(results),
            "qc_failures": qc_failures,
            "preprocessing": "0.5-50 Hz bandpass + adaptive 50Hz notch + z-score"
        },
        "latency_statistics_ms": {
            "preprocessing": {
                "mean": float(np.mean(preprocess_times)),
                "median": float(np.median(preprocess_times)),
                "std": float(np.std(preprocess_times)),
                "min": float(np.min(preprocess_times)),
                "max": float(np.max(preprocess_times)),
                "p95": float(np.percentile(preprocess_times, 95)),
                "p99": float(np.percentile(preprocess_times, 99))
            },
            "inference": {
                "mean": float(np.mean(inference_times)),
                "median": float(np.median(inference_times)),
                "std": float(np.std(inference_times)),
                "min": float(np.min(inference_times)),
                "max": float(np.max(inference_times)),
                "p95": float(np.percentile(inference_times, 95)),
                "p99": float(np.percentile(inference_times, 99))
            },
            "total": {
                "mean": float(np.mean(total_times)),
                "median": float(np.median(total_times)),
                "std": float(np.std(total_times)),
                "min": float(np.min(total_times)),
                "max": float(np.max(total_times)),
                "p95": float(np.percentile(total_times, 95)),
                "p99": float(np.percentile(total_times, 99))
            }
        },
        "windows": results
    }
    
    # ========================================================================
    # REPORT
    # ========================================================================
    print("=" * 70)
    print("📊 LATENCY SUMMARY")
    print("=" * 70)
    print(f"\nWindows processed: {len(results)} (QC failures: {qc_failures})")
    print(f"\nPREPROCESSING:")
    print(f"  Mean:   {report['latency_statistics_ms']['preprocessing']['mean']:.2f} ms")
    print(f"  Median: {report['latency_statistics_ms']['preprocessing']['median']:.2f} ms")
    print(f"  P95:    {report['latency_statistics_ms']['preprocessing']['p95']:.2f} ms")
    
    print(f"\nINFERENCE:")
    print(f"  Mean:   {report['latency_statistics_ms']['inference']['mean']:.2f} ms")
    print(f"  Median: {report['latency_statistics_ms']['inference']['median']:.2f} ms")
    print(f"  P95:    {report['latency_statistics_ms']['inference']['p95']:.2f} ms")
    
    print(f"\nTOTAL:")
    print(f"  Mean:   {report['latency_statistics_ms']['total']['mean']:.2f} ms")
    print(f"  Median: {report['latency_statistics_ms']['total']['median']:.2f} ms")
    print(f"  P95:    {report['latency_statistics_ms']['total']['p95']:.2f} ms")
    
    # Save report
    with open(output_json, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Report saved: {output_json}")
    
    if args.save_windows:
        print(f"✓ Windows saved: {windows_dir} ({len(results)} files)")
    
    print("\n" + "=" * 70)
    print("✅ PROOF-OF-CONCEPT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
