# -*- coding: utf-8 -*-
"""
benchmark_complete_pipeline.py

Purpose:
- Benchmark COMPLETE pipeline (preprocessing + ONNX inference)
- Use REAL PTB-XL ECG data (not random)
- Report separate timings: preprocessing, inference, total
- Project ARM latency (conservative 2× factor)

Usage:
    python benchmark_complete_pipeline.py

Author: PhD Research Project
Date: October 2025
"""

import os
import sys
import time
import json
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

NUM_WARMUP = 20
NUM_RUNS = 200

OUTPUT_JSON = Path("benchmark_complete_pipeline_results.json")


# ============================================================================
# Main Benchmark
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("COMPLETE PIPELINE BENCHMARK (Preprocessing + ONNX Inference)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {ONNX_MODEL.name}")
    print(f"  Warmup runs: {NUM_WARMUP}")
    print(f"  Benchmark runs: {NUM_RUNS}")
    print(f"  Platform: x86-64 (Windows)")
    print()
    
    # Verify files
    assert ONNX_MODEL.exists(), f"ONNX model not found: {ONNX_MODEL}"
    assert DATABASE_CSV.exists(), f"Database not found: {DATABASE_CSV}"
    
    # Load ONNX session
    print("Loading ONNX model...")
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1  # Single-threaded for deterministic timing
    session = ort.InferenceSession(str(ONNX_MODEL), sess_opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"✓ ONNX model loaded: {ONNX_MODEL.name}")
    
    # Initialize preprocessor
    preprocessor = ECGPreprocessor(
        use_bandpass=True,
        bandpass_low_hz=0.5,
        bandpass_high_hz=50.0,
        use_notch=True,
        qc_mode='lenient'
    )
    print("✓ Preprocessing pipeline initialized (0.5-50 Hz + adaptive notch)")
    
    # Load sample ECG from PTB-XL
    print(f"\nLoading sample ECG from PTB-XL...")
    db = pd.read_csv(DATABASE_CSV)
    sample_row = db[db.strat_fold == 10].iloc[0]
    filename = sample_row['filename_hr']
    rec_path = PTB_XL_ROOT / filename
    
    sig, fields = wfdb.rdsamp(str(rec_path))
    raw_ecg = sig.T.astype(np.float32)  # (12, n_samples)
    fs = fields['fs']
    print(f"✓ Loaded: {filename} ({raw_ecg.shape[1]} samples @ {fs} Hz)")
    
    # Lead names
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # ========================================================================
    # WARMUP
    # ========================================================================
    print(f"\nWarmup ({NUM_WARMUP} iterations)...")
    for _ in range(NUM_WARMUP):
        processed_ecg, _ = preprocessor.process(raw_ecg, lead_names, fs)
        x = processed_ecg[np.newaxis, :, :].astype(np.float32)
        _ = session.run([output_name], {input_name: x})
    print("✓ Warmup complete")
    
    # ========================================================================
    # BENCHMARK
    # ========================================================================
    print(f"\nBenchmarking ({NUM_RUNS} iterations)...")
    
    preprocess_times = []
    inference_times = []
    total_times = []
    
    for i in range(NUM_RUNS):
        # Measure preprocessing
        t0_preprocess = time.perf_counter()
        processed_ecg, metadata = preprocessor.process(raw_ecg, lead_names, fs)
        t1_preprocess = time.perf_counter()
        preprocess_time_ms = (t1_preprocess - t0_preprocess) * 1000
        preprocess_times.append(preprocess_time_ms)
        
        # Measure inference
        x = processed_ecg[np.newaxis, :, :].astype(np.float32)
        t0_inference = time.perf_counter()
        logits = session.run([output_name], {input_name: x})[0]
        t1_inference = time.perf_counter()
        inference_time_ms = (t1_inference - t0_inference) * 1000
        inference_times.append(inference_time_ms)
        
        # Total time
        total_time_ms = preprocess_time_ms + inference_time_ms
        total_times.append(total_time_ms)
        
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{NUM_RUNS} completed...")
    
    print("✓ Benchmark complete\n")
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    preprocess_times = np.array(preprocess_times)
    inference_times = np.array(inference_times)
    total_times = np.array(total_times)
    
    results = {
        "configuration": {
            "model": str(ONNX_MODEL),
            "num_runs": NUM_RUNS,
            "platform": "x86-64 (Windows)",
            "preprocessing": "0.5-50 Hz bandpass + adaptive 50Hz notch + z-score",
            "onnx_threads": 1
        },
        "x86_latency_ms": {
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
        "arm_projected_latency_ms": {
            "note": "Conservative 2x slowdown projection for ARM Cortex-A72 vs x86-64",
            "preprocessing_mean": float(np.mean(preprocess_times) * 2.0),
            "inference_mean": float(np.mean(inference_times) * 2.0),
            "total_mean": float(np.mean(total_times) * 2.0),
            "realtime_requirement_ms": 500,
            "meets_requirement": float(np.mean(total_times) * 2.0) < 500
        }
    }
    
    # ========================================================================
    # REPORT
    # ========================================================================
    print("=" * 70)
    print("BENCHMARK RESULTS (x86-64)")
    print("=" * 70)
    print(f"\nPREPROCESSING:")
    print(f"  Mean:   {results['x86_latency_ms']['preprocessing']['mean']:.2f} ms")
    print(f"  Median: {results['x86_latency_ms']['preprocessing']['median']:.2f} ms")
    print(f"  P95:    {results['x86_latency_ms']['preprocessing']['p95']:.2f} ms")
    print(f"  P99:    {results['x86_latency_ms']['preprocessing']['p99']:.2f} ms")
    
    print(f"\nINFERENCE:")
    print(f"  Mean:   {results['x86_latency_ms']['inference']['mean']:.2f} ms")
    print(f"  Median: {results['x86_latency_ms']['inference']['median']:.2f} ms")
    print(f"  P95:    {results['x86_latency_ms']['inference']['p95']:.2f} ms")
    print(f"  P99:    {results['x86_latency_ms']['inference']['p99']:.2f} ms")
    
    print(f"\nTOTAL PIPELINE:")
    print(f"  Mean:   {results['x86_latency_ms']['total']['mean']:.2f} ms")
    print(f"  Median: {results['x86_latency_ms']['total']['median']:.2f} ms")
    print(f"  P95:    {results['x86_latency_ms']['total']['p95']:.2f} ms")
    print(f"  P99:    {results['x86_latency_ms']['total']['p99']:.2f} ms")
    
    print("\n" + "=" * 70)
    print("PROJECTED ARM LATENCY (Conservative 2× Factor)")
    print("=" * 70)
    print(f"\nPreprocessing: {results['arm_projected_latency_ms']['preprocessing_mean']:.2f} ms")
    print(f"Inference:     {results['arm_projected_latency_ms']['inference_mean']:.2f} ms")
    print(f"TOTAL:         {results['arm_projected_latency_ms']['total_mean']:.2f} ms")
    print(f"\nReal-time requirement: <500 ms")
    print(f"Meets requirement: {'✅ YES' if results['arm_projected_latency_ms']['meets_requirement'] else '❌ NO'}")
    
    # Save results
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved: {OUTPUT_JSON}")
    print("=" * 70)


if __name__ == "__main__":
    main()
