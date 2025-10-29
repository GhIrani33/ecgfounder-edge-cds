# -*- coding: utf-8 -*-
"""
preprocessing_validation_with_noise.py

Purpose:
- Validate preprocessing pipeline on REAL PTB-XL data
- Test 3 conditions:
  1. Clean (baseline performance)
  2. Noisy without preprocessing (show degradation)
  3. Noisy with preprocessing (show recovery)

Usage:
    python preprocessing_validation_with_noise.py

Output:
    - preprocessing_validation_report_with_noise.json
    - preprocessing_validation_comparison.png

Author: Ghasem
https://github.com/GhIrani33
"""


import json
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
import onnxruntime as ort
import matplotlib.pyplot as plt

# Import our modules
from ecg_preprocessing_pipeline import ECGPreprocessor
from ecg_noise_simulator import ECGNoiseSimulator


# ============================================================================
# Configuration
# ============================================================================
PTB_XL_ROOT = Path(r"...\Dataset\PTB-XL")
RECORDS_DIR = PTB_XL_ROOT / "records500"
DATABASE_CSV = PTB_XL_ROOT / "ptbxl_database.csv"

ONNX_MODEL = Path(r"...\onnx_models\ecg_founder_all71.onnx")
LABELS_JSON = Path(r"...\labels_all71.json")

OUTPUT_JSON = Path("preprocessing_validation_report_with_noise.json")
OUTPUT_PLOT = Path("preprocessing_validation_comparison.png")

NUM_SAMPLES = 100  # Number of test samples
NOISE_LEVEL = 'medium'  # 'low', 'medium', or 'high'


# ============================================================================
# Load PTB-XL Sample
# ============================================================================
def load_ptbxl_sample(record_path: Path) -> tuple:
    """
    Load a PTB-XL ECG recording
    
    Returns:
    --------
    signal : np.ndarray
        ECG signal (12, n_samples)
    fs : int
        Sampling frequency
    """
    sig, fields = wfdb.rdsamp(str(record_path))
    signal = sig.T.astype(np.float32)  # (12, n_samples)
    fs = fields['fs']
    
    return signal, fs


def extract_labels(db_row, label_list):
    """Extract multi-label vector from PTB-XL metadata"""
    import ast
    scp_codes = ast.literal_eval(db_row['scp_codes'])
    
    labels = np.zeros(len(label_list), dtype=np.float32)
    for code in scp_codes.keys():
        if code in label_list:
            labels[label_list.index(code)] = 1.0
    
    return labels


# ============================================================================
# ONNX Inference
# ============================================================================
def run_onnx_inference(session, ecg_input):
    """
    Run ONNX model inference
    
    Parameters:
    -----------
    session : ort.InferenceSession
    ecg_input : np.ndarray
        Shape (12, 5000)
    
    Returns:
    --------
    logits : np.ndarray
        Shape (n_classes,)
    """
    x = ecg_input[np.newaxis, :, :].astype(np.float32)  # Add batch dim
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    logits = session.run([output_name], {input_name: x})[0][0]
    return logits


# ============================================================================
# Main Validation
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("PREPROCESSING VALIDATION WITH SYNTHETIC NOISE")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  PTB-XL: {PTB_XL_ROOT}")
    print(f"  ONNX Model: {ONNX_MODEL}")
    print(f"  Samples: {NUM_SAMPLES}")
    print(f"  Noise Level: {NOISE_LEVEL}")
    print()
    
    # Verify paths
    assert DATABASE_CSV.exists(), f"Database not found: {DATABASE_CSV}"
    assert ONNX_MODEL.exists(), f"ONNX model not found: {ONNX_MODEL}"
    assert LABELS_JSON.exists(), f"Labels not found: {LABELS_JSON}"
    
    # Load labels
    with open(LABELS_JSON, 'r') as f:
        labels_data = json.load(f)
    label_list = labels_data['labels']
    print(f"✓ Loaded {len(label_list)} diagnostic labels")
    
    # Load ONNX model
    session = ort.InferenceSession(str(ONNX_MODEL), providers=['CPUExecutionProvider'])
    print(f"✓ Loaded ONNX model")
    
    # Load test set
    db = pd.read_csv(DATABASE_CSV)
    test_fold = db[db.strat_fold == 10].head(NUM_SAMPLES)
    print(f"✓ Loaded {len(test_fold)} test samples\n")
    
    # Initialize preprocessor
    preprocessor = ECGPreprocessor(
        use_bandpass=True,
        bandpass_low_hz=0.5,
        bandpass_high_hz=50.0,
        use_notch=True,
        qc_mode='lenient'
    )
    
    # Storage for results
    results_clean = {'logits': [], 'labels': []}
    results_noisy_raw = {'logits': [], 'labels': []}
    results_noisy_filtered = {'logits': [], 'labels': []}
    
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    print("Processing samples...")
    success_count = 0
    
    for idx, (_, row) in enumerate(test_fold.iterrows()):
        try:
            # Construct record path
            # Format: "records500/00000/00001_hr"
            filename = row['filename_hr']  # e.g., "records500/00000/00001_hr"
            
            # Build full path directly from filename
            rec_path = PTB_XL_ROOT / filename
            
            # Load ECG
            raw_ecg, fs = load_ptbxl_sample(rec_path)
            
            # Extract ground truth labels
            y_true = extract_labels(row, label_list)
            
            # ====== EXPERIMENT 1: Clean ECG ======
            clean_processed, _ = preprocessor.process(raw_ecg, lead_names, fs)
            logits_clean = run_onnx_inference(session, clean_processed)
            results_clean['logits'].append(logits_clean)
            results_clean['labels'].append(y_true)
            
            # ====== EXPERIMENT 2: Noisy ECG (no preprocessing) ======
            noisy_ecg = ECGNoiseSimulator.add_realistic_noise_mix(raw_ecg, fs, NOISE_LEVEL)
            
            # Minimal preprocessing: only z-score + length standardization
            noisy_normalized = (noisy_ecg - np.mean(noisy_ecg, axis=1, keepdims=True)) / \
                              (np.std(noisy_ecg, axis=1, keepdims=True) + 1e-8)
            
            # Standardize length to 5000 samples
            if noisy_normalized.shape[1] > 5000:
                start = (noisy_normalized.shape[1] - 5000) // 2
                noisy_normalized = noisy_normalized[:, start:start+5000]
            elif noisy_normalized.shape[1] < 5000:
                pad = 5000 - noisy_normalized.shape[1]
                noisy_normalized = np.pad(noisy_normalized, ((0,0), (0, pad)), mode='constant')
            
            logits_noisy_raw = run_onnx_inference(session, noisy_normalized.astype(np.float32))
            results_noisy_raw['logits'].append(logits_noisy_raw)
            results_noisy_raw['labels'].append(y_true)
            
            # ====== EXPERIMENT 3: Noisy ECG (with preprocessing) ======
            noisy_processed, _ = preprocessor.process(noisy_ecg, lead_names, fs)
            logits_noisy_filtered = run_onnx_inference(session, noisy_processed)
            results_noisy_filtered['logits'].append(logits_noisy_filtered)
            results_noisy_filtered['labels'].append(y_true)
            
            success_count += 1
            
            if (idx + 1) % 20 == 0:
                print(f"  Processed {idx + 1}/{NUM_SAMPLES} samples (success: {success_count})...")
        
        except Exception as e:
            print(f"⚠️ Error processing sample {idx}: {e}")
            continue
    
    print(f"✓ Successfully processed {success_count}/{NUM_SAMPLES} samples\n")
    
    # Check if we have enough data
    if success_count == 0:
        print("❌ ERROR: No samples were successfully processed!")
        print("Please check:")
        print(f"  1. Database CSV exists: {DATABASE_CSV.exists()}")
        print(f"  2. Records directory exists: {RECORDS_DIR.exists()}")
        print(f"  3. Sample record path format")
        return
    
    # ====== COMPUTE METRICS ======
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    def compute_metrics(results):
        if len(results['logits']) == 0:
            return {'macro_auroc': 0.0, 'macro_auprc': 0.0, 'per_class_auroc': []}
        
        logits = np.array(results['logits'])
        labels = np.array(results['labels'])
        probs = 1.0 / (1.0 + np.exp(-logits))
        
        # Per-class AUROC
        aurocs = []
        for i in range(labels.shape[1]):
            if labels[:, i].max() > labels[:, i].min():
                try:
                    aurocs.append(roc_auc_score(labels[:, i], probs[:, i]))
                except:
                    aurocs.append(np.nan)
            else:
                aurocs.append(np.nan)
        
        macro_auroc = np.nanmean(aurocs)
        
        # AUPRC
        auprcs = []
        for i in range(labels.shape[1]):
            if labels[:, i].max() > labels[:, i].min():
                try:
                    auprcs.append(average_precision_score(labels[:, i], probs[:, i]))
                except:
                    auprcs.append(np.nan)
            else:
                auprcs.append(np.nan)
        
        macro_auprc = np.nanmean(auprcs)
        
        return {
            'macro_auroc': float(macro_auroc),
            'macro_auprc': float(macro_auprc),
            'per_class_auroc': [float(x) if not np.isnan(x) else None for x in aurocs]
        }
    
    metrics_clean = compute_metrics(results_clean)
    metrics_noisy_raw = compute_metrics(results_noisy_raw)
    metrics_noisy_filtered = compute_metrics(results_noisy_filtered)
    
    print(f"\n1. CLEAN ECG (baseline):")
    print(f"   Macro AUROC: {metrics_clean['macro_auroc']:.4f}")
    print(f"   Macro AUPRC: {metrics_clean['macro_auprc']:.4f}")
    
    print(f"\n2. NOISY ECG (no filtering):")
    print(f"   Macro AUROC: {metrics_noisy_raw['macro_auroc']:.4f}")
    print(f"   Macro AUPRC: {metrics_noisy_raw['macro_auprc']:.4f}")
    print(f"   Δ vs Clean: {metrics_noisy_raw['macro_auroc'] - metrics_clean['macro_auroc']:+.4f} AUROC")
    
    print(f"\n3. NOISY ECG (with preprocessing):")
    print(f"   Macro AUROC: {metrics_noisy_filtered['macro_auroc']:.4f}")
    print(f"   Macro AUPRC: {metrics_noisy_filtered['macro_auprc']:.4f}")
    print(f"   Δ vs Noisy: {metrics_noisy_filtered['macro_auroc'] - metrics_noisy_raw['macro_auroc']:+.4f} AUROC")
    
    if metrics_clean['macro_auroc'] > metrics_noisy_raw['macro_auroc']:
        recovery_rate = (metrics_noisy_filtered['macro_auroc'] - metrics_noisy_raw['macro_auroc']) / \
                        (metrics_clean['macro_auroc'] - metrics_noisy_raw['macro_auroc'])
        print(f"\n✅ Preprocessing recovers {recovery_rate*100:.1f}% of noise-induced degradation")
    else:
        print("\n⚠️ Warning: Noise did not degrade performance (unexpected)")
    
    # ====== SAVE REPORT ======
    report = {
        'experiment': 'Preprocessing Validation with Synthetic Noise',
        'configuration': {
            'num_samples_requested': NUM_SAMPLES,
            'num_samples_processed': success_count,
            'noise_level': NOISE_LEVEL,
            'preprocessing': {
                'bandpass': '0.5-50 Hz',
                'notch': 'adaptive (50 Hz)',
                'normalization': 'z-score per lead'
            }
        },
        'metrics': {
            'clean': metrics_clean,
            'noisy_without_preprocessing': metrics_noisy_raw,
            'noisy_with_preprocessing': metrics_noisy_filtered
        },
        'conclusion': {
            'noise_degradation_auroc': float(metrics_noisy_raw['macro_auroc'] - metrics_clean['macro_auroc']),
            'preprocessing_improvement_auroc': float(metrics_noisy_filtered['macro_auroc'] - metrics_noisy_raw['macro_auroc']),
            'recovery_rate_percent': float(recovery_rate * 100) if metrics_clean['macro_auroc'] > metrics_noisy_raw['macro_auroc'] else None
        }
    }
    
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Report saved: {OUTPUT_JSON}")
    
    # ====== PLOT COMPARISON ======
    fig, ax = plt.subplots(figsize=(10, 6))
    
    conditions = ['Clean\n(Baseline)', 'Noisy\n(No Filter)', 'Noisy\n(Filtered)']
    aurocs = [
        metrics_clean['macro_auroc'],
        metrics_noisy_raw['macro_auroc'],
        metrics_noisy_filtered['macro_auroc']
    ]
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    
    bars = ax.bar(conditions, aurocs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, auroc in zip(bars, aurocs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{auroc:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Macro AUROC', fontsize=12, fontweight='bold')
    ax.set_title(f'Preprocessing Pipeline Validation: Noise Recovery Analysis\n({success_count} samples, {NOISE_LEVEL} noise)', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0.6, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved: {OUTPUT_PLOT}")
    
    print("\n" + "=" * 70)
    print("✅ VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

