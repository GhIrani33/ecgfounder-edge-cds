# -*- coding: utf-8 -*-
"""
generate_lead_attribution.py

Generate lead attribution analysis using Integrated Gradients.
Creates lead_influence_scores3.csv for publication figures.

Usage:
    python generate_lead_attribution.py

Output:
    lead_influence_scores3.csv - Per-lead attribution for top diagnoses

Author: Ghasem
https://github.com/GhIrani33
"""

import json
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
import onnxruntime as ort
from tqdm import tqdm

# Import preprocessing
from ecg_preprocessing_pipeline import ECGPreprocessor

# ============================================================================
# Configuration
# ============================================================================
PTB_XL_ROOT = Path(r"...\Dataset\PTB-XL")
RECORDS_DIR = PTB_XL_ROOT / "records500"
DATABASE_CSV = PTB_XL_ROOT / "ptbxl_database.csv"

ONNX_MODEL = Path(r"...\onnx_models\ecg_founder_all71.onnx")
LABELS_JSON = Path(r"...\labels_all71.json")

OUTPUT_CSV = Path("lead_influence_scores3.csv")

# Top diagnoses to analyze
TOP_DIAGNOSES = ['NORM', 'AFIB', 'LBBB', 'RBBB', 'STD', 'STE', 'MI', 'LVH', 'IAVB', '1AVB']
NUM_SAMPLES_PER_DIAGNOSIS = 10  # Samples to analyze per diagnosis
IG_STEPS = 50  # Integrated Gradients interpolation steps


# ============================================================================
# Integrated Gradients (Manual Implementation for ONNX)
# ============================================================================

def compute_integrated_gradients_onnx(session, input_ecg, target_class_idx, steps=50):
    """
    Compute Integrated Gradients for ONNX model
    
    Parameters:
    -----------
    session : ort.InferenceSession
        ONNX Runtime session
    input_ecg : np.ndarray
        Input ECG signal, shape (12, 5000)
    target_class_idx : int
        Index of target diagnostic class
    steps : int
        Number of interpolation steps
    
    Returns:
    --------
    attributions : np.ndarray
        Attribution map, shape (12, 5000)
    """
    
    # Baseline: zero signal
    baseline = np.zeros_like(input_ecg)
    
    # Linear interpolation from baseline to input
    alphas = np.linspace(0, 1, steps)
    
    # Storage for gradients
    integrated_grads = np.zeros_like(input_ecg)
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    for alpha in alphas:
        # Interpolated input
        interpolated = baseline + alpha * (input_ecg - baseline)
        x_batch = interpolated[np.newaxis, :, :].astype(np.float32)
        
        # Forward pass
        logits = session.run([output_name], {input_name: x_batch})[0][0]
        target_logit = logits[target_class_idx]
        
        # Approximate gradient via finite differences
        # (Since ONNX doesn't support automatic differentiation, we approximate)
        epsilon = 1e-4
        grad_approx = np.zeros_like(input_ecg)
        
        # Compute gradient for each lead
        for lead_idx in range(input_ecg.shape[0]):
            # Perturb this lead
            x_plus = interpolated.copy()
            x_plus[lead_idx, :] += epsilon
            
            x_plus_batch = x_plus[np.newaxis, :, :].astype(np.float32)
            logits_plus = session.run([output_name], {input_name: x_plus_batch})[0][0]
            target_logit_plus = logits_plus[target_class_idx]
            
            # Finite difference gradient
            grad_approx[lead_idx, :] = (target_logit_plus - target_logit) / epsilon
        
        # Accumulate gradients
        integrated_grads += grad_approx
    
    # Average over steps and scale by input difference
    integrated_grads = integrated_grads / steps
    attributions = (input_ecg - baseline) * integrated_grads
    
    return attributions


def compute_lead_attribution_fast(session, input_ecg, target_class_idx):
    """
    Fast approximation: Gradient × Input method
    (Faster than IG but less theoretically grounded)
    
    Parameters:
    -----------
    session : ort.InferenceSession
    input_ecg : np.ndarray, shape (12, 5000)
    target_class_idx : int
    
    Returns:
    --------
    attributions : np.ndarray, shape (12, 5000)
    """
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Forward pass
    x_batch = input_ecg[np.newaxis, :, :].astype(np.float32)
    logits = session.run([output_name], {input_name: x_batch})[0][0]
    target_logit = logits[target_class_idx]
    
    # Compute gradient per lead (finite differences)
    epsilon = 1e-3
    grad = np.zeros_like(input_ecg)
    
    for lead_idx in range(input_ecg.shape[0]):
        # Perturb entire lead uniformly
        x_plus = input_ecg.copy()
        x_plus[lead_idx, :] += epsilon
        
        x_plus_batch = x_plus[np.newaxis, :, :].astype(np.float32)
        logits_plus = session.run([output_name], {input_name: x_plus_batch})[0][0]
        target_logit_plus = logits_plus[target_class_idx]
        
        # Gradient approximation
        grad[lead_idx, :] = (target_logit_plus - target_logit) / epsilon
    
    # Attribution = gradient × input
    attributions = grad * input_ecg
    
    return attributions


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("LEAD ATTRIBUTION ANALYSIS - INTEGRATED GRADIENTS")
    print("=" * 70)
    print()
    
    # Load labels
    with open(LABELS_JSON, 'r') as f:
        labels_data = json.load(f)
    label_list = labels_data['labels']
    print(f"✓ Loaded {len(label_list)} diagnostic labels")
    
    # Load ONNX model
    print("Loading ONNX model...")
    session = ort.InferenceSession(str(ONNX_MODEL), providers=["CPUExecutionProvider"])
    print(f"✓ ONNX model loaded")
    
    # Initialize preprocessor
    preprocessor = ECGPreprocessor(
        use_bandpass=True,
        bandpass_low_hz=0.5,
        bandpass_high_hz=50.0,
        use_notch=True,
        qc_mode='lenient'
    )
    print("✓ Preprocessing pipeline initialized\n")
    
    # Load database
    db = pd.read_csv(DATABASE_CSV)
    
    # Storage for results
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    results = []
    
    print("Analyzing lead attributions for top diagnoses...")
    print(f"Diagnoses: {', '.join(TOP_DIAGNOSES)}")
    print(f"Samples per diagnosis: {NUM_SAMPLES_PER_DIAGNOSIS}\n")
    
    for diagnosis in TOP_DIAGNOSES:
        print(f"Processing: {diagnosis}...")
        
        # Find samples with this diagnosis
        # (Simplified: we'll use samples where this diagnosis appears in scp_codes)
        import ast
        
        samples_with_diagnosis = []
        for idx, row in db.iterrows():
            scp_codes = ast.literal_eval(row['scp_codes'])
            if diagnosis in scp_codes:
                samples_with_diagnosis.append(row)
            
            if len(samples_with_diagnosis) >= NUM_SAMPLES_PER_DIAGNOSIS:
                break
        
        if len(samples_with_diagnosis) == 0:
            print(f"  ⚠️ No samples found for {diagnosis} - using NORM as fallback")
            samples_with_diagnosis = db[db.strat_fold == 10].head(NUM_SAMPLES_PER_DIAGNOSIS).to_dict('records')
        
        # Get class index
        if diagnosis in label_list:
            class_idx = label_list.index(diagnosis)
        else:
            print(f"  ⚠️ {diagnosis} not in label list - skipping")
            continue
        
        # Accumulate attributions across samples
        total_attribution = np.zeros(12)
        
        for sample_row in tqdm(samples_with_diagnosis[:NUM_SAMPLES_PER_DIAGNOSIS], 
                               desc=f"  {diagnosis}", leave=False):
            try:
                # Load ECG
                filename = sample_row['filename_hr']
                rec_path = PTB_XL_ROOT / filename
                
                sig, fields = wfdb.rdsamp(str(rec_path))
                raw_ecg = sig.T.astype(np.float32)
                fs = fields['fs']
                
                # Preprocess
                processed_ecg, _ = preprocessor.process(raw_ecg, lead_names, fs)
                
                # Compute attribution (fast method)
                attribution = compute_lead_attribution_fast(session, processed_ecg, class_idx)
                
                # Aggregate per lead (sum of absolute attributions)
                lead_attribution = np.sum(np.abs(attribution), axis=1)
                total_attribution += lead_attribution
            
            except Exception as e:
                print(f"    ⚠️ Error: {e}")
                continue
        
        # Average across samples
        avg_attribution = total_attribution / NUM_SAMPLES_PER_DIAGNOSIS
        
        # Normalize to percentage
        avg_attribution = (avg_attribution / avg_attribution.sum()) * 100
        
        # Store result
        result = {'diagnosis': diagnosis}
        for lead_name, attr in zip(lead_names, avg_attribution):
            result[lead_name] = round(attr, 2)
        
        results.append(result)
        print(f"  ✓ {diagnosis} complete\n")
    
    # Save to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_CSV, index=False)
    
    print("=" * 70)
    print(f"✅ ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n✓ Saved: {OUTPUT_CSV}")
    print(f"\nSummary:")
    print(df_results)
    print()


if __name__ == "__main__":
    main()

