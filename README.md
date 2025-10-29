# ECGFounder Real-Time Deployment: Proof-of-Concept for CDS

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Real-time deployment of ECGFounder foundation model [1] for automated 12-lead ECG interpretation on low-cost edge hardware. This proof-of-concept demonstrates technical feasibility for clinical decision support systems in resource-constrained settings.

![Performance](results/figures/figure2_diagnostic_performance.png)

---

## Overview

This repository implements an end-to-end pipeline for deploying the ECGFounder foundation model [1] (trained on 10.5M ECG recordings) on ARM-based edge devices (Raspberry Pi 4). The system processes raw ECG signals directly from acquisition devices and provides real-time diagnostic predictions across 150 Label.

**Objective:** Demonstrate that state-of-the-art ECG AI can achieve clinical-grade performance on affordable hardware, enabling deployment in settings where GPU infrastructure is unavailable.

**Key Results:**
- Diagnostic accuracy: 0.909 macro AUROC (71 conditions validated on PTB-XL benchmark, n=2,163)
- Latency: 115 ms on Raspberry Pi 4 (real-time capable)
- Noise robustness: 95% performance recovery under clinical noise conditions
- Hardware cost: $75 (50× cheaper than GPU-based systems)

**Status:** Research prototype validated on benchmark dataset. Requires prospective clinical validation before deployment.

---

## Methodology

### 1. Model Deployment
- **Foundation model:** ECGFounder [1] (28M parameters, Net1D architecture)
- **Conversion:** PyTorch → ONNX (opset 14) for cross-platform inference
- **Optimization:** Single-threaded CPU inference with validated numerical equivalence (Pearson r > 0.998)

### 2. Preprocessing Pipeline
- **Bandpass filter:** 0.5-50 Hz (2nd-order Butterworth, zero-phase)
- **Adaptive notch:** 50 Hz powerline rejection (Q=30, applied when SNR > 10 dB threshold)
- **Quality control:** Signal-to-noise ratio, saturation detection, lead-off detection
- **Normalization:** Z-score per lead
- **Design rationale:** Match ECGFounder training distribution while adding clinical robustness

### 3. Validation Protocol
- **Dataset:** PTB-XL [2](21,837 recordings, stratified 10-fold cross-validation)
- **Test set:** Fold 10 (n=2,163, independent evaluation)
- **Metrics:** Macro AUROC, per-class AUROC, precision-recall
- **Noise testing:** Controlled injection of powerline (50 Hz), baseline wander, EMG artifacts
- **Latency benchmarking:** 200-iteration profiling with 20-sample warmup on x86-64 and ARM

### 4. Explainability Analysis
Integrated Gradients attribution analysis for per-lead importance. Clinical validation of attribution patterns against established ECG interpretation criteria (e.g., Cornell criteria for LVH).

**Note:** Explainability methods are under active development. Current implementation provides lead-level attribution; deeper clinical validation in progress.

---

## Performance Summary

| Metric | Value | Comparison |
|--------|-------|------------|
| Macro AUROC (71 classes) | 0.909 | > Ribeiro 2020 (0.887)[3], Hannun 2019 (0.853)[4] |
| Inference latency (ARM) | 115 ms | Real-time capable (<500 ms threshold) |
| Noise recovery rate | 95% | AUROC: 0.909 → 0.545 (noisy) → 0.891 (filtered) |
| Hardware cost | $100 | cheaper than GPU systems |

See [docs/METHODS.md](docs/METHODS.md) for complete methodology and [docs/VALIDATION.md](docs/VALIDATION.md) for detailed results.

---

## Quick Start

### Installation
```
git clone https://github.com/GhIrani33/ecgfounder-edge-cds.git
cd ecgfounder-edge-cds
pip install -r requirements.txt

# Download model weights (see setup/download_models.sh)
bash setup/download_models.sh
```

### Basic Inference
```
from src.preprocessing import ECGPreprocessor
from src.inference import ONNXInference

# Initialize
preprocessor = ECGPreprocessor()
model = ONNXInference("models/onnx/ecg_founder_150class.onnx")

# Process ECG (shape: [12 leads, 5000 samples] = 10 sec @ 500 Hz)
ecg_signal = load_ecg("data/sample.wfdb")  # Raw signal required
processed = preprocessor.process(ecg_signal)
predictions = model.predict(processed)

print(predictions.top_k(5))  # Top 5 diagnoses with probabilities
```

**Important:** System requires **raw digital ECG signals** (WFDB, DICOM, HL7 aECG formats), not scanned paper ECG images. Image-based ECG digitization introduces prohibitive accuracy degradation.

---

## Repository Structure

```
ecg-founder-edge-cds/
├── src/                           # Source code
│   ├── preprocessing/             # Signal processing pipeline
│   ├── models/                    # ECGFounder architecture- Place holder
│   ├── inference/                 # ONNX inference & conversion
│   ├── explainability/            # Attribution analysis
│   └── validation/                # Benchmark evaluation scripts
├── results/                       # Pre-computed validation results
│   ├── validation/                # Performance metrics (JSON)
│   ├── explainability/            # Attribution scores (CSV)
│   └── figures/                   # figures
│
├── docs/                          # Detailed documentation
│   ├── METHODS.md                 # Scientific methodology
│   
│   └── DEPLOYMENT.md              # Hardware setup guide
```

**Note:** Model weights and PTB-XL dataset are not included (large files). Download via scripts in `setup/`.

---

## Datasets & Resources

**PTB-XL Database** [2]
- 21,837 clinical 12-lead ECGs (10 seconds each, 500 Hz)
- Download: https://physionet.org/content/ptb-xl/1.0.3/
- License: CC BY 4.0

**ECGFounder Model** [1]
- Pre-trained on 10.5M ECGs from diverse clinical sources
- Original repository: https://github.com/NickLJLee/ECGFounder?tab=readme-ov-file
- Weights available upon request from authors

---

## Limitations & Future Work

### Current Status
✅ **Technical validation complete:** Model deployment, preprocessing, latency benchmarking, benchmark evaluation.
⚠️ **Explainability in progress:** Lead-level attribution validated; segment-level analysis under development.
❌ **Clinical validation pending:** Prospective trial with real patients required.

### Known Limitations
- Validated only on PTB-XL benchmark dataset (not real-world clinical data)
- 79 of 150 diagnostic classes lack ground truth validation (PTB-XL covers 71 classes)
- No regulatory approval (FDA/CE) - research use only
- Requires high-quality digital ECG signals (device integration needed)

### Next Steps
1. **Device integration:** HL7/DICOM parser for hospital ECG machines
2. **Retrospective validation:** 500+ ECGs from hospital database
3. **Prospective clinical trial:** IRB-approved validation study
4. **Regulatory pathway:** FDA 510(k) or CE Mark submission

---

## Disclaimer

⚠️ **Research Prototype:** This system has been validated on benchmark datasets only. It is **NOT FDA/CE approved** and must **NOT be used for primary clinical diagnosis** without:
- Institutional Review Board (IRB) approval
- Physician supervision
- Prospective clinical validation

**Intended use:** Research, proof-of-concept demonstration, and algorithm development for clinical decision support systems.

---

## Citation

```
@software{ecgfounder_edge_cds_2025,
  title={ECGFounder Real-Time Deployment: Proof-of-Concept for Edge-Based Clinical Decision Support},
  author={[Ghasem Dolatkhah laein]},
  year={2025},
  url={https://github.com/GhIrani33/ecgfounder-edge-cds},
  note={Real-time deployment of ECGFounder foundation model on ARM hardware}
}
```

Please also cite the original ECGFounder paper:
```
@article{li2024ecgfounder,
  title={An Electrocardiogram Foundation Model Built on over 10 Million Recordings with External Evaluation across Multiple Domains},
  author={Li, Qihan and others},
  journal={Nature Digital Medicine},
  year={2024}
}
```

---

## References

[1] Li, Q., et al. (2024). "An Electrocardiogram Foundation Model Built on over 10 Million Recordings with External Evaluation across Multiple Domains." *Nature Digital Medicine*.

[2] Wagner, P., et al. (2020). "PTB-XL, a large publicly available electrocardiography dataset." *Scientific Data*, 7(1), 154.

[3] Ribeiro, A. H., et al. (2020). "Automatic diagnosis of the 12-lead ECG using a deep neural network." *Nature Communications*, 11, 1760.

[4] Hannun, A. Y., et al. (2019). "Cardiologist-level arrhythmia detection." *Nature Medicine*, 25, 65-69.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

**Note:** ECGFounder model weights are subject to original authors' license terms.

---

## Acknowledgments

This work builds upon the ECGFounder foundation model developed by Li et al. [1]. Special thanks to the PTB-XL team for providing the benchmark dataset and the ONNX Runtime team for enabling cross-platform deployment.

---

## Contact

**Maintainer:** Ghasem  
**Email:** dr.ghasemdolatkhah@gmail.com  
**Issues:** [GitHub Issues](https://github.com/GhIrani33/ecgfounder-edge-cds/issues)

For questions about ECGFounder model, contact original authors.
