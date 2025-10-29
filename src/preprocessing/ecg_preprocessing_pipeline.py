# -*- coding: utf-8 -*-
"""
ecg_preprocessing_pipeline.py - FINAL VERSION FOR HOSPITAL-GRADE ECG

Target Use Case:
- Hospital-grade 12-lead ECG devices (Schiller, GE, Philips, etc.)
- Hardware analog filtering: 0.05-150 Hz
- Resolution: ≥16-bit ADC
- 500 Hz sampling rate
- Gel electrodes with low impedance

Preprocessing Philosophy:
- Matches ECGFounder training protocol (0.5-50 Hz bandpass + z-score)
- Adaptive notch filter (only applies if powerline interference detected)
- Minimal QC to avoid false rejections on clean hospital data
- Scientifically sound: preserves train-test distribution consistency

Author: Ghasem

"""

import warnings
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample, welch
from typing import Tuple, Dict, List

warnings.filterwarnings("ignore")


class ECGPreprocessor:
    """
    ECG Preprocessor for Hospital-Grade 12-Lead ECG Devices
    
    Configuration:
    - Bandpass: 0.5-50 Hz (matches ECGFounder training)
    - Notch: Adaptive (only if powerline SNR > 10 dB)
    - QC: Lenient (designed for clean hospital data)
    """
    
    def __init__(
        self,
        # ========== TARGET FORMAT (NEVER CHANGE) ==========
        target_fs: int = 500,
        target_length: int = 5000,
        
        # ========== FILTERING (MATCHES ECGFounder TRAINING) ==========
        use_bandpass: bool = True,
        bandpass_low_hz: float = 0.5,      # ✅ ECGFounder value
        bandpass_high_hz: float = 50.0,    # ✅ ECGFounder value
        bandpass_order: int = 2,           # 2nd-order Butterworth
        
        use_notch: bool = True,
        notch_freq_hz: float = 50.0,       # 50 Hz for Europe/Asia, 60 Hz for Americas
        notch_quality: float = 30.0,
        notch_threshold_db: float = 10.0,  # Apply notch only if powerline SNR > 10 dB
        
        # ========== QUALITY CONTROL ==========
        qc_mode: str = "lenient",          # "strict" | "lenient" | "disabled"
        qc_snr_threshold_db: float = 5.0,
        qc_saturation_uv: float = 15000.0,
        qc_min_std_uv: float = 5.0,
        qc_max_std_uv: float = 5000.0,
    ):
        """
        Initialize ECG Preprocessor
        
        Parameters:
        -----------
        target_fs : int
            Target sampling frequency (default: 500 Hz)
        target_length : int
            Target number of samples (default: 5000 = 10 seconds @ 500 Hz)
        use_bandpass : bool
            Enable bandpass filtering (default: True)
        bandpass_low_hz : float
            Lower cutoff frequency (default: 0.5 Hz, matches ECGFounder)
        bandpass_high_hz : float
            Upper cutoff frequency (default: 50 Hz, matches ECGFounder)
        use_notch : bool
            Enable adaptive notch filtering (default: True)
        notch_freq_hz : float
            Notch filter center frequency (default: 50 Hz)
        notch_threshold_db : float
            Minimum powerline SNR to trigger notch filter (default: 10 dB)
        qc_mode : str
            Quality control mode: "strict", "lenient", or "disabled"
        """
        self.target_fs = int(target_fs)
        self.target_length = int(target_length)
        
        self.use_bandpass = bool(use_bandpass)
        self.bandpass_low_hz = float(bandpass_low_hz)
        self.bandpass_high_hz = float(bandpass_high_hz)
        self.bandpass_order = int(bandpass_order)
        
        self.use_notch = bool(use_notch)
        self.notch_freq_hz = float(notch_freq_hz)
        self.notch_quality = float(notch_quality)
        self.notch_threshold_db = float(notch_threshold_db)
        
        self.qc_mode = str(qc_mode).lower()
        self.qc_snr_threshold_db = float(qc_snr_threshold_db)
        self.qc_saturation_uv = float(qc_saturation_uv)
        self.qc_min_std_uv = float(qc_min_std_uv)
        self.qc_max_std_uv = float(qc_max_std_uv)
        
        # Standard 12-lead order
        self.standard_lead_order = ["I", "II", "III", "aVR", "aVL", "aVF", 
                                     "V1", "V2", "V3", "V4", "V5", "V6"]
    
    def process(
        self, 
        raw_ecg: np.ndarray, 
        lead_names: List[str], 
        original_fs: int
    ) -> Tuple[np.ndarray, Dict]:
        """
        Process raw ECG to ECGFounder-compatible format
        
        Parameters:
        -----------
        raw_ecg : np.ndarray
            Raw ECG signal, shape (n_leads, n_samples)
        lead_names : List[str]
            Names of leads (e.g., ["I", "II", "V1", ...])
        original_fs : int
            Original sampling frequency in Hz
        
        Returns:
        --------
        processed_ecg : np.ndarray
            Processed ECG, shape (12, 5000), dtype float32
        metadata : Dict
            Processing metadata and QC results
        """
        metadata = {
            "original_fs": original_fs,
            "original_length": raw_ecg.shape[1],
            "qc_pass": True,
            "qc_warnings": [],
            "processing_steps": []
        }
        
        ecg = raw_ecg.astype(np.float32)
        
        # Step 1: Resample to target frequency
        if original_fs != self.target_fs:
            ecg = self._resample(ecg, original_fs, self.target_fs)
            metadata["processing_steps"].append(f"resampled_{original_fs}Hz_to_{self.target_fs}Hz")
        
        # Step 2: Bandpass filter (0.5-50 Hz)
        if self.use_bandpass:
            ecg = self._bandpass_filter(ecg, self.target_fs)
            metadata["processing_steps"].append(f"bandpass_{self.bandpass_low_hz}-{self.bandpass_high_hz}Hz")
        
        # Step 3: Adaptive notch filter (only if powerline detected)
        if self.use_notch:
            powerline_detected, snr_db = self._detect_powerline(ecg, self.target_fs)
            metadata["powerline_snr_db"] = float(snr_db)
            
            if powerline_detected:
                ecg = self._notch_filter(ecg, self.target_fs)
                metadata["processing_steps"].append(f"notch_{self.notch_freq_hz}Hz_applied")
            else:
                metadata["processing_steps"].append(f"notch_skipped_SNR_{snr_db:.1f}dB")
        
        # Step 4: Ensure 12 standard leads
        ecg, lead_warnings = self._ensure_12_leads(ecg, lead_names)
        if lead_warnings:
            metadata["qc_warnings"].extend(lead_warnings)
        
        # Step 5: Length standardization (truncate/pad to 5000 samples)
        ecg = self._standardize_length(ecg, self.target_length)
        
        # Step 6: Z-score normalization (per lead)
        ecg = self._zscore_normalize(ecg)
        metadata["processing_steps"].append("zscore_normalized")
        
        # Step 7: Quality control
        if self.qc_mode != "disabled":
            qc_results = self._quality_control(ecg, self.target_fs)
            metadata.update(qc_results)
        
        return ecg, metadata
    
    def _resample(self, signal: np.ndarray, fs_src: int, fs_dst: int) -> np.ndarray:
        """Resample signal to target frequency"""
        if fs_src == fs_dst:
            return signal
        
        n_samples_dst = int(round(signal.shape[1] * fs_dst / fs_src))
        resampled = np.zeros((signal.shape[0], n_samples_dst), dtype=signal.dtype)
        
        for i in range(signal.shape[0]):
            resampled[i] = resample(signal[i], n_samples_dst)
        
        return resampled
    
    def _bandpass_filter(self, signal: np.ndarray, fs: int) -> np.ndarray:
        """Apply Butterworth bandpass filter"""
        nyquist = 0.5 * fs
        low = self.bandpass_low_hz / nyquist
        high = self.bandpass_high_hz / nyquist
        
        # Prevent filter instability
        low = max(low, 0.001)
        high = min(high, 0.999)
        
        b, a = butter(self.bandpass_order, [low, high], btype='band')
        
        filtered = np.zeros_like(signal)
        for i in range(signal.shape[0]):
            filtered[i] = filtfilt(b, a, signal[i])
        
        return filtered
    
    def _detect_powerline(self, signal: np.ndarray, fs: int) -> Tuple[bool, float]:
        """
        Detect if powerline interference is present
        
        Returns:
        --------
        detected : bool
            True if powerline SNR > threshold
        snr_db : float
            Powerline signal-to-noise ratio in dB
        """
        # Analyze lead II (most representative)
        lead_idx = 1 if signal.shape[0] > 1 else 0
        freqs, psd = welch(signal[lead_idx], fs=fs, nperseg=min(1024, signal.shape[1]))
        
        # Power at notch frequency
        idx_notch = np.argmin(np.abs(freqs - self.notch_freq_hz))
        power_notch = psd[idx_notch]
        
        # Background power (±5 Hz around notch, excluding ±2 Hz)
        freq_low = self.notch_freq_hz - 5
        freq_high = self.notch_freq_hz + 5
        idx_bg = np.where(
            (freqs >= freq_low) & (freqs <= freq_high) & 
            (np.abs(freqs - self.notch_freq_hz) > 2.0)
        )[0]
        
        power_bg = np.mean(psd[idx_bg]) if len(idx_bg) > 0 else 1e-10
        
        # SNR in dB
        snr_db = 10 * np.log10(power_notch / power_bg + 1e-10)
        
        return snr_db > self.notch_threshold_db, snr_db
    
    def _notch_filter(self, signal: np.ndarray, fs: int) -> np.ndarray:
        """Apply IIR notch filter at powerline frequency"""
        b, a = iirnotch(w0=self.notch_freq_hz, Q=self.notch_quality, fs=fs)
        
        filtered = np.zeros_like(signal)
        for i in range(signal.shape[0]):
            filtered[i] = filtfilt(b, a, signal[i])
        
        return filtered
    
    def _ensure_12_leads(self, signal: np.ndarray, lead_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Ensure output has 12 standard leads
        Derives augmented limb leads from I and II if needed
        """
        warnings = []
        
        # Create dictionary of available leads
        lead_dict = {name: signal[i] for i, name in enumerate(lead_names) if i < signal.shape[0]}
        
        # Derive augmented leads if I and II are available
        if "I" in lead_dict and "II" in lead_dict:
            if "III" not in lead_dict:
                lead_dict["III"] = lead_dict["II"] - lead_dict["I"]
            if "aVR" not in lead_dict:
                lead_dict["aVR"] = -(lead_dict["I"] + lead_dict["II"]) / 2.0
            if "aVL" not in lead_dict:
                lead_dict["aVL"] = lead_dict["I"] - lead_dict["II"] / 2.0
            if "aVF" not in lead_dict:
                lead_dict["aVF"] = lead_dict["II"] - lead_dict["I"] / 2.0
        
        # Build 12-lead array
        n_samples = signal.shape[1]
        ecg_12lead = np.zeros((12, n_samples), dtype=signal.dtype)
        
        for i, lead_name in enumerate(self.standard_lead_order):
            if lead_name in lead_dict:
                ecg_12lead[i] = lead_dict[lead_name]
            else:
                # Zero-pad missing leads
                ecg_12lead[i] = np.zeros(n_samples)
                warnings.append(f"lead_{lead_name}_missing_zero_padded")
        
        return ecg_12lead, warnings
    
    def _standardize_length(self, signal: np.ndarray, target_len: int) -> np.ndarray:
        """Truncate or zero-pad to target length"""
        current_len = signal.shape[1]
        
        if current_len == target_len:
            return signal
        
        if current_len > target_len:
            # Center crop
            start = (current_len - target_len) // 2
            return signal[:, start:start + target_len]
        else:
            # Zero-pad (center)
            pad_total = target_len - current_len
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            return np.pad(signal, ((0, 0), (pad_left, pad_right)), mode='constant')
    
    def _zscore_normalize(self, signal: np.ndarray) -> np.ndarray:
        """Z-score normalization (per lead)"""
        normalized = np.zeros_like(signal, dtype=np.float32)
        
        for i in range(signal.shape[0]):
            mean = np.mean(signal[i])
            std = np.std(signal[i])
            
            if std > 1e-8:
                normalized[i] = (signal[i] - mean) / std
            else:
                normalized[i] = signal[i] - mean  # Avoid division by zero
        
        # Handle NaN/Inf
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        return normalized
    
    def _quality_control(self, signal: np.ndarray, fs: int) -> Dict:
        """Quality control checks"""
        qc_result = {
            "qc_pass": True,
            "qc_warnings": []
        }
        
        for i, lead_name in enumerate(self.standard_lead_order):
            lead_signal = signal[i]
            
            # Check 1: All zeros (lead-off)
            if np.all(lead_signal == 0.0):
                qc_result["qc_warnings"].append(f"{lead_name}_all_zeros")
                if self.qc_mode == "strict":
                    qc_result["qc_pass"] = False
                continue
            
            # Check 2: Near-constant (poor contact)
            if np.std(lead_signal) < (self.qc_min_std_uv / 1000.0):  # Assuming normalized
                qc_result["qc_warnings"].append(f"{lead_name}_low_variance")
            
            # Check 3: Excessive noise
            if np.std(lead_signal) > (self.qc_max_std_uv / 100.0):
                qc_result["qc_warnings"].append(f"{lead_name}_high_variance")
            
            # Check 4: Saturation
            if np.max(np.abs(lead_signal)) > (self.qc_saturation_uv / 100.0):
                qc_result["qc_warnings"].append(f"{lead_name}_possible_saturation")
                if self.qc_mode == "strict":
                    qc_result["qc_pass"] = False
        
        return qc_result
