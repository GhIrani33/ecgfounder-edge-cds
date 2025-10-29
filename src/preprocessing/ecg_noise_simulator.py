# -*- coding: utf-8 -*-
"""
ecg_noise_simulator.py

Author: Ghasem
https://github.com/GhIrani33
"""

import numpy as np
from scipy import signal as sp_signal


class ECGNoiseSimulator:
    """
    Add realistic artifacts to clean hospital-grade ECG signals
    
    IMPORTANT: Auto-detects signal unit (mV vs μV) and scales noise appropriately
    """
    
    @staticmethod
    def detect_signal_scale(ecg: np.ndarray) -> float:
        """
        Detect if signal is in mV or μV based on typical amplitudes
        
        Returns:
        --------
        scale_factor : float
            1.0 if signal in μV, 0.001 if signal in mV
        """
        signal_std = np.std(ecg)
        
        # Typical ECG std in μV: 100-500 μV
        # Typical ECG std in mV: 0.1-0.5 mV
        if signal_std < 1.0:
            # Likely in mV
            return 0.001
        else:
            # Likely in μV
            return 1.0
    
    @staticmethod
    def add_powerline_interference(
        ecg: np.ndarray, 
        fs: int = 500, 
        amplitude_uv: float = 100, 
        freq_hz: float = 50
    ) -> np.ndarray:
        """Add sinusoidal powerline noise"""
        t = np.arange(ecg.shape[1]) / fs
        powerline = amplitude_uv * np.sin(2 * np.pi * freq_hz * t)
        return ecg + powerline[np.newaxis, :]
    
    @staticmethod
    def add_baseline_drift(
        ecg: np.ndarray, 
        fs: int = 500, 
        drift_amplitude_uv: float = 200
    ) -> np.ndarray:
        """Add low-frequency baseline wander"""
        t = np.arange(ecg.shape[1]) / fs
        drift = drift_amplitude_uv * (
            0.5 * np.sin(2 * np.pi * 0.15 * t) +
            0.3 * np.sin(2 * np.pi * 0.33 * t) +
            0.2 * np.cumsum(np.random.randn(len(t))) / len(t)
        )
        return ecg + drift[np.newaxis, :]
    
    @staticmethod
    def add_gaussian_noise(ecg: np.ndarray, noise_std_uv: float = 50) -> np.ndarray:
        """Add white Gaussian noise"""
        noise = np.random.normal(0, noise_std_uv, ecg.shape)
        return ecg + noise
    
    @staticmethod
    def add_muscle_artifact(
        ecg: np.ndarray, 
        fs: int = 500, 
        amplitude_uv: float = 150, 
        freq_range: tuple = (15, 40)
    ) -> np.ndarray:
        """Add high-frequency EMG contamination"""
        t = np.arange(ecg.shape[1]) / fs
        nyquist = fs / 2
        low_norm = freq_range[0] / nyquist
        high_norm = freq_range[1] / nyquist
        
        b, a = sp_signal.butter(4, [low_norm, high_norm], btype='band')
        muscle = amplitude_uv * sp_signal.filtfilt(b, a, np.random.randn(len(t)))
        
        return ecg + muscle[np.newaxis, :]
    
    @staticmethod
    def add_electrode_motion(
        ecg: np.ndarray, 
        fs: int = 500, 
        n_events: int = 3, 
        amplitude_uv: float = 500
    ) -> np.ndarray:
        """Add transient spikes from electrode movement"""
        noisy_ecg = ecg.copy()
        
        for _ in range(n_events):
            lead_idx = np.random.randint(0, ecg.shape[0])
            time_idx = np.random.randint(500, ecg.shape[1] - 500)
            
            pulse_width = int(0.05 * fs)
            pulse = amplitude_uv * sp_signal.windows.gaussian(pulse_width, std=pulse_width / 6)
            
            noisy_ecg[lead_idx, time_idx:time_idx + pulse_width] += pulse
        
        return noisy_ecg
    
    @classmethod
    def add_realistic_noise_mix(
        cls, 
        ecg: np.ndarray, 
        fs: int = 500, 
        noise_level: str = 'medium'
    ) -> np.ndarray:
        """
        Add combination of realistic noise sources
        
        AUTO-SCALES noise amplitude based on signal unit (mV vs μV)
        """
        noisy_ecg = ecg.copy()
        
        # ✅ AUTO-DETECT SIGNAL SCALE
        scale_factor = cls.detect_signal_scale(ecg)
        
        print(f"[Noise Simulator] Detected scale_factor = {scale_factor:.4f} " +
              f"({'mV' if scale_factor < 0.1 else 'μV'})")
        
        if noise_level == 'low':
            noisy_ecg = cls.add_powerline_interference(noisy_ecg, fs, amplitude_uv=50 * scale_factor)
            noisy_ecg = cls.add_baseline_drift(noisy_ecg, fs, drift_amplitude_uv=100 * scale_factor)
            noisy_ecg = cls.add_gaussian_noise(noisy_ecg, noise_std_uv=20 * scale_factor)
        
        elif noise_level == 'medium':
            noisy_ecg = cls.add_powerline_interference(noisy_ecg, fs, amplitude_uv=150 * scale_factor)
            noisy_ecg = cls.add_baseline_drift(noisy_ecg, fs, drift_amplitude_uv=250 * scale_factor)
            noisy_ecg = cls.add_gaussian_noise(noisy_ecg, noise_std_uv=50 * scale_factor)
            noisy_ecg = cls.add_muscle_artifact(noisy_ecg, fs, amplitude_uv=100 * scale_factor)
        
        elif noise_level == 'high':
            noisy_ecg = cls.add_powerline_interference(noisy_ecg, fs, amplitude_uv=300 * scale_factor)
            noisy_ecg = cls.add_baseline_drift(noisy_ecg, fs, drift_amplitude_uv=500 * scale_factor)
            noisy_ecg = cls.add_gaussian_noise(noisy_ecg, noise_std_uv=100 * scale_factor)
            noisy_ecg = cls.add_muscle_artifact(noisy_ecg, fs, amplitude_uv=200 * scale_factor)
            noisy_ecg = cls.add_electrode_motion(noisy_ecg, fs, n_events=2, amplitude_uv=800 * scale_factor)
        
        else:
            raise ValueError(f"Unknown noise_level: {noise_level}. Use 'low', 'medium', or 'high'")
        
        # Report SNR
        signal_power = np.var(ecg)
        noise_power = np.var(noisy_ecg - ecg)
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
        print(f"[Noise Simulator] SNR = {snr_db:.1f} dB")
        
        return noisy_ecg

