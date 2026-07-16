# -*- coding: utf-8 -*-
"""
File Name: DIHC_FeatureExtractor.py
Author: WWM Emran (Emran Ali)
Involvement: HumachLab & Deakin - Innovation in Healthcare (DIHC)


Updated: Refactored for modern SciPy compatibility, safety, and performance.
Updated by Shivapratap Gopakumar, Engineering for Health Group, Amrita Vishwa Vidyapeetham
"""

from __future__ import annotations
import collections
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy as sp
import scipy.signal as sig
from scipy.signal import butter, lfilter, welch
from scipy.stats import entropy as scipyEntropy

# Handle the deprecation/removal of simps in modern SciPy versions
try:
    from scipy.integrate import simpson
except ImportError:
    from scipy.integrate import simps as simpson  # type: ignore

from antropy import (
    app_entropy, detrended_fluctuation, higuchi_fd, hjorth_params, 
    katz_fd, lziv_complexity, num_zerocross, perm_entropy, 
    petrosian_fd, sample_entropy, spectral_entropy, svd_entropy
)
import pyeeg
from mne.time_frequency import psd_array_multitaper

try:
    if "ipykernel" in sys.modules:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Internal module imports (using relative paths to avoid circular/namespace clutter)
from .DIHC_EntropyProfile import DIHC_EntropyProfile
from .DIHC_FeatureDetails import DIHC_FeatureDetails, DIHC_FeatureGroup
from .DIHC_FeatureExtrantor_Helper import (
    compute_distribution_entropy_jit,
    compute_entropy_profile_jit,
    compute_fuzzy_entropy_jit
)


class DIHC_FeatureExtractor:

    def __init__(
        self, 
        manage_exceptional_data: int = 0, 
        signal_frequency: float = 256.0, 
        sample_per_second: int = 1280, 
        filtering_enabled: bool = False, 
        lowcut: float = 1.0, 
        highcut: float = 48.0, 
        verbose_progress: bool = False,
        **kwargs: Any  # Capture potential legacy backward-compatible fields
    ):
        self.manage_exceptional_data = manage_exceptional_data
        self.signal_frequency = signal_frequency
        self.sample_per_second = sample_per_second
        self.filtering_enabled = filtering_enabled
        self.lowcut = lowcut
        self.highcut = highcut
        
        # Backward compatibility layer for misspelled parameter
        self.verbose_progress = kwargs.get("varbose_progress", verbose_progress)

        # Feature groupings from central specifications
        self.td_linear_statistical = DIHC_FeatureDetails.td_linear_statistical
        self.td_nonlinear_entropy = DIHC_FeatureDetails.td_nonlinear_entropy
        self.td_nonlinear_complexity_and_fractal_dimensions = DIHC_FeatureDetails.td_nonlinear_complexity_and_fractal_dimensions
        self.td_nonlinear_samp_entropy_profiling = DIHC_FeatureDetails.td_nonlinear_samp_entropy_profiling
        self.fd_linear_statistical = DIHC_FeatureDetails.fd_linear_statistical
        self.fd_linear_statistical_binwise = DIHC_FeatureDetails.fd_linear_statistical_binwise
        self.fd_spectral_band_power = DIHC_FeatureDetails.fd_spectral_band_power
        self.band_frequency_list = DIHC_FeatureDetails.band_frequency_list

        self.band: Tuple[float, float] = (0.0, self.signal_frequency)
        self.feature_list = [
            self.td_linear_statistical, self.td_nonlinear_entropy,
            self.td_nonlinear_complexity_and_fractal_dimensions,
            self.td_nonlinear_samp_entropy_profiling, self.fd_linear_statistical,
            self.fd_linear_statistical_binwise, self.fd_spectral_band_power
        ]

        self.fd_data_dict: Optional[Dict[float, complex]] = None
        self.entropy_profile: Optional[np.ndarray] = None

    def get_new_features_to_calculate(self, feature_types: List[int]) -> List[str]:
        if len(feature_types) == 0:
            return [item for sublist in self.feature_list for item in sublist]
        return [item for i, sublist in enumerate(self.feature_list) for item in sublist if i in feature_types]

    def generate_features(self, seg_srl: int, seg_data: np.ndarray, feature_names: Optional[List[Any]]) -> pd.DataFrame:
        self.fd_data_dict = None
        self.entropy_profile = None 
        feature_values = []
        seg_values = np.asarray(seg_data).copy()

        if feature_names is None or len(feature_names) == 0:
            resolved_feature_names = DIHC_FeatureGroup.all.value
        elif not isinstance(feature_names[0], DIHC_FeatureGroup):
            raise TypeError("Invalid features specified. Elements must be instances of DIHC_FeatureGroup.")
        else:
            feature_names_copy = list(feature_names)
            extracted_names = []
            for itm in feature_names_copy:
                extracted_names.extend(itm.value)

            unique_features = list(set(extracted_names))
            all_feature_names = DIHC_FeatureGroup.all.value
            resolved_feature_names = [it for it in all_feature_names if it in unique_features] 

        if self.verbose_progress:
            prog_bar = tqdm(resolved_feature_names, desc="Feature extraction started...")
        else:
            prog_bar = resolved_feature_names

        for feat in prog_bar:
            if self.verbose_progress:
                prog_bar.set_description(f"For segment: {seg_srl}, extracting feature: {feat} ||")

            try:
                final_feat = feat
                final_data: Union[np.ndarray, List[float]] = seg_values

                if feat.startswith('fd_'):
                    if (feat in self.fd_linear_statistical) or (feat in self.fd_linear_statistical_binwise):
                        if self.fd_data_dict is None:
                            self.fd_data_dict = self.fd_spectralAmplitude(seg_values)

                        final_feat_list = feat.split('_')
                        fnl = len(final_feat_list)
                        if fnl > 1:
                            final_feat = final_feat_list[1]
                            final_data = list(self.fd_data_dict.values())
                            freq_keys = list(self.fd_data_dict.keys())

                            if fnl > 2:
                                band_range = self.band_frequency_list[final_feat_list[2]]
                                filtered_keys = [i for i in freq_keys if band_range[0] <= i < band_range[1]]
                                final_data = [self.fd_data_dict[x] for x in filtered_keys]

                    elif feat in self.fd_spectral_band_power:
                        final_feat_list = feat.split('_')
                        fnl = len(final_feat_list)
                        if fnl > 1:
                            final_feat = f'{final_feat_list[0]}_{final_feat_list[1]}'
                            self.band = (0.0, self.signal_frequency)
                            if fnl > 2:
                                self.band = self.band_frequency_list[final_feat_list[2]]

                elif feat.startswith('entropyProfiled_'):
                    if self.entropy_profile is None:
                        entProf_obj = DIHC_EntropyProfile()
                        enProf = entProf_obj.get_sample_entropy_profile(final_data)
                        dat = np.asarray(enProf)
                        if isinstance(dat[0], (list, np.ndarray)):
                            dat2 = [np.float64(item) for sublist in dat for item in sublist]
                        else:
                            dat2 = [np.float64(item) for item in dat]
                        self.entropy_profile = np.array(dat2, dtype=np.float64)

                    final_feat_list = feat.split('_')
                    final_feat = final_feat_list[1]
                    final_data = self.entropy_profile

                method = getattr(self, final_feat)

            except AttributeError as exc:
                raise NotImplementedError(
                    f"Class `{self.__class__.__name__}` does not implement `{final_feat}`"
                ) from exc

            feat_val = 0.0
            final_data_arr = np.asarray(final_data, dtype=np.float64)
            is_invariant = np.all(final_data_arr == final_data_arr[0]) if final_data_arr.size > 1 else True
            
            if not is_invariant:
                feat_val = method(final_data_arr)

            feature_values.append([round(feat_val, 2)])

        if self.verbose_progress and hasattr(prog_bar, 'close'):
            prog_bar.close()

        np_feat_value = np.array(feature_values).T
        all_features = pd.DataFrame()
        if len(resolved_feature_names) > 0 and len(np_feat_value) > 0:
            all_features = pd.DataFrame(np_feat_value, columns=resolved_feature_names)

        # Handling exceptional floating data limits
        if self.manage_exceptional_data == 0:
            all_features = all_features[all_features != np.inf]
        elif self.manage_exceptional_data == 1:
            all_features = all_features[all_features != np.inf].fillna(0)
        elif self.manage_exceptional_data == 2:
            all_features = all_features[all_features != np.inf].dropna()
        elif self.manage_exceptional_data == 3:
            all_features = all_features[all_features != np.inf]
            all_features = all_features.fillna(all_features.mean()) 

        return all_features 

    # ==========================================
    # Time Domain Metric Processing
    # ==========================================

    def total(self, data: np.ndarray) -> float:
        return float(np.sum(data))

    def summation(self, data: np.ndarray) -> float:
        return float(np.sum(data))

    def average(self, data: np.ndarray) -> float:
        return float(np.mean(data))

    def minimum(self, data: np.ndarray) -> float:
        return float(np.min(data))

    def maximum(self, data: np.ndarray) -> float:
        return float(np.max(data))

    def mean(self, data: np.ndarray) -> float:
        return float(np.mean(data))

    def median(self, data: np.ndarray) -> float:
        return float(np.median(data))

    def standardDeviation(self, data: np.ndarray) -> float:
        return float(np.std(data))

    def variance(self, data: np.ndarray) -> float:
        return float(np.var(data))

    def kurtosis(self, data: np.ndarray) -> float:
        return float(sp.stats.kurtosis(data))

    def skewness(self, data: np.ndarray) -> float:
        return float(sp.stats.skew(data))

    def peakOrMax(self, data: np.ndarray) -> float:
        return self.maximum(data)

    def numberOfPeaks(self, data: np.ndarray) -> int:
        peaks, _ = sig.find_peaks(data)
        return len(peaks)

    def numberOfZeroCrossing(self, data: np.ndarray) -> int:
        return int(num_zerocross(data))

    def positiveToNegativeSampleRatio(self, data: np.ndarray) -> float:
        try:
            pos = np.sum(data >= 0)
            neg = np.sum(data < 0)
            return float(pos / neg) if neg != 0 else 0.0
        except ZeroDivisionError:
            return 0.0

    def positiveToNegativePeakRatio(self, data: np.ndarray) -> float:
        try:
            pos_peaks = len(sig.find_peaks(data)[0])
            neg_peaks = len(sig.find_peaks(-data)[0])
            return float(pos_peaks / neg_peaks) if neg_peaks != 0 else 0.0
        except ZeroDivisionError:
            return 0.0

    def meanAbsoluteValue(self, data: np.ndarray) -> float:
        return self.mean(np.abs(data))

    # ==========================================
    # Non-linear Entropy Architectures
    # ==========================================

    def approximateEntropy(self, data: np.ndarray) -> float:
        try:
            return float(app_entropy(data))
        except ZeroDivisionError:
            return 0.0

    def sampleEntropy(self, data: np.ndarray) -> float:
        if len(data) < 5000:
            return 0.0
        try:
            return float(sample_entropy(data))
        except ZeroDivisionError:
            return 0.0

    def permutationEntropy(self, data: np.ndarray) -> float:
        try:
            return float(perm_entropy(data))
        except ZeroDivisionError:
            return 0.0

    def spectralEntropy(self, data: np.ndarray) -> float:
        try:
            return float(spectral_entropy(data, self.signal_frequency, method='welch'))
        except ZeroDivisionError:
            return 0.0

    def singularValueDecompositionEntropy(self, data: np.ndarray) -> float:
        try:
            return float(svd_entropy(data))
        except ZeroDivisionError:
            return 0.0

    def fuzzyEntropy(self, data: np.ndarray, m: int = 2, tau: int = 1, r: float = 0.2) -> float:
        try:
            return float(compute_fuzzy_entropy_jit(np.asarray(data).flatten(), m, tau, r))
        except Exception:
            return 0.0

    def distributionEntropy(self, data: np.ndarray, m: int = 2, M: int = 500) -> float:
        flat_data = np.asarray(data).flatten()
        if len(flat_data) <= m:
            raise ValueError("Input data length must be strictly greater than embedding dimension m.")
        return float(compute_distribution_entropy_jit(flat_data, m, M))

    def distributionEntropy4(self, data: np.ndarray) -> float:
        return self.distributionEntropy(data, m=4)

    def distributionEntropy6(self, data: np.ndarray) -> float:
        return self.distributionEntropy(data, m=6)

    def distributionEntropy8(self, data: np.ndarray) -> float:
        return self.distributionEntropy(data, m=8)

    def distributionEntropy10(self, data: np.ndarray) -> float:
        return self.distributionEntropy(data, m=10)

    def shannonEntropy(self, data: np.ndarray, m: int = 2) -> float:
        bases = collections.Counter(data)
        dist = [i / sum(bases.values()) for i in bases.values()]
        return float(scipyEntropy(dist, base=m))

    def _x_log2_x(self, data: np.ndarray) -> np.ndarray:
        res = data * np.log2(data)
        if np.size(data) == 1:
            if np.isclose(data, 0.0):
                res = 0.0
        else:
            res[np.isclose(data, 0.0)] = 0.0
        return res

    def renyiEntropy(self, data: np.ndarray, alpha: float = 2.0) -> float:
        assert alpha >= 0, f"renyi_entropy requires values of alpha >= 0, but alpha = {alpha}."
        if np.isinf(alpha):
            return float(-np.log2(np.max(data)))
        elif np.isclose(alpha, 0.0):
            return float(np.log2(len(data)))
        elif np.isclose(alpha, 1.0):
            return float(-np.sum(self._x_log2_x(data)))
        else:
            return float((1.0 / (1.0 - alpha)) * np.log2(np.sum(data ** alpha))) 

    # ==========================================
    # Fractal Dimension Processing
    # ==========================================

    def hjorthMobility(self, data: np.ndarray) -> float:
        try:
            hjm, _ = hjorth_params(data)
            return float(hjm)
        except ZeroDivisionError:
            return 0.0

    def hjorthComplexity(self, data: np.ndarray) -> float:
        try:
            _, hjc = hjorth_params(data)
            return float(hjc)
        except ZeroDivisionError:
            return 0.0

    def hurstExponent(self, data: np.ndarray) -> float:
        try:
            return float(pyeeg.hurst(data))
        except ZeroDivisionError:
            return 0.0

    def fisherInfo(self, data: np.ndarray, tau: int = 1, m: int = 2) -> float:
        try:
            return float(pyeeg.fisher_info(data, tau, m))
        except ZeroDivisionError:
            return 0.0

    def lempelZivComplexity(self, data: np.ndarray) -> float:
        try:
            return float(lziv_complexity(data))
        except ZeroDivisionError:
            return 0.0

    def petrosianFd(self, data: np.ndarray) -> float:
        try:
            return float(petrosian_fd(data))
        except ZeroDivisionError:
            return 0.0

    def katzFd(self, data: np.ndarray) -> float:
        try:
            return float(katz_fd(data))
        except ZeroDivisionError:
            return 0.0

    def higuchiFd(self, data: np.ndarray) -> float:
        try:
            return float(higuchi_fd(data))
        except ZeroDivisionError:
            return 0.0

    def detrendedFluctuation(self, data: np.ndarray) -> float:
        try:
            return float(detrended_fluctuation(data))
        except ZeroDivisionError:
            return 0.0 

    # ==========================================
    # Frequency Domain Features & Filtering
    # ==========================================

    def _butter_bandpass(self, lowcut: float, highcut: float, fs: float, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def _butter_bandpass_filter(self, data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5) -> np.ndarray:
        b, a = self._butter_bandpass(lowcut, highcut, fs, order=order)
        return lfilter(b, a, data)

    def _fast_fourier_transformation(self, data: np.ndarray, fs: float, fft_type: int = 2) -> Dict[float, Any]:
        if fft_type == 0:
            fft_data = sp.fft.fft(data)
            freqs = sp.fft.fftfreq(len(data)) * fs
            return {freqs[i]: fft_data[i] for i in range(len(freqs))}
        elif fft_type == 1:
            fft_data = sp.fftpack.fft(data)
            freqs = sp.fftpack.fftfreq(len(data)) * fs
            return {freqs[i]: fft_data[i] for i in range(len(freqs))}
        else:
            fft_data = np.abs(np.fft.fft(data))
            freqs = np.fft.fftfreq(len(data), d=1.0 / fs)
            return {freqs[i]: fft_data[i] for i in range(len(freqs))}

    def fd_spectralAmplitude(self, data: np.ndarray) -> Dict[float, Any]:
        filtered_data = data
        if self.filtering_enabled:
            filtered_data = self._butter_bandpass_filter(
                data, self.lowcut, self.highcut, self.sample_per_second, order=6
            )
        return self._fast_fourier_transformation(filtered_data, self.sample_per_second)

    def fd_bandPower(self, data: np.ndarray, method: str = 'multitaper', window_sec: Optional[float] = None, relative: bool = False) -> float:
        sf = self.signal_frequency
        low, high = self.band

        if len(data) < 2:
            return 0.0

        try:
            if method == 'welch':
                nperseg = int(window_sec * sf) if window_sec is not None else int((2 / low) * sf) if low != 0 else len(data)
                nperseg = min(len(data), nperseg)
                if nperseg < 2:
                    return 0.0
                freqs, psd = welch(data, sf, nperseg=nperseg)
            else:
                psd, freqs = psd_array_multitaper(data, sf, adaptive=True, normalization='full', verbose=0)

            if len(psd) == 0 or len(freqs) == 0:
                return 0.0

            freq_res = freqs[1] - freqs[0]
            idx_band = np.logical_and(freqs >= low, freqs <= high)

            if not np.any(idx_band):
                return 0.0

            # Safe evaluation using upgraded simpson integration
            bp = float(simpson(psd[idx_band], dx=freq_res))

            if relative:
                total_power = simpson(psd, dx=freq_res)
                bp = bp / total_power if total_power != 0 else 0.0

            return bp

        except Exception:
            return 0.0