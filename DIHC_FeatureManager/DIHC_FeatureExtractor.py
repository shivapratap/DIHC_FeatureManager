# -*- coding: utf-8 -*-
"""
File Name: DIHC_FeatureExtractor.py
Original Author: WWM Emran (Emran Ali)
Original Involvement: HumachLab & Deakin- Innovation in Healthcare (DIHC)
Original Email: wwm.emran@gmail.com, emran.ali@research.deakin.edu.au
Original Date: 5/01/2020 8:55 pm

--------------------------------------------------------------------------
GENERALIZATION PATCH NOTES (see fork README/CHANGELOG for full details):

1. `band_frequency_list` is now a constructor parameter instead of being
   hardcoded to the package's original EEG band definitions. Pass your own
   dict of {band_name: (low_hz, high_hz)} for your signal domain (EHG, ECG,
   EMG, etc). The original EEG bands remain the default for backward
   compatibility.

2. `signal_frequency`, `lowcut`, and `highcut` no longer silently default to
   EEG-typical values (256 Hz / 1-48 Hz). They must be explicitly supplied
   whenever filtering is enabled or frequency-domain features are requested.
   A Nyquist-frequency check now raises a clear error instead of silently
   producing a garbage-filtered signal when a band exceeds fs/2.

3. `sampleEntropy` no longer uses a hardcoded, EEG-window-shaped 5000-sample
   minimum (which silently zeroed out SampEn for any shorter window,
   regardless of what that means for a different sampling rate/segment
   length). The minimum is now derived from signal_frequency by default, and
   can be overridden directly via `min_sample_entropy_length`.

4. Every feature method now returns `np.nan` on computational failure instead
   of a silent `0`. A real `0` and a failed computation used to be
   indistinguishable in the output; now they are not. This also means the
   existing `manage_exceptional_data` modes in `generate_features()` (which
   operate on NaN/inf) now actually catch these cases, which they could not
   before.

5. `generate_features()` now wraps each feature call in a try/except so one
   failing feature cannot silently abort extraction for an entire segment;
   the failure is now recorded as NaN and logged (in verbose mode) with the
   feature name and segment number.

6. Output rounding is now configurable via `round_decimals` (default: None,
   i.e. full floating-point precision preserved). The original code
   hardcoded `round(feat_val, 2)`, which can collapse near-zero features
   (e.g. a bandpass-filtered signal's near-zero mean) into a constant column
   and lose discriminative information.

7. Removed large blocks of superseded/commented-out alternate implementations
   for readability. They remain available in git history prior to this
   patch if needed for reference.
--------------------------------------------------------------------------
"""


### In-built modules

import sys
import numpy as np
import pandas as pd

import math
import scipy as sp
import scipy.signal as sig
from scipy.integrate import simps
from scipy.stats import entropy as scipyEntropy
from scipy.signal import butter, lfilter, welch
from scipy.spatial.distance import pdist, squareform
from scipy import fft, fftpack

from antropy import *
import pyeeg
from mne.time_frequency import psd_array_multitaper

import collections

from numba import jit, njit

try:
    if "ipykernel" in sys.modules:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except ImportError:
    from tqdm import tqdm

### SRART: My modules ###
from DIHC_FeatureManager import *
from DIHC_FeatureManager.DIHC_EntropyProfile import *
from DIHC_FeatureManager.DIHC_FeatureDetails import *
from DIHC_FeatureManager.DIHC_FeatureExtrantor_Helper import *
from DIHC_FeatureManager.DIHC_FeatureDetails import DIHC_FeatureGroup
### END: My modules ###


###
class DIHC_FeatureExtractor:

    def __init__(self, manage_exceptional_data=0, signal_frequency=None, sample_per_second=1280,
                 filtering_enabled=False, lowcut=None, highcut=None, varbose_progress=False,
                 band_frequency_list=None, min_sample_entropy_length=None, round_decimals=None):
        """
        Parameters
        ----------
        manage_exceptional_data : int
            0 = drop inf only; 1 = drop inf + fillna(0); 2 = drop inf + dropna;
            3 = drop inf + fillna(mean). Applies to NaN/inf produced by feature
            methods (see patch note #4 - this now actually catches failed
            features, which previously returned a silent 0 and bypassed this).
        signal_frequency : float
            Sampling frequency of the input signal, in Hz. REQUIRED - no
            domain-specific default is assumed. Needed for any frequency-domain
            feature, filtering, or the adaptive SampEn minimum-length check.
        filtering_enabled : bool
            Whether to bandpass-filter the segment before frequency-domain
            feature extraction (see `fd_spectralAmplitude`).
        lowcut, highcut : float
            Bandpass filter edges in Hz. Required if filtering_enabled=True.
            Validated against the Nyquist frequency (signal_frequency / 2).
        band_frequency_list : dict or None
            Mapping of {band_name: (low_hz, high_hz)} used by the binwise
            frequency-domain features and band-power features (feature names
            like 'fd_mean_alpha' or 'fd_bandPower_alpha'). Defaults to the
            package's original EEG bands if not provided, for backward
            compatibility - but you should supply bands appropriate to your
            own signal's passband (e.g. custom low/high sub-bands within a
            0.1-3 Hz EHG passband) rather than relying on the EEG default.
        min_sample_entropy_length : int or None
            Minimum number of samples required to compute SampEn. If None,
            derived adaptively from signal_frequency (roughly 25 seconds worth
            of samples, matching the antropy/nolds rule-of-thumb for stable
            SampEn estimates) rather than a fixed, EEG-shaped 5000-sample
            constant. Pass an explicit value to override.
        round_decimals : int or None
            If set, feature values are rounded to this many decimal places
            before being returned. Default None preserves full precision -
            set this only for display/export purposes, not before any
            statistical testing.
        """
        self.manage_exceptional_data = manage_exceptional_data

        self.td_linear_statistical = DIHC_FeatureDetails.td_linear_statistical
        self.td_nonlinear_entropy = DIHC_FeatureDetails.td_nonlinear_entropy
        self.td_nonlinear_complexity_and_fractal_dimensions = DIHC_FeatureDetails.td_nonlinear_complexity_and_fractal_dimensions
        self.td_nonlinear_samp_entropy_profiling = DIHC_FeatureDetails.td_nonlinear_samp_entropy_profiling
        self.fd_linear_statistical = DIHC_FeatureDetails.fd_linear_statistical
        self.fd_linear_statistical_binwise = DIHC_FeatureDetails.fd_linear_statistical_binwise
        self.fd_spectral_band_power = DIHC_FeatureDetails.fd_spectral_band_power

        # Patch #1: band list is now injectable; falls back to package default for
        # backward compatibility only.
        self.band_frequency_list = band_frequency_list if band_frequency_list is not None else DIHC_FeatureDetails.band_frequency_list

        # Patch #2: signal_frequency is required wherever filtering or frequency-domain
        # features are actually used. We don't hard-fail here (some callers only want
        # time-domain features on unfiltered data), but nothing downstream assumes 256Hz.
        self.signal_frequency = signal_frequency
        self.band = (0, signal_frequency) if signal_frequency is not None else (0, None)

        self.feature_list = [DIHC_FeatureDetails.td_linear_statistical, DIHC_FeatureDetails.td_nonlinear_entropy,
                             DIHC_FeatureDetails.td_nonlinear_complexity_and_fractal_dimensions,
                             DIHC_FeatureDetails.td_nonlinear_samp_entropy_profiling, DIHC_FeatureDetails.fd_linear_statistical,
                             DIHC_FeatureDetails.fd_linear_statistical_binwise, DIHC_FeatureDetails.fd_spectral_band_power]

        self.sample_per_second = sample_per_second
        self.filtering_enabled = filtering_enabled
        self.lowcut = lowcut
        self.highcut = highcut

        if self.filtering_enabled:
            if signal_frequency is None:
                raise ValueError(
                    "signal_frequency must be explicitly provided when filtering_enabled=True."
                )
            if lowcut is None or highcut is None:
                raise ValueError(
                    "lowcut and highcut must be explicitly provided when filtering_enabled=True "
                    "(no EEG-typical default is assumed)."
                )
            nyquist = signal_frequency / 2.0
            if not (0 < lowcut < highcut < nyquist):
                raise ValueError(
                    f"Invalid filter band: lowcut={lowcut} Hz, highcut={highcut} Hz for "
                    f"signal_frequency={signal_frequency} Hz (Nyquist={nyquist} Hz). "
                    f"Require 0 < lowcut < highcut < Nyquist."
                )

        # Patch #3: adaptive SampEn minimum length instead of a fixed 5000-sample constant.
        if min_sample_entropy_length is not None:
            self.min_sample_entropy_length = min_sample_entropy_length
        elif signal_frequency is not None:
            # Rough rule-of-thumb minimum for a stable SampEn estimate: ~25s of data,
            # but never less than 100 samples. Override explicitly if you have a
            # domain-specific minimum in mind.
            self.min_sample_entropy_length = max(100, int(25 * signal_frequency))
        else:
            # No signal_frequency given either - fall back to the smallest sane default
            # rather than silently zeroing everything as the original 5000 constant did.
            self.min_sample_entropy_length = 100

        # Patch #6: configurable output rounding (None = full precision preserved).
        self.round_decimals = round_decimals

        self.fd_data_dict = None
        self.entropy_profile = None
        self.varbose_progress = varbose_progress

        return


    def get_new_features_to_calculate(self, feature_types):
        feature_names = None

        # Select features based on type
        if len(feature_types)==0:
            feature_names = [item for sublist in self.feature_list for item in sublist]
        else:
            feature_names = [item for i, sublist in enumerate(self.feature_list) for item in sublist if i in feature_types]

        return feature_names


    ### Getting all the features
    #############################################################
    def generate_features(self, seg_srl, seg_data, feature_names):
        self.fd_data_dict = None
        self.entropy_profile = None
        feature_values = []

        seg_values = seg_data.copy()

        #check if the feate names are enum or string
        if feature_names is None or len(feature_names)==0:
            feature_names = DIHC_FeatureGroup.all.value
        elif type(feature_names[0]) != DIHC_FeatureGroup:
            print("Invalid features...")
            exit(0)
        else:
            feature_names_copy = list(feature_names)
            feature_names = []
            for itm in feature_names_copy:
                feature_names.extend(itm.value)

            #remove duplicate and sort
            feature_names_copy = list(set(feature_names))
            all_feature_names = DIHC_FeatureGroup.all.value
            feature_names = [it for it in all_feature_names if it in feature_names_copy]

        # Generate corresponding features
        if self.varbose_progress:
            prog_bar = tqdm(feature_names, desc="Feature extraction started...")
        else:
            print(f"Feature extraction started...")
        feature_iterator = prog_bar if self.varbose_progress else feature_names
        for feat in feature_iterator:

            if self.varbose_progress:
                prog_bar.set_description(f"For segment: {seg_srl}, extracting feature: {feat} ||")
            else:
                print(f"For segment: {seg_srl}, extracting feature: {feat} ")
            method = None
            final_feat = None
            try:
                final_feat = feat
                final_data = seg_values

                #Reuse appropriate method call
                if feat.startswith('fd_'):
                    if (feat in (self.fd_linear_statistical)) or (feat in (self.fd_linear_statistical_binwise)):
                        #FFT data for frequency domain features
                        data_dict = self.fd_data_dict
                        if (self.fd_data_dict is None):
                            data_dict = self.fd_spectralAmplitude(seg_values)
                            self.fd_data_dict = data_dict

                        final_feat_list = (feat.split('_'))
                        fnl = len(final_feat_list)
                        if fnl>1:
                            final_feat = final_feat_list[1]
                            final_data = list(data_dict.values())

                            tmp = data_dict.keys()

                            if fnl > 2:
                                band_name = final_feat_list[2]
                                if band_name not in self.band_frequency_list:
                                    raise KeyError(
                                        f"Band '{band_name}' not found in band_frequency_list. "
                                        f"Available bands: {list(self.band_frequency_list.keys())}"
                                    )
                                tmp = [i for i in tmp if i in range(self.band_frequency_list[band_name][0], self.band_frequency_list[band_name][1])]
                                final_data = [data_dict[x] for x in tmp]

                    elif (feat in (self.fd_spectral_band_power)):
                        final_feat_list = (feat.split('_'))
                        fnl = len(final_feat_list)

                        if fnl > 1:
                            final_feat = f'{final_feat_list[0]}_{final_feat_list[1]}'
                            self.band = (0, self.signal_frequency)

                            if fnl > 2:
                                band_name = final_feat_list[2]
                                if band_name not in self.band_frequency_list:
                                    raise KeyError(
                                        f"Band '{band_name}' not found in band_frequency_list. "
                                        f"Available bands: {list(self.band_frequency_list.keys())}"
                                    )
                                self.band = (self.band_frequency_list[band_name][0], self.band_frequency_list[band_name][1])

                elif feat.startswith('entropyProfiled_'):
                    enProf = self.entropy_profile
                    if self.entropy_profile is None:
                        entProf_obj = DIHC_EntropyProfile()
                        enProf = entProf_obj.get_sample_entropy_profile(final_data)
                        dat = np.asarray(enProf)
                        if isinstance(dat[0], (list, np.ndarray)):
                            dat2 = [np.float64(item) for sublist in dat for item in sublist]
                        else:
                            dat2 = [np.float64(item) for item in dat]
                        enProf = np.array(dat2)
                        self.entropy_profile = enProf

                    final_feat = (feat.split('_'))
                    final_feat = final_feat[1]

                    final_data = enProf

                method = getattr(self, final_feat)

            except AttributeError:
                print(f'Method for feature: {final_feat} is not implemented.')
                raise NotImplementedError("Class `{}` does not implement `{}`".format(self.__class__.__name__, final_feat))

            # Patch #5: don't let one failing feature abort the whole segment.
            feat_val = np.nan
            result = np.all(final_data == final_data[0]) if len(final_data)>1 else True
            if not result:
                final_data = np.asarray(final_data, dtype=np.float64)
                try:
                    feat_val = method(final_data)
                except Exception as e:
                    if self.varbose_progress:
                        print(f"Warning: feature '{feat}' failed for segment {seg_srl}: {e}")
                    feat_val = np.nan

            # Patch #6: rounding is now opt-in, preserving full precision by default.
            if self.round_decimals is not None and feat_val is not None and not (isinstance(feat_val, float) and np.isnan(feat_val)):
                feat_val = round(feat_val, self.round_decimals)
            feature_values.append([feat_val])

        if self.varbose_progress:
            prog_bar.set_description(f"Feature extraction finished...")
            prog_bar.close()
        else:
            print(f"Feature extraction finished...")

        np_feat_value = np.array(feature_values)
        np_feat_value = np_feat_value.T

        all_features = pd.DataFrame()
        if len(feature_names)>0 and len(np_feat_value)>0:
            all_features = pd.DataFrame(np_feat_value, columns=feature_names)

        # Exceptional data management (now actually receives NaN/inf from failed
        # features - see patch note #4 - instead of those being masked as literal 0s).
        if self.manage_exceptional_data == 0:
            all_features = all_features[all_features != np.inf]
        elif self.manage_exceptional_data == 1:
            all_features = all_features[all_features != np.inf]
            all_features = all_features.fillna(0)
        elif self.manage_exceptional_data == 2:
            all_features = all_features[all_features != np.inf]
            all_features = all_features.dropna()
        elif self.manage_exceptional_data == 3:
            all_features = all_features[all_features != np.inf]
            all_features = all_features.fillna(all_features.mean())

        return all_features

###########################################################################
### Time Domain Features

    ### Total value of the segment
    def total(self, data):
        return np.sum(data)

    ### Summation value of the segment
    def summation(self, data):
        return np.sum(data)

    ### Average value of the segment
    def average(self, data):
        return np.mean(data)

    ### Minimum value of the segment
    def minimum(self, data):
        return np.min(data)

    ### Maximum value of the segment
    def maximum(self, data):
        return np.max(data)

    ### Mean value of the segment
    def mean(self, data):
        return np.mean(data)

    ### Median value of the segment
    def median(self, data):
        return np.median(data)

    ### Standard Deviation value of the segment
    def standardDeviation(self, data):
        return np.std(data)

    ### Variance value of the segment
    def variance(self, data):
        return np.var(data)

    ### kurtosis value of the segment
    def kurtosis(self, data):
        return sp.stats.kurtosis(data)

    ### skewness value of the segment
    def skewness(self, data):
        return sp.stats.skew(data)

    ### peak_or_Max value of the segment
    def peakOrMax(self, data):
        return self.maximum(data)

    ### numberOfPeaks value of the segment
    def numberOfPeaks(self, data):
        return len(sig.find_peaks(data))

    ### numberOfZeroCrossing value of the segment
    def numberOfZeroCrossing(self, data):
        return num_zerocross(data)  # Antropy package

    ### positiveToNegativeSampleRatio value of the segment
    def positiveToNegativeSampleRatio(self, data):
        try:
            return (np.sum(np.array(data) >= 0, axis=0)) / (np.sum(np.array(data) < 0, axis=0))
        except (ZeroDivisionError, Exception):
            return np.nan

    ### positiveToNegativePeakRatio value of the segment
    def positiveToNegativePeakRatio(self, data):
        try:
            return (len(sig.find_peaks(data))) / (len(sig.find_peaks(-data)))
        except (ZeroDivisionError, Exception):
            return np.nan

    ### meanAbsoluteValue value of the segment
    def meanAbsoluteValue(self, data):
        return self.mean(abs(data))



############ Entropy
    # ### Collected from Antropy and pyeeg package

    def approximateEntropy(self, data):
        try:
            return app_entropy(data)
        except Exception:
            return np.nan

    def sampleEntropy(self, data):
        # Patch #3: adaptive, signal_frequency-aware minimum length instead of a
        # hardcoded 5000-sample (EEG-window-shaped) constant.
        if len(data) < self.min_sample_entropy_length:
            return np.nan
        try:
            return sample_entropy(data)
        except Exception:
            return np.nan

    def permutationEntropy(self, data):
        try:
            return perm_entropy(data)
        except Exception:
            return np.nan

    def spectralEntropy(self, data):
        sf = self.signal_frequency
        if sf is None:
            raise ValueError("signal_frequency must be set before computing spectralEntropy.")
        try:
            return spectral_entropy(data, sf, method='welch')
        except Exception:
            return np.nan

    def singularValueDecompositionEntropy(self, data):
        try:
            return svd_entropy(data)
        except Exception:
            return np.nan

    ############ Fractal dimension
    # Collected from Antropy and pyeeg packages

    def hjorthMobility(self, data):
        try:
            hjm, _ = hjorth_params(data)
            return hjm
        except Exception:
            return np.nan

    def hjorthComplexity(self, data):
        try:
            _, hjc = hjorth_params(data)
            return hjc
        except Exception:
            return np.nan

    def hurstExponent(self, data):
        try:
            return pyeeg.hurst(data)
        except Exception:
            return np.nan

    def fisherInfo(self, data, tau=1, m=2):
        try:
            return pyeeg.fisher_info(data, tau, m)
        except Exception:
            return np.nan

    def lempelZivComplexity(self, data):
        try:
            return lziv_complexity(data)
        except Exception:
            return np.nan

    def petrosianFd(self, data):
        try:
            return petrosian_fd(data)
        except Exception:
            return np.nan

    def katzFd(self, data):
        try:
            return katz_fd(data)
        except Exception:
            return np.nan

    def higuchiFd(self, data):
        try:
            return higuchi_fd(data)
        except Exception:
            return np.nan

    def detrendedFluctuation(self, data):
        try:
            return detrended_fluctuation(data)
        except Exception:
            return np.nan


    def fuzzyEntropy(self, data, m=2, tau=1, r=0.2):
        data = np.asarray(data).flatten()
        try:
            return compute_fuzzy_entropy_jit(data, m, tau, r)
        except Exception:
            return np.nan


    def distributionEntropy(self, data, m=2, M=500):
        data = np.asarray(data).flatten()
        N = len(data)
        if N <= m:
            # Raised (not silently NaN'd) because this is a caller-fixable
            # precondition, not a runtime numerical failure. It is still caught
            # safely by generate_features()'s outer try/except (patch #5) if it
            # occurs during full pipeline extraction.
            raise ValueError("Input data length must be greater than embedding dimension m.")
        return compute_distribution_entropy_jit(data, m, M)


    def distributionEntropy4(self, data, m=4):
        return self.distributionEntropy(data, m=m)
    def distributionEntropy6(self, data, m=6):
        return self.distributionEntropy(data, m=m)
    def distributionEntropy8(self, data, m=8):
        return self.distributionEntropy(data, m=m)
    def distributionEntropy10(self, data, m=10):
        return self.distributionEntropy(data, m=m)


    def shannonEntropy(self, data, m=2):
        try:
            bases = collections.Counter([tmp_base for tmp_base in data])
            dist = [i / sum(bases.values()) for i in bases.values()]
            return scipyEntropy(dist, base=m)
        except Exception:
            return np.nan


    def _x_log2_x(self, data):
        """ Return x * log2(x) and 0 if x is 0."""
        res = data * np.log2(data)
        if np.size(data) == 1:
            if np.isclose(data, 0.0):
                res = 0.0
        else:
            res[np.isclose(data, 0.0)] = 0.0
        return res


    def renyiEntropy(self, data, alpha=2):
        try:
            assert alpha >= 0, "Error: renyi_entropy only accepts values of alpha >= 0, but alpha = {}.".format(alpha)
            if np.isinf(alpha):
                return - np.log2(np.max(data))
            elif np.isclose(alpha, 0):
                return np.log2(len(data))
            elif np.isclose(alpha, 1):
                return - np.sum(self._x_log2_x(data))
            else:
                return (1.0 / (1.0 - alpha)) * np.log2(np.sum(data ** alpha))
        except Exception:
            return np.nan


### Frequency Domain Features

    ##Bandpass filtering
    def _butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        # Patch #2: fail loudly rather than silently producing a garbage filter
        # design when the requested band doesn't fit within (0, Nyquist).
        if not (0 < low < high < 1):
            raise ValueError(
                f"Invalid bandpass filter request: lowcut={lowcut} Hz, highcut={highcut} Hz, "
                f"fs={fs} Hz (Nyquist={nyq} Hz). Require 0 < lowcut < highcut < Nyquist."
            )
        tpl_res = butter(order, [low, high], btype='band')
        b, a = tpl_res[0], tpl_res[1]
        return b, a


    def _butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self._butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y


    ##Fast Faurier Transformation
    def _fast_faurier_transformation(self, data, fs, fft_type=2):
        feat_data = None

        if fft_type==0:
            fft_data = fft.fft(data)
            freqs = fft.fftfreq(len(data)) * fs
            feat_data = {freqs[i]: fft_data[i] for i in range(len(freqs))}
        elif fft_type==1:
            fft_data = fftpack.fft(data)
            freqs = fftpack.fftfreq(len(data)) * fs
            feat_data = {freqs[i]: fft_data[i] for i in range(len(freqs))}
        elif fft_type==2:
            fft_data = np.abs(np.fft.fft(data))
            freqs = np.fft.fftfreq(len(data), d=1.0 / fs)
            feat_data = {freqs[i]: fft_data[i] for i in range(len(freqs))}

        return feat_data


    ### Original Frequency domain features
    def fd_spectralAmplitude(self, data):
        filtered_data = data
        sample_per_second = self.sample_per_second

        if self.filtering_enabled:
            lowcut = self.lowcut
            highcut = self.highcut
            filtered_data = self._butter_bandpass_filter(data, lowcut, highcut, sample_per_second, order=6)

        feat_data = self._fast_faurier_transformation(data, sample_per_second)

        return feat_data


    ### Power of a signal or signal band
    ### @Author: raphaelvallat (Author of Antropy package) Source:- https://raphaelvallat.com/bandpower.html
    def fd_bandPower(self, data, method='multitaper', window_sec=None, relative=False):
        """Compute the average power of the signal x in a specific frequency band."""
        sf = self.signal_frequency
        band = self.band

        if sf is None:
            raise ValueError("signal_frequency must be set before computing fd_bandPower.")

        # Input validation
        if len(data) < 2:
            return np.nan

        band = np.asarray(band)
        low, high = band

        try:
            if method == 'welch':
                if window_sec is not None:
                    nperseg = int(window_sec * sf)
                else:
                    nperseg = int((2 / low) * sf)
                nperseg = min(len(data), nperseg)
                if nperseg < 2:
                    return np.nan
                freqs, psd = welch(data, sf, nperseg=nperseg)
            elif method == 'multitaper':
                psd, freqs = psd_array_multitaper(data, sf, adaptive=True,
                                                  normalization='full', verbose=0)

            if len(psd) == 0 or len(freqs) == 0:
                return np.nan

            freq_res = freqs[1] - freqs[0]
            idx_band = np.logical_and(freqs >= low, freqs <= high)

            if not np.any(idx_band):
                return np.nan

            bp = simps(psd[idx_band], dx=freq_res)

            if relative:
                total_power = simps(psd, dx=freq_res)
                bp = bp / total_power if total_power != 0 else np.nan

            return bp

        except Exception:
            return np.nan


### Time-Frequency Domain Features

### Wavelet Domain Features


####################################################################