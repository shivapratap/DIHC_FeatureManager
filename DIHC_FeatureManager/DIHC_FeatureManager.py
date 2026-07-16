"""Feature-manager orchestration for DIHC_FeatureManager.

This revision keeps the original public API while fixing import, validation,
segmentation, mutable-default, and error-handling problems.
"""
from __future__ import annotations

import sys
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

try:
    if "ipykernel" in sys.modules:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else _NullProgress()


class _NullProgress:
    def update(self, *args, **kwargs):
        pass

    def set_description(self, *args, **kwargs):
        pass

    def close(self):
        pass


# Use explicit relative imports. The original self-import
# ``from DIHC_FeatureManager import *`` caused circular-import behaviour.
from .DIHC_EntropyProfile import DIHC_EntropyProfile
from .DIHC_FeatureExtractor import DIHC_FeatureExtractor
from .DIHC_DataSegmenter import DIHC_DataSegmenter


class DIHC_FeatureManager:
    """Coordinate segmentation, feature extraction and entropy profiling."""

    def __init__(self, varbose_progress: bool = False, verbose_progress: Optional[bool] = None):
        # Preserve the original misspelled argument for backward compatibility.
        if verbose_progress is not None:
            varbose_progress = verbose_progress
        self.data_segmenter = None
        self.feat_extractor = None
        self.feat_selector = None
        self.entProf_extractor = None
        self.prog_bar = None
        self.varbose_progress = bool(varbose_progress)
        self.data = None
        self.segment_length = None
        self.segment_overlap = 0
        self.signal_frequency = None

    @staticmethod
    def _as_1d_numeric(data) -> np.ndarray:
        arr = np.asarray(data)
        if arr.ndim != 1:
            arr = np.ravel(arr)
        if arr.size == 0:
            raise ValueError("data must contain at least one sample")
        if not np.issubdtype(arr.dtype, np.number):
            try:
                arr = arr.astype(float)
            except (TypeError, ValueError) as exc:
                raise TypeError("data must be numeric") from exc
        if not np.all(np.isfinite(arr)):
            raise ValueError("data contains NaN or infinite values")
        return arr

    @staticmethod
    def _validate_segmentation(segment_length, segment_overlap, signal_frequency):
        if signal_frequency is None:
            raise ValueError("signal_frequency must be explicitly provided")
        if signal_frequency <= 0:
            raise ValueError("signal_frequency must be > 0")
        if segment_length is not None and segment_length <= 0:
            raise ValueError("segment_length must be > 0")
        if segment_overlap is None:
            segment_overlap = 0
        if segment_overlap < 0:
            raise ValueError("segment_overlap must be >= 0")
        if segment_length is not None and segment_overlap >= segment_length:
            raise ValueError("segment_overlap must be smaller than segment_length")
        return float(signal_frequency), segment_overlap

    def get_segment_metadata(self):
        if self.data is None:
            raise RuntimeError("No data have been assigned")

        n_samples = len(self.data)
        if self.segment_length is None:
            return n_samples, n_samples, 1

        seg_len = int(round(float(self.segment_length) * float(self.signal_frequency)))
        overlap_len = int(round(float(self.segment_overlap or 0) * float(self.signal_frequency)))
        seg_mov = seg_len - overlap_len

        if seg_len <= 0:
            raise ValueError("segment_length produces a zero-length segment")
        if seg_mov <= 0:
            raise ValueError("segment_overlap leaves a non-positive step size")
        if n_samples < seg_len:
            num_segs = 0
        else:
            num_segs = 1 + (n_samples - seg_len) // seg_mov
        return seg_len, seg_mov, int(num_segs)

    def _new_segmenter(self, data):
        return DIHC_DataSegmenter(
            data,
            segment_length=self.segment_length,
            segment_overlap=self.segment_overlap,
            signal_frequency=self.signal_frequency,
            varbose_progress=self.varbose_progress,
        )

    def get_segments_for_data(self, data, segment_length=None, segment_overlap=0, signal_frequency=None):
        data = self._as_1d_numeric(data)
        signal_frequency, segment_overlap = self._validate_segmentation(
            segment_length, segment_overlap, signal_frequency
        )
        self.data = data
        self.segment_length = segment_length
        self.segment_overlap = segment_overlap
        self.signal_frequency = signal_frequency

        _, _, num_segs = self.get_segment_metadata()
        if segment_length is None:
            return data.reshape(1, -1)
        if num_segs == 0:
            return np.empty((0, int(round(segment_length * signal_frequency))))

        self.data_segmenter = self._new_segmenter(data)
        segments = list(self.data_segmenter.generate_segments())
        if not segments:
            return np.empty((0, int(round(segment_length * signal_frequency))))
        return np.vstack([np.asarray(seg).reshape(1, -1) for seg in segments])

    def extract_features_from_data(
        self,
        data,
        feature_names=None,
        segment_length=None,
        segment_overlap=0,
        signal_frequency=None,
        filtering_enabled=False,
        lowcut=None,
        highcut=None,
        manage_exceptional_data=0,
        band_frequency_list=None,
        min_sample_entropy_length=None,
        round_decimals=None,
    ):
        data = self._as_1d_numeric(data)
        signal_frequency, segment_overlap = self._validate_segmentation(
            segment_length, segment_overlap, signal_frequency
        )
        self.data = data
        self.segment_length = segment_length
        self.segment_overlap = segment_overlap
        self.signal_frequency = signal_frequency

        seg_len, _, num_segs = self.get_segment_metadata()
        sample_per_second = len(data) if segment_length is None else seg_len
        names = [] if feature_names is None else feature_names

        self.feat_extractor = DIHC_FeatureExtractor(
            manage_exceptional_data=manage_exceptional_data,
            signal_frequency=signal_frequency,
            sample_per_second=sample_per_second,
            filtering_enabled=filtering_enabled,
            lowcut=lowcut,
            highcut=highcut,
            varbose_progress=self.varbose_progress,
            band_frequency_list=band_frequency_list,
            min_sample_entropy_length=min_sample_entropy_length,
            round_decimals=round_decimals,
        )

        if segment_length is None or num_segs == 0:
            return self.feat_extractor.generate_features(1, data, names).reset_index(drop=True)

        self.data_segmenter = self._new_segmenter(data)
        rows = []
        bar = tqdm(total=num_segs, desc="Feature extraction", file=sys.stdout) if self.varbose_progress else None
        try:
            for seg_no, seg_data in enumerate(self.data_segmenter.generate_segments(), start=1):
                rows.append(self.feat_extractor.generate_features(seg_no, seg_data, names))
                if bar is not None:
                    bar.update(1)
        finally:
            if bar is not None:
                bar.close()
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    @staticmethod
    def validate_segment_data_for_signal_frequency(data, signal_frequency):
        if signal_frequency is None:
            return False
        try:
            arr = np.asarray(data)
        except Exception:
            return False
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
            return False
        return np.issubdtype(arr.dtype, np.number) and np.isfinite(arr).all()

    def extract_features_from_segments(
        self,
        data,
        feature_names=None,
        signal_frequency=None,
        filtering_enabled=False,
        lowcut=None,
        highcut=None,
        manage_exceptional_data=0,
        band_frequency_list=None,
        min_sample_entropy_length=None,
        round_decimals=None,
    ):
        if signal_frequency is None or signal_frequency <= 0:
            raise ValueError("signal_frequency must be explicitly provided and > 0")
        arr = np.asarray(data)
        if not self.validate_segment_data_for_signal_frequency(arr, signal_frequency):
            raise ValueError("data must be a finite, rectangular 2-D numeric segment array")

        self.data = arr
        self.segment_length = arr.shape[1] / float(signal_frequency)
        self.segment_overlap = 0
        self.signal_frequency = float(signal_frequency)
        names = [] if feature_names is None else feature_names

        self.feat_extractor = DIHC_FeatureExtractor(
            manage_exceptional_data=manage_exceptional_data,
            signal_frequency=self.signal_frequency,
            sample_per_second=arr.shape[1],
            filtering_enabled=filtering_enabled,
            lowcut=lowcut,
            highcut=highcut,
            varbose_progress=self.varbose_progress,
            band_frequency_list=band_frequency_list,
            min_sample_entropy_length=min_sample_entropy_length,
            round_decimals=round_decimals,
        )

        rows = []
        bar = tqdm(total=len(arr), desc="Feature extraction", file=sys.stdout) if self.varbose_progress else None
        try:
            for seg_no, seg_data in enumerate(arr, start=1):
                rows.append(self.feat_extractor.generate_features(seg_no, seg_data, names))
                if bar is not None:
                    bar.update(1)
        finally:
            if bar is not None:
                bar.close()
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def extract_sampEn_profile_from_data(
        self, data, segment_length=None, segment_overlap=0, signal_frequency=None
    ):
        data = self._as_1d_numeric(data)
        signal_frequency, segment_overlap = self._validate_segmentation(
            segment_length, segment_overlap, signal_frequency
        )
        self.data = data
        self.segment_length = segment_length
        self.segment_overlap = segment_overlap
        self.signal_frequency = signal_frequency
        _, _, num_segs = self.get_segment_metadata()

        self.entProf_extractor = DIHC_EntropyProfile(varbose_progress=self.varbose_progress)
        if segment_length is None or num_segs == 0:
            return self.entProf_extractor.generate_sampEn_profile(data).reset_index(drop=True)

        self.data_segmenter = self._new_segmenter(data)
        profiles = []
        bar = tqdm(total=num_segs, desc="Entropy profile", file=sys.stdout) if self.varbose_progress else None
        try:
            for seg_no, seg_data in enumerate(self.data_segmenter.generate_segments(), start=1):
                frame = self.entProf_extractor.generate_sampEn_profile(seg_data).reset_index(drop=True)
                frame.insert(0, "Segment_No", seg_no)
                profiles.append(frame)
                if bar is not None:
                    bar.update(1)
        finally:
            if bar is not None:
                bar.close()
        return pd.concat(profiles, ignore_index=True) if profiles else pd.DataFrame()
