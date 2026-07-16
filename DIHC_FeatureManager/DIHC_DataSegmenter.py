# -*- coding: utf-8 -*-
"""
File Name: DIHC_DataSegmenter.py
Updated for cleaner segmentation steps, proper validation errors, and typo corrections.
"""

import sys
from typing import Generator, Optional, Tuple
import numpy as np

try:
    if "ipykernel" in sys.modules:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs): return iterable

class DIHC_DataSegmenter:

    def __init__(
        self, 
        data: np.ndarray, 
        segment_length: Optional[float] = None, 
        segment_overlap: float = 0.0, 
        signal_frequency: float = 256.0, 
        verbose_progress: bool = False,
        **kwargs
    ):
        self.data = np.asarray(data)
        self.segment_length = segment_length
        self.segment_overlap = segment_overlap
        self.signal_frequency = signal_frequency
        self.verbose_progress = kwargs.get("varbose_progress", verbose_progress)
        
        self.seg_len, self.seg_mov, self.num_segs = self.get_segment_metadata()

    def get_segment_metadata(self) -> Tuple[int, int, int]:
        n_samples = len(self.data)
        
        # Default fallback to entire window structure if limits are not set
        if self.segment_length is None or self.signal_frequency is None:
            return n_samples, n_samples, 1

        seg_len = int(self.segment_length * self.signal_frequency)
        seg_mov = int(seg_len - (self.segment_overlap * self.signal_frequency))
        
        if seg_len <= 0 or seg_mov <= 0:
            return n_samples, n_samples, 1

        num_segs = max(0, int((n_samples - seg_len) / seg_mov + 1))
        return seg_len, seg_mov, num_segs

    def generate_segments(self) -> Generator[np.ndarray, None, None]:
        if len(self.data) == 0:
            raise ValueError("Data vector is completely empty. Cannot compute segments.")

        if self.segment_length is None:
            yield self.data
            return

        if self.seg_len > len(self.data):
            yield self.data
            return

        seg_srl = 1
        seg_st = 0
        
        prog_bar = None
        if self.verbose_progress:
            prog_bar = tqdm(total=self.num_segs, desc='Segmentation starting...', file=sys.stdout)

        while seg_st < len(self.data):
            if prog_bar:
                prog_bar.set_description(f'Generating segment# {seg_srl} ||')
            
            seg_end = min(seg_st + self.seg_len, len(self.data))
            seg_data = self.data[int(seg_st):int(seg_end)]
            
            # Prevent empty chunks near the tail end of calculation
            if len(seg_data) == 0:
                break
                
            yield seg_data
            
            seg_st += self.seg_mov
            seg_srl += 1
            if prog_bar:
                prog_bar.update(1.0)

        if prog_bar:
            prog_bar.set_description('Segmentation completed.')
            prog_bar.close()