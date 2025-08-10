"""
File Name: HumachLab_FeatureDetails.py
Author: WWM Emran (Emran Ali)
Involvement: HumachLab & Deakin- Innovation in Healthcare (DIHC)
Email: wwm.emran@gmail.com, emran.ali@research.deakin.edu.au
Date: 3/09/2021 7:38 pm
"""
import  sys
import time
import pandas as pd

try:
    if "ipykernel" in sys.modules:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except ImportError:
    from tqdm import tqdm

### SRART: My modules ###
from DIHC_FeatureManager import *
from DIHC_FeatureManager.DIHC_FeatureExtractor import *
### END: My modules ###


class DIHC_DataSegmenter:

    # ## Initialization
    def __init__(self, data, segment_length=None, segment_overlap=0, signal_frequency=256, prog_bar=None, varbose_progress=False):
        self.data = data
        self.segment_length = segment_length
        self.segment_overlap = segment_overlap
        self.signal_frequency = signal_frequency

        # self.prog_bar = prog_bar
        self.varbose_progress = varbose_progress
        self.seg_len, self.seg_mov, self.num_segs = self.get_segment_metadata()
        # print(f'---->> {seg_srl}, {seg_st}, {seg_len}, {seg_mov}')

        # if self.varbose_progress:
        #     self.prog_bar = tqdm(total=self.num_segs, desc=f'Segmentation started...', position=0, file=sys.stdout)
        #     # self.prog_bar = tqdm(range(num_segs), desc=f'Segmentation started...')
        # else:
        #     print(f'Segmentation started...')
        return

    def get_segment_metadata(self):
        hyp_seg_multiplier = 1
        seg_len = int(self.segment_length*hyp_seg_multiplier) if self.signal_frequency==None else int(self.segment_length*self.signal_frequency)
        seg_mov = int(seg_len-self.segment_overlap*hyp_seg_multiplier) if self.signal_frequency==None else int(seg_len-(self.segment_overlap*self.signal_frequency))
        num_segs = max(0, int((len(self.data)-seg_len)/seg_mov +1) )
        return seg_len, seg_mov, num_segs

    # ## Data Segmentor
    def generate_segments(self):
        if len(self.data)==0:
            print(f'Data is empty...')
            exit(0)
            # return

        if self.segment_length is None:
            print(f'Dealing with entire signal...')
        else:
            if self.seg_len > len(self.data):
                print(f'Data can\'t be segmented...')
                return self.data
            else:
                seg_srl = 1
                seg_st = 0
                # seg_len, seg_mov, num_segs = self.get_segment_metadata()
                seg_len = self.seg_len
                seg_mov = self.seg_mov
                num_segs = self.num_segs
                # print(f'---->> {seg_srl}, {seg_st}, {seg_len}, {seg_mov}')

                if self.varbose_progress:
                    self.prog_bar = tqdm(total=self.num_segs, desc=f'Segmentation started...', position=0, file=sys.stdout)
                    # self.prog_bar = tqdm(range(num_segs), desc=f'Segmentation started...')
                else:
                    print(f'Segmentation started...')

                # self.prog_bar = tqdm(total=num_segs, desc=f'Segmentation started...')
                while (seg_st<len(self.data)):
                    if self.varbose_progress:
                        self.prog_bar.set_description(f'Generating segment# {seg_srl} ||')
                    else:
                        print(f'Generating segment# {seg_srl} ')
                    seg_end = seg_st+seg_len
                    if seg_end>len(self.data):
                        seg_end = len(self.data) 
                        # break 
                    # print(f'====> {type(seg_st)} = {type(seg_end)}')
                    seg_data = self.data[int(seg_st):int(seg_end)]
                    seg_st += seg_mov
                    seg_srl += 1
                    if self.varbose_progress:
                        self.prog_bar.update(1.0)
                    yield seg_data

                if self.varbose_progress:
                    self.prog_bar.set_description(f'Segmentation finished...')
                    self.prog_bar.close()
                else:
                    print(f'Segmentation finished...')
        return


