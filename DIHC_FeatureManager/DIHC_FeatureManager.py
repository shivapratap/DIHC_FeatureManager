"""
File Name: DIHC_FeatureManager.py
Author: WWM Emran (Emran Ali)
Involvement: HumachLab & Deakin- Innovation in Healthcare (DIHC)
Email: wwm.emran@gmail.com, emran.ali@research.deakin.edu.au 
Date: 3/09/2021 7:38 pm
"""
import  sys
import numpy as np
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
from DIHC_FeatureManager.DIHC_EntropyProfile import *
from DIHC_FeatureManager.DIHC_FeatureExtractor import *
from DIHC_FeatureManager.DIHC_DataSegmenter import *
### END: My modules ###


class DIHC_FeatureManager:

    # ## Initialization
    def __init__(self):
        self.data_segmenter = None
        self.feat_extractor = None
        self.feat_selector = None
        self.prog_bar = None
        # self.prog_bar = tqdm(total=100, desc=f'Feature manager started...')
        return

    # ## Segment generater
    def get_segments_for_data(self, data, segment_length=None, segment_overlap=0, signal_frequency=256):
        if len(data)==0:
            print(f'Data is empty...')
            exit(0)
            # return

        # print('Data segmentation started...')
        seg_len = int(segment_length * signal_frequency)
        seg_mov = int(seg_len - (segment_overlap * signal_frequency))
        num_segs = max(0, int((len(data) - seg_len) / seg_mov + 1))
        self.prog_bar = tqdm(total=num_segs, desc=f'Data segmentation started...', position=0, file=sys.stdout)
        self.data_segmenter = DIHC_DataSegmenter(data, segment_length=segment_length, segment_overlap=segment_overlap,
                                                 signal_frequency=signal_frequency)
        seg_generator = self.data_segmenter.generate_segments()
        seg_srl = 0
        all_seg_data = np.array([])
        while True:
            try:
                self.prog_bar.set_description(f'Generating segments #{seg_srl+1} ||')
                seg_data = next(seg_generator)
                if seg_srl==0:
                    all_seg_data = np.array(seg_data)
                else:
                    all_seg_data = np.vstack([all_seg_data, seg_data])
                seg_srl += 1
                self.prog_bar.update(1.0)
            except StopIteration:
                # print('Finished segmentation of data...')
                # self.prog_bar.set_description(f'Finished segmentation of data...')
                break
        # print('Finished segmentation of data...')
        self.prog_bar.set_description(f'Finished segmentation of data...')
        self.prog_bar.close()
        return all_seg_data


    # ## Feature extractor - from data
    def extract_features_from_data(self, data, feature_names=[], segment_length=None, segment_overlap=0, signal_frequency=256,
                               filtering_enabled=False, lowcut=1, highcut=48, manage_exceptional_data=0):
        if len(data)==0:
            print(f'Data is empty...')
            exit(0)
            # return

        sampPS = len(data) if segment_length is None else (segment_length*signal_frequency)

        self.feat_extractor = DIHC_FeatureExtractor(manage_exceptional_data=manage_exceptional_data, signal_frequency=signal_frequency,
                                                    sample_per_second=sampPS, filtering_enabled=filtering_enabled, lowcut=lowcut, highcut=highcut)
        all_feat_df = pd.DataFrame()

        if segment_length is None:
            print(f'Dealing with entire signal...')
            all_feat_df= self.feat_extractor.generate_features(1, data, feature_names)
        else:
            if (segment_length*signal_frequency) > len(data):
                print(f'Data can\'t be segmented...')
                all_feat_df = self.feat_extractor.generate_features(1, data, feature_names)
            else:
                print(f'Data started segmenting for features: {feature_names}')
                seg_len = int(segment_length*signal_frequency)
                seg_mov = int(seg_len-(segment_overlap*signal_frequency))
                num_segs = max(0, int((len(data)-seg_len)/seg_mov +1) )
                self.prog_bar = tqdm(total=num_segs, desc=f'Feature extraction started...', position=0, file=sys.stdout)
                self.data_segmenter = DIHC_DataSegmenter(data, segment_length=segment_length, segment_overlap=segment_overlap, signal_frequency=signal_frequency)
                seg_generator = self.data_segmenter.generate_segments()
                seg_srl = 0
                while True:
                    try:
                        self.prog_bar.set_description(f'Extracting features for segment# {seg_srl+1} ||')
                        seg_data = next(seg_generator)
                        # print('SEG data', seg_data)
                        feat_df = self.feat_extractor.generate_features(seg_srl+1, seg_data, feature_names)
                        all_feat_df = pd.concat([all_feat_df, feat_df])
                        all_feat_df = all_feat_df.reset_index(drop=True)
                        self.prog_bar.update(1.0)
                    except StopIteration:
                        # print('Finished extracting features for all segments...')
                        # self.prog_bar.set_description(f'Finished extracting features for all segments...')
                        break
                    seg_srl+=1
                # print('Finished extracting features for all segments...')
                self.prog_bar.set_description(f'Finished extracting features for all segments...')
                self.prog_bar.close()
        return all_feat_df


    # ## Validate_segment_data_for_signal_frequency
    def validate_segment_data_for_signal_frequency(self, data, signal_frequency):
        first_shape = len(data[0:])
        for sg in data[:-1, :]:
            if len(sg) != first_shape:
                return False
        if first_shape<len(data[-1, :]):
            return False
        return True



    # ## Feature extractor - from segments
    def extract_features_from_segments(self, data, feature_names=[], signal_frequency=256,
                               filtering_enabled=False, lowcut=1, highcut=48, manage_exceptional_data=0):
        if len(data)==0:
            print(f'Data is empty...')
            exit(0)
            # return

        isValid = self.validate_segment_data_for_signal_frequency(data, signal_frequency)

        if not isValid:
            print('The segment is not valid...')
            exit(0)

        segment_length = data[0, :]/signal_frequency
        sampPS = len(data[0]) if segment_length is None else (segment_length*signal_frequency)

        self.feat_extractor = DIHC_FeatureExtractor(manage_exceptional_data=manage_exceptional_data, signal_frequency=signal_frequency,
                                                    sample_per_second=sampPS, filtering_enabled=filtering_enabled, lowcut=lowcut, highcut=highcut)
        all_feat_df = pd.DataFrame()

        self.prog_bar = tqdm(total=len(data), desc=f'Feature extraction started...', position=0, file=sys.stdout)

        for seg_srl, seg_data in enumerate(data):
            self.prog_bar.set_description(f'Extracting features for segment# {seg_srl+1} ||')
            feat_df = self.feat_extractor.generate_features(seg_srl+1, seg_data, feature_names)
            all_feat_df = pd.concat([all_feat_df, feat_df])
            all_feat_df = all_feat_df.reset_index(drop=True)
            self.prog_bar.update(1.0)

        self.prog_bar.set_description(f'Finished extracting features for all segments...')
        self.prog_bar.close()

        return all_feat_df


    # ## Entropy profile extractor - from data
    def extract_sampEn_profile_from_data(self, data, segment_length=None, segment_overlap=0, signal_frequency=256):
        if len(data)==0:
            print(f'Data is empty...')
            exit(0)
            # return

        # sampPS = len(data) if segment_length is None else (segment_length*signal_frequency)

        self.entProf_extractor = DIHC_EntropyProfile()
        all_entProf_df = pd.DataFrame()

        if segment_length is None:
            print(f'Dealing with entire signal...')
            all_entProf_df = self.entProf_extractor.generate_sampEn_profile(data)
        else:
            if (segment_length*signal_frequency) > len(data):
                print(f'Data can\'t be segmented...')
                all_entProf_df = self.entProf_extractor.generate_sampEn_profile(data)
            else:
                print(f'Entropy profile calculation started...')
                seg_len = int(segment_length*signal_frequency)
                seg_mov = int(seg_len-(segment_overlap*signal_frequency))
                num_segs = max(0, int((len(data)-seg_len)/seg_mov +1) )
                self.prog_bar = tqdm(total=num_segs, desc=f'Entropy profile calculation started...', position=0, file=sys.stdout)
                self.data_segmenter = DIHC_DataSegmenter(data, segment_length=segment_length, segment_overlap=segment_overlap, signal_frequency=signal_frequency)
                seg_generator = self.data_segmenter.generate_segments()
                seg_num = 0
                while True:
                    try:
                        self.prog_bar.set_description(f'Generating SampEn entropy profile for segment# {seg_num + 1} ||')
                        seg_data = next(seg_generator)
                        # print('SEG data', seg_data)
                        entProf_df = self.entProf_extractor.generate_sampEn_profile(seg_data)
                        # print('xxxxxx- ', entProf_df.shape[0], [seg_num]*entProf_df.shape[0])
                        seg_num_df = pd.DataFrame([seg_num]*entProf_df.shape[0], columns=['Segment_No'])
                        seg_num_df = seg_num_df.reset_index(drop=True)
                        entProf_df = pd.concat([seg_num_df, entProf_df], axis=1)
                        all_entProf_df = pd.concat([all_entProf_df, entProf_df])
                        all_entProf_df = all_entProf_df.reset_index(drop=True)
                        # print(type(entProf_df), entProf_df)
                        self.prog_bar.update(1.0)
                    except StopIteration:
                        # print('Finished generating Sample entropy profile for all segments...')
                        # self.prog_bar.set_description(f'Finished generating Sample entropy profile for all segments...')
                        break
                    seg_num += 1
                # print('Finished generating Sample entropy profile for all segments...')
                self.prog_bar.set_description(f'Finished generating entropy profile for all segments...')
                self.prog_bar.close()
        return all_entProf_df




