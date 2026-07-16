"""
File Name: DIHC_FeatureManager.py
Original Author: WWM Emran (Emran Ali)
Original Involvement: HumachLab & Deakin- Innovation in Healthcare (DIHC)
Original Email: wwm.emran@gmail.com, emran.ali@research.deakin.edu.au
Original Date: 3/09/2021 7:38 pm

--------------------------------------------------------------------------
GENERALIZATION PATCH NOTES (paired with DIHC_FeatureExtractor.py patch - see
that file's header for the full rationale):

1. `signal_frequency` no longer defaults to 256 Hz (EEG-typical). It now
   defaults to None and is validated explicitly - callers must state their
   real sampling rate, so a forgotten argument fails loudly instead of
   silently analyzing data at the wrong assumed rate.

2. `lowcut`/`highcut` no longer default to EEG-typical values (1/48 Hz).
   They default to None and are only required (and validated against
   Nyquist) when filtering_enabled=True.

3. New passthrough parameters: `band_frequency_list`, `min_sample_entropy_length`,
   `round_decimals` - forwarded to DIHC_FeatureExtractor so callers can
   configure these per signal domain instead of inheriting EEG-shaped
   defaults. See DIHC_FeatureExtractor.py for what each controls.
--------------------------------------------------------------------------
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
    def __init__(self, varbose_progress=False):
        self.data_segmenter = None
        self.feat_extractor = None
        self.feat_selector = None
        self.prog_bar = None
        self.varbose_progress = varbose_progress
        return

    def get_segment_metadata(self):
        hyp_seg_multiplier = 1

        seg_len = int(len(self.data)*hyp_seg_multiplier)
        seg_mov = int(len(self.data)*hyp_seg_multiplier)
        num_segs = 1

        if ((self.signal_frequency is not None) and (self.segment_length is not None)):
            seg_len = int(len(self.data)*hyp_seg_multiplier) if ((self.signal_frequency is None) or (self.segment_length is None)) else int(self.segment_length*self.signal_frequency)
        if ((self.signal_frequency is not None) and (self.segment_overlap is not None)):
            seg_mov = int(seg_len-self.segment_overlap*hyp_seg_multiplier) if self.signal_frequency==None else int(seg_len-(self.segment_overlap*self.signal_frequency))

        if seg_len > 0 and seg_mov > 0:
            num_segs = max(0, int((len(self.data)-seg_len)/seg_mov +1) )

        return seg_len, seg_mov, num_segs

    # ## Segment generater
    def get_segments_for_data(self, data, segment_length=None, segment_overlap=0, signal_frequency=None):
        if len(data)==0:
            print(f'Data is empty...')
            exit(0)

        if signal_frequency is None:
            raise ValueError(
                "signal_frequency must be explicitly provided - no default sampling rate is assumed."
            )

        self.data = data
        self.segment_length = segment_length
        self.segment_overlap = segment_overlap
        self.signal_frequency = signal_frequency

        seg_len, seg_mov, num_segs = self.get_segment_metadata()

        if self.varbose_progress:
            self.prog_bar = tqdm(total=num_segs, desc=f'Data segmentation started...', position=0, file=sys.stdout)
        else:
            print(f'Data segmentation started...')
        self.data_segmenter = DIHC_DataSegmenter(data, segment_length=segment_length, segment_overlap=segment_overlap,
                                                 signal_frequency=signal_frequency, varbose_progress=self.varbose_progress)
        seg_generator = self.data_segmenter.generate_segments()
        seg_srl = 0
        all_seg_data = np.array([])
        while True:
            try:
                if self.varbose_progress:
                    self.prog_bar.set_description(f'Generating segments #{seg_srl+1} ||')
                else:
                    print(f'Generating segments #{seg_srl+1} ')
                seg_data = next(seg_generator)
                if seg_srl==0:
                    all_seg_data = np.array(seg_data)
                else:
                    all_seg_data = np.vstack([all_seg_data, seg_data])
                seg_srl += 1
                if self.varbose_progress:
                    self.prog_bar.update(1.0)
            except StopIteration:
                break
        if self.varbose_progress:
            self.prog_bar.set_description(f'Finished segmentation of data...')
            self.prog_bar.close()
        else:
            print(f'Finished segmentation of data...')
        return all_seg_data


    # ## Feature extractor - from data
    def extract_features_from_data(self, data, feature_names=[], segment_length=None, segment_overlap=0,
                               signal_frequency=None, filtering_enabled=False, lowcut=None, highcut=None,
                               manage_exceptional_data=0, band_frequency_list=None,
                               min_sample_entropy_length=None, round_decimals=None):
        """
        See DIHC_FeatureExtractor.__init__ for the meaning of band_frequency_list,
        min_sample_entropy_length, and round_decimals. signal_frequency, lowcut,
        and highcut are validated there (Nyquist check) rather than here.
        """
        if len(data)==0:
            print(f'Data is empty...')
            exit(0)

        if signal_frequency is None:
            raise ValueError(
                "signal_frequency must be explicitly provided - no default sampling rate is assumed."
            )

        self.data = data
        self.segment_length = segment_length
        self.segment_overlap = segment_overlap
        self.signal_frequency = signal_frequency

        seg_len, seg_mov, num_segs = self.get_segment_metadata()
        sampPS = len(data) if segment_length is None else seg_len

        self.feat_extractor = DIHC_FeatureExtractor(
            manage_exceptional_data=manage_exceptional_data, signal_frequency=signal_frequency,
            sample_per_second=sampPS, filtering_enabled=filtering_enabled, lowcut=lowcut, highcut=highcut,
            varbose_progress=self.varbose_progress, band_frequency_list=band_frequency_list,
            min_sample_entropy_length=min_sample_entropy_length, round_decimals=round_decimals)
        all_feat_df = pd.DataFrame()

        if segment_length is None:
            print(f'Dealing with entire signal...')
            all_feat_df= self.feat_extractor.generate_features(1, data, feature_names)
        else:
            if seg_len > len(data):
                print(f'Data can\'t be segmented...')
                all_feat_df = self.feat_extractor.generate_features(1, data, feature_names)
            else:
                print(f'Data started segmenting for features: {feature_names}')
                if self.varbose_progress:
                    self.prog_bar = tqdm(total=num_segs, desc=f'Feature extraction started...', position=0, file=sys.stdout)
                else:
                    print(f'Feature extraction started...')
                self.data_segmenter = DIHC_DataSegmenter(data, segment_length=segment_length, segment_overlap=segment_overlap,
                                                         signal_frequency=signal_frequency, varbose_progress=self.varbose_progress)
                seg_generator = self.data_segmenter.generate_segments()
                seg_srl = 0
                while True:
                    try:
                        if self.varbose_progress:
                            self.prog_bar.set_description(f'Extracting features for segment# {seg_srl+1} ||')
                        else:
                            print(f'Extracting features for segment# {seg_srl+1}')
                        seg_data = next(seg_generator)
                        feat_df = self.feat_extractor.generate_features(seg_srl+1, seg_data, feature_names)
                        all_feat_df = pd.concat([all_feat_df, feat_df])
                        all_feat_df = all_feat_df.reset_index(drop=True)
                        if self.varbose_progress:
                            self.prog_bar.update(1.0)
                    except StopIteration:
                        break
                    seg_srl+=1
                if self.varbose_progress:
                    self.prog_bar.set_description(f'Finished extracting features for all segments...')
                    self.prog_bar.close()
                else:
                    print(f'Finished extracting features for all segments...')
        return all_feat_df


    # ## Validate_segment_data_for_signal_frequency
    def validate_segment_data_for_signal_frequency(self, data, signal_frequency):
        first_shape = len(data[0:])

        if signal_frequency is None:
            print(f'Signal frequency is None...')
            return True

        for sg in data[:-1, :]:
            if len(sg) != first_shape:
                return False
        if first_shape<len(data[-1, :]):
            return False
        return True



    # ## Feature extractor - from segments
    def extract_features_from_segments(self, data, feature_names=[], signal_frequency=None,
                               filtering_enabled=False, lowcut=None, highcut=None, manage_exceptional_data=0,
                               band_frequency_list=None, min_sample_entropy_length=None, round_decimals=None):
        if len(data)==0:
            print(f'Data is empty...')
            exit(0)

        if signal_frequency is None:
            raise ValueError(
                "signal_frequency must be explicitly provided - no default sampling rate is assumed."
            )

        segment_length = data[0, :] if signal_frequency==None else data[0, :]/signal_frequency
        self.data = data
        self.segment_length = segment_length
        self.segment_overlap = 0
        self.signal_frequency = signal_frequency

        seg_len, seg_mov, num_segs = self.get_segment_metadata()
        sampPS = len(data) if segment_length is None else seg_len

        isValid = self.validate_segment_data_for_signal_frequency(data, signal_frequency)

        if not isValid:
            print('The segment is not valid...')
            exit(0)

        self.feat_extractor = DIHC_FeatureExtractor(
            manage_exceptional_data=manage_exceptional_data, signal_frequency=signal_frequency,
            sample_per_second=sampPS, filtering_enabled=filtering_enabled, lowcut=lowcut,
            highcut=highcut, varbose_progress=self.varbose_progress, band_frequency_list=band_frequency_list,
            min_sample_entropy_length=min_sample_entropy_length, round_decimals=round_decimals)
        all_feat_df = pd.DataFrame()

        if self.varbose_progress:
            self.prog_bar = tqdm(total=len(data), desc=f'Feature extraction started...', position=0, file=sys.stdout)
        else:
            print(f'Feature extraction started...')

        for seg_srl, seg_data in enumerate(data):
            if self.varbose_progress:
                self.prog_bar.set_description(f'Extracting features for segment# {seg_srl+1} ||')
            else:
                print(f'Extracting features for segment# {seg_srl+1} ')
            feat_df = self.feat_extractor.generate_features(seg_srl+1, seg_data, feature_names)
            all_feat_df = pd.concat([all_feat_df, feat_df])
            all_feat_df = all_feat_df.reset_index(drop=True)
            if self.varbose_progress:
                self.prog_bar.update(1.0)

        if self.varbose_progress:
            self.prog_bar.set_description(f'Finished extracting features for all segments...')
            self.prog_bar.close()
        else:
            print(f'Finished extracting features for all segments...')

        return all_feat_df


    # ## Entropy profile extractor - from data
    def extract_sampEn_profile_from_data(self, data, segment_length=None, segment_overlap=0, signal_frequency=None):
        if len(data)==0:
            print(f'Data is empty...')
            exit(0)

        if signal_frequency is None:
            raise ValueError(
                "signal_frequency must be explicitly provided - no default sampling rate is assumed."
            )

        self.data = data
        self.segment_length = segment_length
        self.segment_overlap = segment_overlap
        self.signal_frequency = signal_frequency

        seg_len, seg_mov, num_segs = self.get_segment_metadata()

        self.entProf_extractor = DIHC_EntropyProfile(varbose_progress=self.varbose_progress)
        all_entProf_df = pd.DataFrame()

        if segment_length is None:
            print(f'Dealing with entire signal...')
            all_entProf_df = self.entProf_extractor.generate_sampEn_profile(data)
        else:
            if seg_len > len(data):
                print(f'Data can\'t be segmented...')
                all_entProf_df = self.entProf_extractor.generate_sampEn_profile(data)
            else:
                if self.varbose_progress:
                    self.prog_bar = tqdm(total=num_segs, desc=f'Entropy profile calculation started...', position=0, file=sys.stdout)
                else:
                    print(f'Entropy profile calculation started...')
                self.data_segmenter = DIHC_DataSegmenter(data, segment_length=segment_length, segment_overlap=segment_overlap,
                                                         signal_frequency=signal_frequency, varbose_progress=self.varbose_progress)
                seg_generator = self.data_segmenter.generate_segments()
                seg_num = 0
                while True:
                    try:
                        if self.varbose_progress:
                            self.prog_bar.set_description(f'Generating SampEn entropy profile for segment# {seg_num + 1} ||')
                        else:
                            print(f'Generating SampEn entropy profile for segment# {seg_num + 1} ')
                        seg_data = next(seg_generator)
                        entProf_df = self.entProf_extractor.generate_sampEn_profile(seg_data)
                        seg_num_df = pd.DataFrame([seg_num]*entProf_df.shape[0], columns=['Segment_No'])
                        seg_num_df = seg_num_df.reset_index(drop=True)
                        entProf_df = pd.concat([seg_num_df, entProf_df], axis=1)
                        all_entProf_df = pd.concat([all_entProf_df, entProf_df])
                        all_entProf_df = all_entProf_df.reset_index(drop=True)
                        if self.varbose_progress:
                            self.prog_bar.update(1.0)
                    except StopIteration:
                        break
                    seg_num += 1

                if self.varbose_progress:
                    self.prog_bar.set_description(f'Finished generating entropy profile for all segments...')
                    self.prog_bar.close()
                else:
                    print(f'Finished generating entropy profile for all segments...')
        return all_entProf_df