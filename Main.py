#%% md
# ## Testing the DIHC Feature Manager Package
#%%
# %pip install ipywidgets tqdm
# jupyter nbextension enable --py widgetsnbextension
# !jupyter nbextension enable --py widgetsnbextension
#%%

#%%

#%% md
# ### Sample EEG Data Test
#%%

#%% md
# #### Settings and Data Reading
#%% md
# #####  Importing the package
# Load the package "DIHC_FeatureManager" which is in the same directory as this notebook (or your main python script/notebook)
#%%
# Importing necessary modules
import numpy as np
import pandas as pd
from DIHC_FeatureManager.DIHC_FeatureManager import *
#%%

#%% md
# ##### Data loading
# Reading sample data from the file "signal_data.csv" which is in the same directory as this notebook (or your main python script/notebook)
#%%
print(f'Data reading started...')
sample_df = pd.read_csv('./eeg_signal_data.csv')
print(f'Data reading completed...')
print(f"Data read from file: ")
sample_df
#%%

#%% md
# ##### Data inspection
# Observing the column names, shape and other basic information of the data
#%%
print(f'Columns available in the dataframe: {sample_df.columns}')
print(f'Columns available in the dataframe: {sample_df.shape}')
print(f'Dataframe details: ')
sample_df.info()
#%%

#%% md
# ##### Settings and partial sample data access
# This data file contains 3 columns: "time", "signal", and "label".
# Only the "signal" column that contains the (time series) data will be used for feature extraction.
# For simplicity only the first few seconds of the signal and few segments are used for feature extraction.
#%%
# Preset examples:
# 1) Low-frequency short recording
# signal_frequency = 10      # Hz
# segment_length   = 5       # s
# segment_overlap  = 0       # s
# total_segments   = 40000

# 2) Low-frequency long recording
# signal_frequency = 32      # Hz
# segment_length   = 90 * 60 # s
# segment_overlap  = 0       # s
# total_segments   = 4

# Active configuration
signal_frequency = 256  # Hz
segment_length = 5  # s
segment_overlap = 0  # s
total_segments = 4

print(f"""Settings:
- Signal frequency: {signal_frequency} Hz
- Segment length: {segment_length} s
- Segment overlap: {segment_overlap} s
- Total segments: {total_segments}
""")

#%%
print(f'Data sub-sampling started...')
# sample_data = np.array([52, 54, 6, 45, 14, 40, 42, 48, 52, 20, 28, 8, 63, 47, 23])

# sample_data = sample_df['signal'].values.tolist()
sample_data = sample_df.loc[:(total_segments*segment_length)*signal_frequency-1, 'signal'].values#.tolist()
# sample_data = sample_df.loc[:(total_segments*segment_length)*signal_frequency-1, 'signal'].values#.tolist()
# sample_data = sample_df.loc[:20*signal_frequency-1, 'signal'].values#.tolist()
# sample_data = sample_df.loc[:5100, 'signal'].values#.tolist()
# sample_data = sample_df.iloc[:20*256-1, 0:1].values#.tolist()
# print(len(sample_data))
# print(sample_data.shape, sample_data)
print(f'Data sub-sampling completed...')

print(f"Sample data shape: {sample_data.shape}")
sample_data
#%%

#%%

#%% md
# #### Segmentation Test
#%% md
# ##### Segmentation using `get_segments_for_data` function
# - Create the object of the class "DIHC_FeatureManager" and call the method "get_segments_for_data" to extract features from the data.
# - Use different parameters of the method "get_segments_for_data" to extract different number of segments.
#%%
print(f'Data segmentation started...')

feature_manager = DIHC_FeatureManager()
segmented_data_array = feature_manager.get_segments_for_data(sample_data, segment_length=segment_length, segment_overlap=segment_overlap, signal_frequency=signal_frequency)
# segmented_data_array = feature_manager.get_segments_for_data(sample_data, segment_length=5, signal_frequency=signal_frequency)
# segmented_data_array = feature_manager.get_segments_for_data(sample_data, segment_length=5, segment_overlap=3, signal_frequency=signal_frequency)

print(f'Data segmentation completed...')
#%%

#%% md
# ##### Display the segmented data
#%%
print(f"Segmented data array shape: {segmented_data_array.shape}")
print(f"Extracted segments data: ")
segmented_data_array
#%%

#%%

#%% md
# #### Feature Extraction Test
#%% md
# ##### Feature extraction using `extract_features_from_data` function
# - Create the object of the class "DIHC_FeatureManager" and call the method "extract_features_from_data" to extract features from the data
# - Use different parameters of the method "extract_features_from_data" to extract different features
# - Additionally can remove some computationally expensive features to save time and/or solve the memory exhaustion problem for larger segments
#%% md
# ###### Select features and/or remove computationally expensive features
#%%
# Comment of uncommenting the following line to remove computationally expensive features

feature_names = None  ###For all features- None works similarly as [DIHC_FeatureGroup.all]
# feature_names = [DIHC_FeatureGroup.tdNl, DIHC_FeatureGroup.fdLin]  ###For some specific features- in this case, Time-domain non-linear and frequency domain linear features
# feature_names = DIHC_FeatureGroup.remove_computationally_expensive_features( comp_exp_list_index=4 ) ###For all features, except the level-4 computationally expensive ones
# feature_names = DIHC_FeatureGroup.remove_computationally_expensive_features( feature_list=[DIHC_FeatureGroup.tdNl, DIHC_FeatureGroup.fdLin], comp_exp_list_index=4 ) ###For some specific features- in this case, Time-domain non-linear and frequency domain linear features, except the level-4 computationally expensive ones

# # sel_feats = ['approximateEntropy', 'sampleEntropy', 'shannonEntropy', 'lempelZivComplexity', 'fd_maximum_delta', 'fd_mean_alpha']
# sel_feats = ['approximateEntropy', 'sampleEntropy', 'shannonEntropy', 'lempelZivComplexity']
# feature_names = DIHC_FeatureGroup.select_some_specific_features(sel_feature_list=sel_feats)

print(f"Final feature list: {feature_names}")
#%%
print(f'Feature extraction started...')

feature_manager = DIHC_FeatureManager()
feature_df = pd.DataFrame()

if feature_names is None:
    feature_df = feature_manager.extract_features_from_data(sample_data, segment_length=segment_length, segment_overlap=segment_overlap, signal_frequency=signal_frequency)
else:
    feature_df = feature_manager.extract_features_from_data(sample_data, feature_names=feature_names, segment_length=segment_length, segment_overlap=segment_overlap, signal_frequency=signal_frequency)
# feature_df = feature_manager.extract_features_from_data(sample_data, segment_length=segment_length, segment_overlap=segment_overlap, signal_frequency=signal_frequency)
# feature_df = feature_manager.extract_features_from_data(sample_data, segment_length=5, signal_frequency=signal_frequency)
# feature_df = feature_manager.extract_features_from_data(sample_data, segment_length=5, segment_overlap=4, signal_frequency=signal_frequency)
# feature_df = feature_manager.extract_features_from_data(sample_data, feature_names=[DIHC_FeatureGroup.fdNlPw, DIHC_FeatureGroup.fdNlPwBnd], segment_length=5, signal_frequency=signal_frequency)
# feature_df = feature_manager.extract_features_from_data(sample_data, feature_names=[DIHC_FeatureGroup.tdNlEn, DIHC_FeatureGroup.td], segment_length=5, signal_frequency=signal_frequency)
# feature_df = feature_manager.extract_features_from_data(sample_data, feature_names=[DIHC_FeatureGroup.tdNlEn, DIHC_FeatureGroup.tdNl], segment_length=5, signal_frequency=signal_frequency)

print(f'Feature extraction completed...')
#%%

#%% md
# ##### Display all features
#%%
print(f"For a total of {feature_df.shape[0]} segments, {feature_df.shape[1]} features were extracted")
print(f"The name of the features are: {feature_df.columns}")
print(f"Extracted features: ")
feature_df

#%%

#%% md
# ##### Save all features
#%%
# # save_file_path = './eeg_all_features_matlab.csv'
# save_file_path = './eeg_all_features_python.csv'
# feature_df.to_csv(save_file_path, index=False)
# print(f"All EEG features successfully saved to: {save_file_path}")
#%%

#%%

#%% md
# #### Entropy (SampEn) Profile Test
#%% md
# ##### Sample Entropy (SampEn) Profile extraction
# - Create the object of the class "DIHC_FeatureManager" and call the method "extract_sampEn_profile_from_data" to extract Sample entropy (SampEn) profile from the data
# - Use different parameters of the method "extract_sampEn_profile_from_data" to extract entropy profile for Sample entropy (SampEn)
#%%
print(f'Entropy profile extraction started...')

feature_manager = DIHC_FeatureManager()

sampEn_Profile_df = feature_manager.extract_sampEn_profile_from_data(sample_data, segment_length=segment_length, segment_overlap=segment_overlap, signal_frequency=signal_frequency)
# sampEn_Profile_df = feature_manager.extract_sampEn_profile_from_data(sample_data, segment_length=5, signal_frequency=signal_frequency)
# sampEn_Profile_df = feature_manager.extract_sampEn_profile_from_data(sample_data, segment_length=5, signal_frequency=signal_frequency)
# sampEn_Profile_df = feature_manager.extract_sampEn_profile_from_data(sample_data, segment_length=5, signal_frequency=signal_frequency)
# sampEn_Profile_df = feature_manager.extract_sampEn_profile_from_data(sample_data, segment_length=5, segment_overlap=0, signal_frequency=signal_frequency)

print(f'Entropy profile extraction completed...')
#%%

#%% md
# ##### Display entropy profile data
#%%
print(f"SampEn entropy profile shape: {sampEn_Profile_df.shape}")
print(f"SampEn entropy profile values: ")
sampEn_Profile_df
#%%

#%% md
# ##### Save entropy profile data
#%%
# # save_file_path = './eeg_sampEn_profile_data_matlab.csv'
# save_file_path = './eeg_sampEn_profile_data_python.csv'
# sampEn_Profile_df.to_csv(save_file_path, index=False)
# print(f"SamEn EEG profile data successfully saved to: {save_file_path}")
#%%

#%%

#%%

#%% md
# ### Sample Hypnogram Data Test
#%%

#%% md
# #### Settings and Data Reading
#%% md
# #####  Importing the package
# Load the package "DIHC_FeatureManager" which is in the same directory as this notebook (or your main python script/notebook)
#%%
# Importing necessary modules
import numpy as np
import pandas as pd
from DIHC_FeatureManager.DIHC_FeatureManager import *
#%%

#%% md
# ##### Data loading
# Reading sample data from the file "signal_data.csv" which is in the same directory as this notebook (or your main python script/notebook)
#%%
print(f'Data reading started...')
sample_df = pd.read_csv('./hypnogram_signal_data.csv')
print(f'Data reading completed...')
print(f"Data read from file: ")
sample_df
#%%

#%% md
# ##### Data inspection
# Observing the column names, shape and other basic information of the data
#%%
print(f'Columns available in the dataframe: {sample_df.columns}')
print(f'Columns available in the dataframe: {sample_df.shape}')
print(f'Dataframe details: ')
sample_df.info()
#%%

#%% md
# ##### Settings and partial sample data access
# This data file contains 3 columns: "time", "signal", and "label".
# Only the "signal" column that contains the (time series) data will be used for feature extraction.
# For simplicity only the first few seconds of the signal and few segments are used for feature extraction.
#%%
signal_frequency = None
pseudo_signal_frequency = 256  # It should not be useful for hypnogram data
epocho_length = 30  # seconds
sample_per_minute = 2  # epochs per minute
segment_length = 90  # minutes
segment_overlap = 0  # minutes
total_segments = 4

print(f"""Settings:
- Pseudo signal frequency: {pseudo_signal_frequency} Hz
- Epoch length: {epocho_length} s
- Epochs per minute: {sample_per_minute}
- Segment length: {segment_length} min
- Segment overlap: {segment_overlap} min
- Total segments: {total_segments}
""")

#%%
print(f'Data sub-sampling started...')
# sample_data = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 2, 2, 4, 4, 4, 2, 2, 0, 0])

# sample_data = sample_df['signal'].values.tolist()
sample_data = sample_df.loc[:total_segments*(segment_length*sample_per_minute)-1, 'signal'].values#.tolist()

# print(len(sample_data))
# print(sample_data.shape, sample_data)
print(f'Data sub-sampling completed...')

print(f"Sample data shape: {sample_data.shape}")
sample_data
#%%

#%%

#%% md
# #### Feature Extraction (Hypnogram) Test
#%% md
# ##### Feature extraction using `extract_features_from_data` function
# - Create the object of the class "DIHC_FeatureManager" and call the method "extract_features_from_data" to extract features from the data
# - Use different parameters of the method "extract_features_from_data" to extract different features
# - Additionally can remove some computationally expensive features to save time and/or solve the memory exhaustion problem for larger segments
# - Additionally can also select some specific features (in this case, only the features that are related to hypnogram data are selected)
#%% md
# ###### Select features and/or remove computationally expensive features
#%%
from DIHC_FeatureManager.DIHC_FeatureManager import DIHC_FeatureGroup


# Comment of uncommenting the following line to remove computationally expensive features

feature_names = None  ###For all features- None works similarly as [DIHC_FeatureGroup.all]
# feature_names = [DIHC_FeatureGroup.tdNl, DIHC_FeatureGroup.fdLin]  ###For some specific features- in this case, Time-domain non-linear and frequency domain linear features
# feature_names = DIHC_FeatureGroup.remove_computationally_expensive_features( comp_exp_list_index=4 ) ###For all features, except the level-4 computationally expensive ones
# feature_names = DIHC_FeatureGroup.remove_computationally_expensive_features( feature_list=[DIHC_FeatureGroup.tdNl, DIHC_FeatureGroup.fdLin], comp_exp_list_index=4 ) ###For some specific features- in this case, Time-domain non-linear and frequency domain linear features, except the level-4 computationally expensive ones

# # sel_feats = ['approximateEntropy', 'sampleEntropy', 'shannonEntropy', 'lempelZivComplexity', 'fd_maximum_delta', 'fd_mean_alpha']
sel_feats = ['approximateEntropy', 'sampleEntropy', 'shannonEntropy', 'lempelZivComplexity']
rem_feats_pattern = ['fd_', 'entropyProfiled_', 'spectral', 'positiveToNegative']
# feature_names = DIHC_FeatureGroup.select_some_specific_features(sel_feature_list=sel_feats, rem_feature_list=rem_feats_pattern)
# feature_names = DIHC_FeatureGroup.select_some_specific_features(sel_feature_list=sel_feats, rem_feature_list=None)
feature_names = DIHC_FeatureGroup.select_some_specific_features(sel_feature_list=None, rem_feature_list=rem_feats_pattern)
# feature_names = DIHC_FeatureGroup.select_some_specific_features(sel_feature_list=None, rem_feature_list=None)

print(f"Final feature list: {feature_names}")
#%%

#%%
print(f'Feature extraction started...')

feature_manager = DIHC_FeatureManager()
feature_df = pd.DataFrame()

if feature_names is None:
    feature_df = feature_manager.extract_features_from_data(sample_data, segment_length=(segment_length*sample_per_minute), segment_overlap=(segment_overlap*sample_per_minute), signal_frequency=signal_frequency)
else:
    feature_df = feature_manager.extract_features_from_data(sample_data, feature_names=feature_names, segment_length=(segment_length*sample_per_minute), segment_overlap=(segment_overlap*sample_per_minute), signal_frequency=signal_frequency)
# feature_df = feature_manager.extract_features_from_data(sample_data, segment_length=segment_length, segment_overlap=segment_overlap, signal_frequency=signal_frequency)
# feature_df = feature_manager.extract_features_from_data(sample_data, segment_length=5, signal_frequency=signal_frequency)
# feature_df = feature_manager.extract_features_from_data(sample_data, segment_length=5, segment_overlap=4, signal_frequency=signal_frequency)
# feature_df = feature_manager.extract_features_from_data(sample_data, feature_names=[DIHC_FeatureGroup.fdNlPw, DIHC_FeatureGroup.fdNlPwBnd], segment_length=5, signal_frequency=signal_frequency)
# feature_df = feature_manager.extract_features_from_data(sample_data, feature_names=[DIHC_FeatureGroup.tdNlEn, DIHC_FeatureGroup.td], segment_length=5, signal_frequency=signal_frequency)
# feature_df = feature_manager.extract_features_from_data(sample_data, feature_names=[DIHC_FeatureGroup.tdNlEn, DIHC_FeatureGroup.tdNl], segment_length=5, signal_frequency=signal_frequency)

print(f'Feature extraction completed...')
#%%

#%% md
# ##### Display all features
#%%
print(f"For a total of {feature_df.shape[0]} segments, {feature_df.shape[1]} features were extracted")
print(f"The name of the features are: {feature_df.columns}")
print(f"Extracted features: ")
feature_df

#%%

#%% md
# ##### Save all features
#%%
# # save_file_path = './hyp_all_features_matlab.csv'
# save_file_path = './hyp_all_features_python.csv'
# feature_df.to_csv(save_file_path, index=False)
# print(f"All hypnogram features successfully saved to: {save_file_path}")
#%%

#%%

#%%

#%%

#%%
