# -*- coding: utf-8 -*-
"""
File Name: DIHC_FeatureDetails.py
Optimized to remove dynamic Enum injection side effects.
"""

from enum import Enum
from typing import List

class DIHC_FeatureDetails:
    td_linear_statistical = ['maximum', 'minimum', 'mean', 'median', 'standardDeviation', 'variance', 'kurtosis', 'skewness', 'numberOfZeroCrossing', 'positiveToNegativeSampleRatio', 'positiveToNegativePeakRatio', 'meanAbsoluteValue']
    td_linear_statistical_short2 = ['Max', 'Min', 'Mean', 'Med', 'Std', 'Var', 'Kur', 'Skew', 'zCross', 'PN-Ratio', 'PN-PkRatio', 'MeanAbs']
    
    td_nonlinear_entropy = ['approximateEntropy', 'sampleEntropy', 'permutationEntropy', 'singularValueDecompositionEntropy', 'fuzzyEntropy', 'distributionEntropy', 'shannonEntropy', 'renyiEntropy']
    td_nonlinear_entropy_short2 = ['ApEn', 'SampEn', 'PermEn', 'SVDEn', 'FuzzEn', 'DistEn', 'ShanEn', 'RenEn']
    
    td_nonlinear_complexity_and_fractal_dimensions = ['lempelZivComplexity', 'hjorthMobility', 'hjorthComplexity', 'fisherInfo', 'petrosianFd', 'katzFd', 'higuchiFd', 'detrendedFluctuation']
    td_nonlinear_complexity_and_fractal_dimensions_short2 = ['LZComp', 'HjMob', 'HjComp', 'FishInfo', 'PetFd', 'KatzFd', 'HigFd', 'DetFluc']
    
    td_nonlinear_samp_entropy_profiling = ['entropyProfiled_total_sampleEntropy', 'entropyProfiled_average_sampleEntropy', 'entropyProfiled_maximum_sampleEntropy', 'entropyProfiled_minimum_sampleEntropy', 'entropyProfiled_median_sampleEntropy', 'entropyProfiled_standardDeviation_sampleEntropy', 'entropyProfiled_variance_sampleEntropy', 'entropyProfiled_kurtosis_sampleEntropy', 'entropyProfiled_skewness_sampleEntropy']
    td_nonlinear_samp_entropy_profiling_short2 = ['TotalSampEn', 'AvgSampEn', 'MaxSampEn', 'MinSampEn', 'MedSampEn', 'StdSampEn', 'VarSampEn', 'KurSampEn', 'SkewSampEn']
    
    fd_linear_statistical = ['fd_maximum', 'fd_minimum', 'fd_mean', 'fd_median', 'fd_standardDeviation', 'fd_variance', 'fd_kurtosis', 'fd_skewness']
    fd_linear_statistical_short2 = ['FD_Max', 'FD_Min', 'FD_Mean', 'FD_Med', 'FD_Std', 'FD_Var', 'FD_Kur', 'FD_Skew']
    
    fd_linear_statistical_binwise = ['fd_maximum_alpha', 'fd_minimum_alpha', 'fd_mean_alpha', 'fd_median_alpha', 'fd_standardDeviation_alpha', 'fd_variance_alpha', 'fd_kurtosis_alpha', 'fd_skewness_alpha', 'fd_maximum_beta', 'fd_minimum_beta', 'fd_mean_beta', 'fd_median_beta', 'fd_standardDeviation_beta', 'fd_variance_beta', 'fd_kurtosis_beta', 'fd_skewness_beta', 'fd_maximum_delta', 'fd_minimum_delta', 'fd_mean_delta', 'fd_median_delta', 'fd_standardDeviation_delta', 'fd_variance_delta', 'fd_kurtosis_delta', 'fd_skewness_delta', 'fd_maximum_theta', 'fd_minimum_theta', 'fd_mean_theta', 'fd_median_theta', 'fd_standardDeviation_theta', 'fd_variance_theta', 'fd_kurtosis_theta', 'fd_skewness_theta', 'fd_maximum_gamma', 'fd_minimum_gamma', 'fd_mean_gamma', 'fd_median_gamma', 'fd_standardDeviation_gamma', 'fd_variance_gamma', 'fd_kurtosis_gamma', 'fd_skewness_gamma']
    fd_linear_statistical_binwise_short2 = ['FD_MaxAlp', 'FD_MinAlp', 'FD_MeanAlp', 'FD_MedAlp', 'FD_StdAlp', 'FD_VarAlp', 'FD_KurAlp', 'FD_SkewAlp', 'FD_MaxBet', 'FD_MinBet', 'FD_MeanBet', 'FD_MedBet', 'FD_StdBet', 'FD_VarBet', 'FD_KurBet', 'FD_SkewBet', 'FD_MaxDel', 'FD_MinDel', 'FD_MeanDel', 'FD_MedDel', 'FD_StdDel', 'FD_VarDel', 'FD_KurDel', 'FD_SkewDel', 'FD_MaxThe', 'FD_MinThe', 'FD_MeanThe', 'FD_MedThe', 'FD_StdThe', 'FD_VarThe', 'FD_KurThe', 'FD_SkewThe', 'FD_MaxOth', 'FD_MinOth', 'FD_MeanOth', 'FD_MedOth', 'FD_StdOth', 'FD_VarOth', 'FD_KurOth', 'FD_SkewOth']
    
    fd_nonlinear_entropy = ['spectralEntropy']
    fd_nonlinear_entropy_short2 = ['SpecEn']
    
    fd_spectral_power = ['fd_bandPower']
    fd_spectral_power_short2 = ['fd_bandPw']
    
    fd_spectral_band_power = ['fd_bandPower_alpha', 'fd_bandPower_beta', 'fd_bandPower_delta', 'fd_bandPower_theta', 'fd_bandPower_gamma']
    fd_spectral_band_power_short2 = ['fd_bandPwAlp', 'fd_bandPwBet', 'fd_bandPwDel', 'fd_bandPwThe', 'fd_bandPwGam']
    
    band_frequency_list = {'alpha': (8, 14), 'beta': (14, 31), 'delta': (0, 5), 'theta': (5, 8), 'gamma': (31, 100)}

    comp_exp_list1 = ['shannonEntropy', 'renyiEntropy']
    comp_exp_list2 = ['approximateEntropy']
    comp_exp_list3 = ['sampleEntropy', 'fuzzyEntropy']
    comp_exp_list4 = fd_spectral_power + fd_spectral_band_power
    comp_exp_list5 = ['distributionEntropy']
    comp_exp_list6 = td_nonlinear_samp_entropy_profiling

    def __init__(self):
        self.all_features = (
            self.td_linear_statistical + self.td_nonlinear_entropy + 
            self.td_nonlinear_complexity_and_fractal_dimensions + self.td_nonlinear_samp_entropy_profiling + 
            self.fd_linear_statistical + self.fd_linear_statistical_binwise + self.fd_nonlinear_entropy + 
            self.fd_spectral_power + self.fd_spectral_band_power
        )
        self.all_features_short2 = (
            self.td_linear_statistical_short2 + self.td_nonlinear_entropy_short2 + 
            self.td_nonlinear_complexity_and_fractal_dimensions_short2 + self.td_nonlinear_samp_entropy_profiling_short2 + 
            self.fd_linear_statistical_short2 + self.fd_linear_statistical_binwise_short2 + 
            self.fd_nonlinear_entropy_short2 + self.fd_spectral_power_short2 + self.fd_spectral_band_power_short2
        )

    def map_feature_names(self, feature_list: List[str]) -> List[str]:
        feat_short_names = []
        for feat in feature_list:
            if feat in self.all_features:
                i = self.all_features.index(feat)
                feat_short_names.append(self.all_features_short2[i])
        return feat_short_names


class DIHC_FeatureGroup(Enum):
    tdLin = DIHC_FeatureDetails.td_linear_statistical
    tdNlEn = DIHC_FeatureDetails.td_nonlinear_entropy
    tdNlComFD = DIHC_FeatureDetails.td_nonlinear_complexity_and_fractal_dimensions
    tdNlEnSamProf = DIHC_FeatureDetails.td_nonlinear_samp_entropy_profiling
    td = tdLin + tdNlEn + tdNlComFD + tdNlEnSamProf
    
    fdLin = DIHC_FeatureDetails.fd_linear_statistical + DIHC_FeatureDetails.fd_linear_statistical_binwise
    fdNl = DIHC_FeatureDetails.fd_nonlinear_entropy + DIHC_FeatureDetails.fd_spectral_power + DIHC_FeatureDetails.fd_spectral_band_power
    fd = fdLin + fdNl
    
    all = td + fd

    @classmethod
    def get_filtered_features(cls, exclude_expensive_level: int = 0) -> List[str]:
        """Safely returns features as standard Python lists without altering Enum definitions."""
        base_features = list(cls.all.value)
        expensive_mappings = [
            DIHC_FeatureDetails.comp_exp_list1, DIHC_FeatureDetails.comp_exp_list2,
            DIHC_FeatureDetails.comp_exp_list3, DIHC_FeatureDetails.comp_exp_list4,
            DIHC_FeatureDetails.comp_exp_list5, DIHC_FeatureDetails.comp_exp_list6
        ]
        
        to_exclude = []
        if 1 <= exclude_expensive_level <= len(expensive_mappings):
            for sublist in expensive_mappings[exclude_expensive_level - 1:]:
                to_exclude.extend(sublist)
        elif exclude_expensive_level > len(expensive_mappings):
            for sublist in expensive_mappings:
                to_exclude.extend(sublist)

        return [f for f in base_features if f not in to_exclude]