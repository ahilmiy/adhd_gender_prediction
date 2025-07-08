import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def extract_meta_features(connectome_df: pd.DataFrame) -> pd.DataFrame:
    """
    Verilen 19900 boyutlu connectome verisinden özet (meta) istatistiksel özellikler üretir.

    Args:
        connectome_df (pd.DataFrame): fMRI connectome verisi (19900 feature)

    Returns:
        pd.DataFrame: meta özellikler (mean, std, min, max, median, skewness, kurtosis)
    """
    meta_df = pd.DataFrame()
    meta_df['mean_conn'] = connectome_df.mean(axis=1)
    meta_df['std_conn'] = connectome_df.std(axis=1)
    meta_df['min_conn'] = connectome_df.min(axis=1)
    meta_df['max_conn'] = connectome_df.max(axis=1)
    meta_df['median_conn'] = connectome_df.median(axis=1)
    meta_df['skewness_conn'] = connectome_df.apply(skew, axis=1)
    meta_df['kurtosis_conn'] = connectome_df.apply(kurtosis, axis=1)
    return meta_df
