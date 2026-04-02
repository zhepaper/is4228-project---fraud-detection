"""
Shared Preprocessing Core
==========================
Generic preprocessing functions organised by feature role.
Each function takes a DataFrame and a list of column names,
applies the transformation, and returns the modified DataFrame.

Dataset-specific adapters call these functions after mapping
their raw columns to semantic roles via feature_roles.py.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# ---------------------------------------------------------------------------
# Temporal features
# ---------------------------------------------------------------------------

def process_temporal_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Handle temporal columns that are already numeric (e.g. step, month).
    No transformation needed — just validate they exist.
    """
    for col in cols:
        if col not in df.columns:
            df[col] = 0   # fill missing temporal cols with 0
    return df


def process_temporal_unix(df: pd.DataFrame, col: str, reference_dt=None) -> pd.DataFrame:
    """
    Extract time components from a Unix/seconds-based timestamp column.
    Used by IEEE adapter for TransactionDT.

    Adds: hour, dayofweek, day, month, is_night, is_weekend
    """
    if col not in df.columns:
        return df

    dt_series = pd.to_datetime(df[col], unit='s', origin=reference_dt or 'unix')

    df['hour']      = dt_series.dt.hour
    df['dayofweek'] = dt_series.dt.dayofweek
    df['day']       = dt_series.dt.day
    df['month']     = dt_series.dt.month
    df['is_night']  = (df['hour'] < 6).astype(int)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    return df

# ---------------------------------------------------------------------------
# Monetary features
# ---------------------------------------------------------------------------

def process_monetary(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Apply log1p transformation to monetary columns.
    Handles negative values by shifting before log (clip at 0).

    Adds: <col>_log for each column in cols.
    """
    for col in cols:
        if col not in df.columns:
            continue
        # Shift negative values to 0 before log (e.g. BAF intended_balcon_amount)
        shifted = df[col].clip(lower=0)
        df[f'{col}_log'] = np.log1p(shifted)
    return df


def flag_extreme_amounts(df: pd.DataFrame, col: str, threshold_pct: float = 99.5) -> pd.DataFrame:
    """
    Add a binary flag for transactions with extreme amounts (above threshold percentile).
    Useful as an additional signal without distorting the distribution.
    """
    if col not in df.columns:
        return df
    cutoff = df[col].quantile(threshold_pct / 100)
    df[f'{col}_extreme'] = (df[col] > cutoff).astype(int)
    return df

# ---------------------------------------------------------------------------
# Categorical features
# ---------------------------------------------------------------------------

def encode_categoricals_ordinal(
    df: pd.DataFrame,
    cols: list,
    encoder: OrdinalEncoder = None,
    fit: bool = False,
) -> tuple[pd.DataFrame, OrdinalEncoder]:
    """
    Apply OrdinalEncoder to low-cardinality categorical columns.
    Tree-based models (LightGBM, XGBoost) work well with ordinal integers.

    Args:
        df      : input DataFrame
        cols    : list of categorical column names
        encoder : existing fitted OrdinalEncoder (for transform-only mode)
        fit     : if True, fit a new encoder on this data

    Returns:
        (transformed DataFrame, fitted encoder)
    """
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df, encoder

    if fit:
        encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-2,   # distinct from sentinel -1
        )
        df[cols] = encoder.fit_transform(df[cols].astype(str))
    else:
        df[cols] = encoder.transform(df[cols].astype(str))

    return df, encoder


def encode_categoricals_frequency(df: pd.DataFrame, cols: list, freq_map: dict = None) -> tuple:
    """
    Apply frequency encoding to high-cardinality categorical columns
    (e.g. email domains, device strings in IEEE).

    Replaces each category with its frequency in the training data.
    Unseen categories get frequency 0.

    Args:
        df       : input DataFrame
        cols     : list of categorical column names
        freq_map : dict of {col: {value: frequency}} — pass None to fit

    Returns:
        (transformed DataFrame, freq_map)
    """
    if freq_map is None:
        freq_map = {}
        for col in cols:
            if col in df.columns:
                freq_map[col] = df[col].value_counts(normalize=True).to_dict()

    for col in cols:
        if col in df.columns and col in freq_map:
            df[col] = df[col].map(freq_map[col]).fillna(0)

    return df, freq_map


def fill_categorical_missing(df: pd.DataFrame, cols: list, fill_value: str = 'missing') -> pd.DataFrame:
    """Fill NaN in categorical columns with a placeholder string."""
    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)
    return df

# ---------------------------------------------------------------------------
# Numeric missing values
# ---------------------------------------------------------------------------

def fill_numeric_sentinel(df: pd.DataFrame, cols: list, sentinel: float = -999) -> pd.DataFrame:
    """
    Fill NaN in numeric columns with a sentinel value.
    LightGBM and XGBoost can learn from sentinel values directly,
    treating them as a distinct signal rather than true missingness.
    Used by IEEE adapter for V/D/C features.
    """
    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna(sentinel)
    return df


def fill_numeric_median(df: pd.DataFrame, cols: list, medians: dict = None) -> tuple:
    """
    Fill NaN in numeric columns with the median (fit on training data).
    Use when sentinel values would distort the feature distribution.

    Returns:
        (transformed DataFrame, medians dict)
    """
    if medians is None:
        medians = {col: df[col].median() for col in cols if col in df.columns}

    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna(medians.get(col, 0))

    return df, medians


def drop_high_missing(df: pd.DataFrame, threshold: float = 0.9) -> tuple:
    """
    Drop columns where the fraction of missing values exceeds threshold.
    Used by IEEE adapter to remove sparse V features.

    Returns:
        (filtered DataFrame, list of dropped column names)
    """
    missing_frac = df.isnull().mean()
    drop_cols = missing_frac[missing_frac > threshold].index.tolist()
    df = df.drop(columns=drop_cols)
    return df, drop_cols

# ---------------------------------------------------------------------------
# Binary / flag features
# ---------------------------------------------------------------------------

def add_sentinel_flags(df: pd.DataFrame, cols: list, sentinel: float = -1) -> pd.DataFrame:
    """
    For columns that use a sentinel value to encode missingness,
    add a binary flag column (<col>_missing) before any imputation.
    The flag itself carries fraud signal (e.g. BAF prev_address_months_count).
    """
    for col in cols:
        if col in df.columns:
            df[f'{col}_missing'] = (df[col] == sentinel).astype(int)
    return df
