"""
IEEE Adapter
============
Transforms raw IEEE-CIS fraud detection data into model-ready features.

IEEE represents a rich online payment fraud setting with 400+ raw columns.
Key challenges: high dimensionality, sparse V features, identity table join,
and high-cardinality categorical columns.

Feature roles covered:
  - temporal    : hour, dayofweek, is_night, is_weekend (from TransactionDT)
  - monetary    : TransactionAmt log + decimal part
  - transaction : ProductCD
  - identity    : email domains, card info, address
  - velocity    : C features (count aggregates)
  - stability   : D features (time-since-last-event)
  - device      : DeviceType, DeviceInfo, id_ series
  - relational  : V features (Vesta-proprietary engineered signals)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.core.feature_roles import get_common_features
from backend.core.preprocessing import (
    fill_categorical_missing,
    fill_numeric_sentinel,
    drop_high_missing,
    process_temporal_unix,
)

TARGET  = "isFraud"
DATASET = "ieee"

# Reference date for TransactionDT (seconds since 2017-12-01)
REFERENCE_DATE = "2017-12-01"

# Columns to drop before modelling
DROP_COLS = ["TransactionID", "TransactionDT", TARGET]

# ---------------------------------------------------------------------------
# Preprocessing & feature engineering
# ---------------------------------------------------------------------------

def load_and_merge(
    transaction_path: str,
    identity_path: str = None,
) -> pd.DataFrame:
    """
    Load transaction CSV and optionally merge with identity table.
    Only ~24% of transactions have identity info — use left join to keep all.

    Args:
        transaction_path : path to train/test transaction CSV
        identity_path    : path to train/test identity CSV (optional)

    Returns:
        merged DataFrame
    """
    df = pd.read_csv(transaction_path)

    if identity_path:
        identity = pd.read_csv(identity_path)
        # Standardise column names: some versions use hyphens
        identity.columns = identity.columns.str.replace('-', '_')
        df = df.merge(identity, on='TransactionID', how='left')

    return df


def preprocess(
    df: pd.DataFrame,
    label_encoders: dict = None,
    drop_cols_from_training: list = None,
    fit: bool = False,
) -> tuple[pd.DataFrame, dict, list]:
    """
    Apply full IEEE preprocessing pipeline.

    Steps:
      1. Drop columns with >90% missing (sparse V features)
      2. Extract time features from TransactionDT (temporal role)
      3. Engineer email domain features (identity role)
      4. Engineer amount features (monetary role)
      5. Create card+address combination feature (identity role)
      6. Fill categorical missing → label encode
      7. Fill numeric missing with sentinel -999

    Args:
        df                     : merged transaction + identity DataFrame
        label_encoders         : dict of {col: LabelEncoder} — pass None when fit=True
        drop_cols_from_training: columns dropped during training (for consistent test transform)
        fit                    : if True, fit new encoders on this data

    Returns:
        (preprocessed DataFrame, label_encoders dict, list of dropped columns)
    """
    df = df.copy()

    # --- Type casting (handles API raw inputs) ---
    # Convert numeric columns from string to float/int before calculations
    potential_num_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in potential_num_cols:
        try:
            numeric_check = pd.to_numeric(df[col], errors='coerce')
            if numeric_check.notnull().any():
                df[col] = numeric_check
        except:
            continue

    # --- Drop high-missing columns (relational role — sparse V features) ---
    if fit:
        df, dropped = drop_high_missing(df, threshold=0.9)
        drop_cols_from_training = dropped
    else:
        # Apply same drops as training to keep consistent feature space
        cols_to_drop = [c for c in (drop_cols_from_training or []) if c in df.columns]
        df = df.drop(columns=cols_to_drop)

    # --- Temporal features ---
    # TransactionDT is seconds elapsed since a reference date
    df = process_temporal_unix(df, col='TransactionDT', reference_dt=REFERENCE_DATE)

    # --- Email domain features (identity role) ---
    for email_col in ['P_emaildomain', 'R_emaildomain']:
        if email_col in df.columns:
            domain_col = f'{email_col}_domain'
            df[domain_col] = df[email_col].fillna('unknown').str.split('.').str[0]

            # Flag mainstream email providers — fraud often uses free/obscure domains
            mainstream = {'gmail', 'yahoo', 'hotmail', 'outlook', 'aol', 'icloud', 'live', 'msn'}
            df[f'{email_col}_is_mainstream'] = df[domain_col].isin(mainstream).astype(int)

    # --- Monetary features (monetary role) ---
    if 'TransactionAmt' in df.columns:
        df['TransactionAmt_log']     = np.log1p(df['TransactionAmt'])
        # Decimal part: fraud often uses round amounts, legitimate purchases don't
        df['TransactionAmt_decimal'] = df['TransactionAmt'] - df['TransactionAmt'].astype(int)

    # --- Card + address combination (identity role) ---
    # Fingerprint for a specific card at a specific billing address
    if 'card1' in df.columns and 'addr1' in df.columns:
        df['card1_addr1'] = (
            df['card1'].astype(str) + '_' + df['addr1'].astype(str)
        )

    # --- Identify categorical and numeric columns ---
    cols_to_exclude = [c for c in DROP_COLS if c in df.columns]
    remaining = [c for c in df.columns if c not in cols_to_exclude]
    cat_cols = df[remaining].select_dtypes(include='object').columns.tolist()
    num_cols = df[remaining].select_dtypes(include=[np.number]).columns.tolist()

    # --- Categorical: fill missing → label encode ---
    df = fill_categorical_missing(df, cat_cols, fill_value='missing')

    if fit:
        label_encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    else:
        for col in cat_cols:
            if col in (label_encoders or {}):
                le = label_encoders[col]
                # Handle unseen labels gracefully
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in le.classes_ else 'missing'
                )
                df[col] = le.transform(df[col])
            else:
                df[col] = 0   # fallback for columns not seen during training

    # --- Numeric: fill missing with sentinel -999 ---
    # LightGBM learns that -999 means "this information was not available"
    df = fill_numeric_sentinel(df, num_cols, sentinel=-999)

    # --- Drop modelling-irrelevant columns ---
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    return df, label_encoders, drop_cols_from_training


def get_feature_list(df_preprocessed: pd.DataFrame, common_only: bool = False) -> list:
    """
    Return the feature list from a preprocessed DataFrame.

    Args:
        df_preprocessed : output of preprocess()
        common_only     : if True, return only features mapped to common roles
    """
    if common_only:
        return get_common_features(DATASET)
    return df_preprocessed.columns.tolist()
