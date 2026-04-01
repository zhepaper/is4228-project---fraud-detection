"""
PaySim Adapter
==============
Transforms raw PaySim transaction data into model-ready features.

PaySim represents a synthetic sequential payment fraud setting.
Its key fraud signals come from balance consistency anomalies,
not from identity or device signals (which don't exist in this dataset).

Feature roles covered:
  - temporal    : step, hour_of_day
  - monetary    : amount, log_amount, amountToBalanceRatio
  - transaction : is_transfer, is_cashout
  - stability   : balance error features, drain flag  (PaySim-specific)
"""

import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from fraud_pipeline.core.feature_roles import FEATURE_ROLES, get_common_features, get_all_features
from fraud_pipeline.core.preprocessing import process_monetary

# ---------------------------------------------------------------------------
# Feature list
# ---------------------------------------------------------------------------

# All features used by the PaySim model (after engineering)
FEATURES = [
    # --- temporal ---
    "hour_of_day",
    # --- monetary ---
    "log_amount",
    "amountToBalanceRatio",
    # --- transaction ---
    "is_transfer",
    "is_cashout",
    # --- stability (PaySim-specific balance signals) ---
    "errorBalanceOrig",
    "errorBalanceDest",
    "errorIsSmallOrig",
    "errorIsSmallDest",
    "balanceDrain",
    "destError",
]

TARGET  = "isFraud"
DATASET = "paysim"

# ---------------------------------------------------------------------------
# Preprocessing & feature engineering
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full PaySim preprocessing pipeline.

    Steps:
      1. Drop columns not used for modelling
      2. Encode transaction type as binary flags (temporal role)
      3. Engineer balance consistency features (stability role)
      4. Log-transform amount (monetary role)
      5. Extract hour of day (temporal role)

    Args:
        df : raw PaySim DataFrame

    Returns:
        df with all engineered features added
    """
    df = df.copy()

    # --- Transaction type encoding (transaction role) ---
    # Fraud only occurs in TRANSFER and CASH_OUT — encode as binary flags
    df['is_transfer'] = (df['type'] == 'TRANSFER').astype(int)
    df['is_cashout']  = (df['type'] == 'CASH_OUT').astype(int)

    # --- Balance consistency features (stability role) ---
    # Fraudulent transactions have near-zero balance errors
    # (the books balance perfectly because the money was simply moved)
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']

    # Flag near-zero errors — strong fraud signal
    df['errorIsSmallOrig'] = (df['errorBalanceOrig'].abs() < 1).astype(int)
    df['errorIsSmallDest'] = (df['errorBalanceDest'].abs() < 1).astype(int)

    # Balance drain: account had funds before, zero after — 97% of fraud does this
    df['balanceDrain'] = (
        (df['oldbalanceOrg'] > 0) & (df['newbalanceOrig'] == 0)
    ).astype(int)

    # Destination account error (receiver side discrepancy)
    df['destError'] = df['newbalanceDest'] - df['oldbalanceDest'] - df['amount']

    # --- Monetary features ---
    df['log_amount'] = np.log1p(df['amount'])

    # Amount as fraction of sender's original balance (avoids division by zero)
    df['amountToBalanceRatio'] = df['amount'] / (df['oldbalanceOrg'] + 1)

    # --- Temporal features ---
    df['hour_of_day'] = df['step'] % 24

    return df


def get_feature_list(common_only: bool = False) -> list:
    """
    Return the feature list used for model training.

    Args:
        common_only : if True, return only features mapped to common roles
                      (temporal + monetary + transaction) for cross-dataset experiment
    """
    if common_only:
        return get_common_features(DATASET)
    return FEATURES
