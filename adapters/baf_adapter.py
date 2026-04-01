"""
BAF Adapter
===========
Transforms raw Bank Account Fraud (BAF) data into model-ready features.

BAF represents a bank account application fraud setting.
Its key signals come from identity verification gaps, behavioral velocity,
and historical stability — plus sentinel -1 values that encode
missing history as an explicit fraud signal.

Feature roles covered:
  - temporal    : month, days_since_request
  - monetary    : income, proposed_credit_limit, intended_balcon_amount
  - transaction : payment_type
  - identity    : phone/email verification, foreign_request
  - velocity    : application frequency in 6h/24h/4w windows
  - stability   : address and banking history (with sentinel -1 flags)
  - device      : device OS, session behavior
  - entity      : customer age, credit score, housing/employment
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from fraud_pipeline.core.feature_roles import get_common_features
from fraud_pipeline.core.preprocessing import (
    add_sentinel_flags,
    encode_categoricals_ordinal,
)

# ---------------------------------------------------------------------------
# Feature list
# ---------------------------------------------------------------------------

FEATURES = [
    # --- temporal ---
    "month",
    "days_since_request",
    # --- monetary ---
    "income",
    "proposed_credit_limit",
    "intended_balcon_amount",
    # --- transaction ---
    "payment_type",
    # --- identity ---
    "email_is_free",
    "phone_home_valid",
    "phone_mobile_valid",
    "foreign_request",
    "name_email_similarity",
    "phone_not_verified",           # engineered
    # --- velocity ---
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "velocity_ratio_6h_24h",        # engineered
    "zip_count_4w",
    "date_of_birth_distinct_emails_4w",
    # --- stability ---
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "prev_address_missing",         # engineered sentinel flag
    "bank_months_missing",          # engineered sentinel flag
    "address_instability",          # engineered
    # --- device ---
    "device_os",
    "device_distinct_emails_8w",
    "keep_alive_session",
    "session_length_in_minutes",
    "bank_branch_count_8w",
    # --- entity ---
    "customer_age",
    "credit_risk_score",
    "has_other_cards",
    "source",
    "employment_status",
    "housing_status",
]

# Columns to apply ordinal encoding
CAT_COLS = ["payment_type", "employment_status", "housing_status", "source", "device_os"]

TARGET  = "fraud_bool"
DATASET = "baf"

# ---------------------------------------------------------------------------
# Preprocessing & feature engineering
# ---------------------------------------------------------------------------

def preprocess(
    df: pd.DataFrame,
    encoder: OrdinalEncoder = None,
    fit: bool = False,
) -> tuple[pd.DataFrame, OrdinalEncoder]:
    """
    Apply full BAF preprocessing pipeline.

    Steps:
      1. Add sentinel flags for -1 values (stability role)
      2. Engineer identity combination feature
      3. Engineer velocity ratio feature
      4. Engineer address instability score
      5. Ordinal-encode categorical columns

    Args:
        df      : raw BAF DataFrame (Base or any Variant)
        encoder : fitted OrdinalEncoder — pass None when fit=True
        fit     : if True, fit a new encoder on this data
                  if False, use the provided encoder (for Variant datasets)

    Returns:
        (preprocessed DataFrame, fitted encoder)
    """
    df = df.copy()

    # --- Sentinel flags (stability role) ---
    # -1 means "no record exists", not a true missing value
    # Missing history is itself a strong fraud signal
    # Note: named explicitly to keep column names short and consistent
    df['prev_address_missing'] = (df['prev_address_months_count'] == -1).astype(int)
    df['bank_months_missing']  = (df['bank_months_count'] == -1).astype(int)

    # --- Identity features (identity role) ---
    # Neither phone verified = stronger risk signal than either alone
    df['phone_not_verified'] = (
        (df['phone_home_valid'] == 0) & (df['phone_mobile_valid'] == 0)
    ).astype(int)

    # --- Velocity features (velocity role) ---
    # Ratio of short-term to medium-term application rate
    # High ratio = sudden burst of applications in the last 6 hours
    df['velocity_ratio_6h_24h'] = df['velocity_6h'] / (df['velocity_24h'] + 1)

    # --- Stability features (stability role) ---
    # Combined address instability score (0 = stable, 2 = both signals present)
    df['address_instability'] = (
        df['prev_address_missing'] +
        (df['current_address_months_count'] < 6).astype(int)
    )

    # --- Categorical encoding ---
    df, encoder = encode_categoricals_ordinal(df, CAT_COLS, encoder=encoder, fit=fit)

    return df, encoder


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
