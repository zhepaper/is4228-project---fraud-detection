"""
FinBank Credit Application Adapter
====================================
Demonstrates the generalisability of the Feature Role Mapping framework.

FinBank is a fictional new client dataset that was never seen during training.
Its column names are completely different from the BAF training data.

The adapter shows how a new dataset can be onboarded by:
  1. Mapping its columns to shared semantic roles
  2. Renaming columns to match the trained model's expected features
  3. Routing to the closest existing pipeline (BAF in this case)

This is the core value proposition of the modular pipeline:
  new client → feature role mapping → rename → existing model → prediction
  No retraining required.

Feature Role Mapping (FinBank → BAF):
  Role          FinBank column              BAF column
  ─────────────────────────────────────────────────────
  monetary      annual_income               income
  monetary      credit_limit_requested      proposed_credit_limit
  monetary      intended_balcon_amount      intended_balcon_amount
  temporal      application_month           month
  temporal      days_since_request          days_since_request
  transaction   channel_type                payment_type
  identity      free_email                  email_is_free
  identity      home_phone_verified         phone_home_valid
  identity      mobile_phone_verified       phone_mobile_valid
  identity      overseas_ip                 foreign_request
  identity      name_email_similarity       name_email_similarity
  velocity      app_count_1h                velocity_6h
  velocity      app_count_24h               velocity_24h
  velocity      velocity_4w                 velocity_4w
  velocity      zip_count_4w                zip_count_4w
  velocity      dob_emails_4w               date_of_birth_distinct_emails_4w
  stability     residence_months            prev_address_months_count
  stability     current_address_months_count current_address_months_count
  stability     banking_history_months      bank_months_count
  device        applicant_device            device_os
  device        device_email_count_8w       device_distinct_emails_8w
  device        keep_alive_session          keep_alive_session
  device        session_minutes             session_length_in_minutes
  device        bank_branch_count_8w        bank_branch_count_8w
  entity        customer_age                customer_age
  entity        risk_score                  credit_risk_score
  entity        has_other_cards             has_other_cards
  entity        application_source          source
  entity        employment_category         employment_status
  entity        residence_type              housing_status
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Column rename map: FinBank → BAF
# ---------------------------------------------------------------------------

COLUMN_MAP = {
    'annual_income'               : 'income',
    'credit_limit_requested'      : 'proposed_credit_limit',
    'intended_balcon_amount'      : 'intended_balcon_amount',
    'application_month'           : 'month',
    'days_since_request'          : 'days_since_request',
    'channel_type'                : 'payment_type',
    'free_email'                  : 'email_is_free',
    'home_phone_verified'         : 'phone_home_valid',
    'mobile_phone_verified'       : 'phone_mobile_valid',
    'overseas_ip'                 : 'foreign_request',
    'name_email_similarity'       : 'name_email_similarity',
    'app_count_1h'                : 'velocity_6h',
    'app_count_24h'               : 'velocity_24h',
    'velocity_4w'                 : 'velocity_4w',
    'zip_count_4w'                : 'zip_count_4w',
    'dob_emails_4w'               : 'date_of_birth_distinct_emails_4w',
    'residence_months'            : 'prev_address_months_count',
    'current_address_months_count': 'current_address_months_count',
    'banking_history_months'      : 'bank_months_count',
    'applicant_device'            : 'device_os',
    'device_email_count_8w'       : 'device_distinct_emails_8w',
    'keep_alive_session'          : 'keep_alive_session',
    'session_minutes'             : 'session_length_in_minutes',
    'bank_branch_count_8w'        : 'bank_branch_count_8w',
    'customer_age'                : 'customer_age',
    'risk_score'                  : 'credit_risk_score',
    'has_other_cards'             : 'has_other_cards',
    'application_source'          : 'source',
    'employment_category'         : 'employment_status',
    'residence_type'              : 'housing_status',
    'device_fraud_history'        : 'device_fraud_count',
}


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename FinBank columns to BAF equivalents.
    After renaming, the BAF adapter handles all preprocessing.

    This single function is the entire 'new client onboarding' cost.
    """
    df = df.copy()

    # Drop label column if present
    df = df.drop(columns=['fraud_label'], errors='ignore')

    # Rename columns using the feature role mapping
    df = df.rename(columns=COLUMN_MAP)

    return df
