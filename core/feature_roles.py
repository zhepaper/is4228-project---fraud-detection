"""
Feature Role Mapping
====================
Each dataset has different column names, but columns can be grouped
into shared semantic roles. This mapping is the foundation of the
modular pipeline — adapters use roles, not raw column names.

Common roles (present in all 3 datasets):
  - temporal      : time-related signals
  - monetary      : amount / value signals
  - transaction   : transaction type / channel signals

Specific roles (dataset-dependent):
  - identity      : identity verification signals     (BAF, IEEE)
  - velocity      : behavioral frequency signals      (BAF, IEEE)
  - stability     : historical account stability      (PaySim, BAF)
  - device        : device / environment signals      (BAF, IEEE)
  - entity        : account / customer profile        (BAF only)
  - relational    : engineered network signals        (IEEE only)
"""

# ---------------------------------------------------------------------------
# Role definitions per dataset
# ---------------------------------------------------------------------------

FEATURE_ROLES = {

    "paysim": {
        "temporal": [
            "step",
            "hour_of_day",
        ],
        "monetary": [
            "amount",
            "log_amount",
            "amountToBalanceRatio",
        ],
        "transaction": [
            "is_transfer",
            "is_cashout",
        ],
        # Balance consistency signals — PaySim's strongest fraud indicators
        "stability": [
            "errorBalanceOrig",
            "errorBalanceDest",
            "errorIsSmallOrig",
            "errorIsSmallDest",
            "balanceDrain",
            "destError",
        ],
        # PaySim has no identity, velocity, device, or relational roles
        # This is the key limitation of synthetic data
    },

    "baf": {
        "temporal": [
            "month",
            "days_since_request",
        ],
        "monetary": [
            "income",
            "proposed_credit_limit",
            "intended_balcon_amount",
        ],
        "transaction": [
            "payment_type",
        ],
        "identity": [
            "email_is_free",
            "phone_home_valid",
            "phone_mobile_valid",
            "foreign_request",
            "name_email_similarity",
            "phone_not_verified",        # engineered: both phones unverified
        ],
        "velocity": [
            "velocity_6h",
            "velocity_24h",
            "velocity_4w",
            "velocity_ratio_6h_24h",     # engineered: short vs long-term ratio
            "zip_count_4w",
            "date_of_birth_distinct_emails_4w",
        ],
        "stability": [
            "prev_address_months_count",
            "current_address_months_count",
            "bank_months_count",
            "prev_address_missing",      # engineered: sentinel -1 flag
            "bank_months_missing",       # engineered: sentinel -1 flag
            "address_instability",       # engineered: combined instability score
        ],
        "device": [
            "device_os",
            "device_distinct_emails_8w",
            "keep_alive_session",
            "session_length_in_minutes",
            "bank_branch_count_8w",
        ],
        "entity": [
            "customer_age",
            "credit_risk_score",
            "has_other_cards",
            "source",
            "employment_status",
            "housing_status",
        ],
    },

    "ieee": {
        "temporal": [
            "hour",
            "dayofweek",
            "day",
            "month",
            "is_night",
            "is_weekend",
        ],
        "monetary": [
            "TransactionAmt",
            "TransactionAmt_log",
            "TransactionAmt_decimal",
        ],
        "transaction": [
            "ProductCD",
        ],
        "identity": [
            "P_emaildomain_domain",
            "R_emaildomain_domain",
            "P_emaildomain_is_mainstream",
            "R_emaildomain_is_mainstream",
            "card1", "card2", "card3", "card4", "card5", "card6",
            "addr1", "addr2",
            "card1_addr1",               # engineered: billing address fingerprint
        ],
        "velocity": [
            # C features: count-type aggregated behavioral signals
            "C1", "C2", "C3", "C4", "C5", "C6", "C7",
            "C8", "C9", "C10", "C11", "C12", "C13", "C14",
        ],
        "stability": [
            # D features: time-difference signals (days since last event)
            "D1", "D2", "D3", "D4", "D5", "D6", "D7",
            "D8", "D9", "D10", "D11", "D12", "D13", "D14", "D15",
        ],
        "device": [
            "DeviceType",
            "DeviceInfo",
            "id_12", "id_13", "id_14", "id_15", "id_16",
            "id_17", "id_18", "id_19", "id_20",
            "id_28", "id_29", "id_30", "id_31", "id_32",
            "id_33", "id_34", "id_35", "id_36", "id_37", "id_38",
        ],
        "relational": [
            # V features: Vesta-proprietary engineered signals
            # Actual columns selected dynamically in adapter (after >90% missing filter)
            "V_features_placeholder",
        ],
    },

}

# ---------------------------------------------------------------------------
# Common roles: present across ALL three datasets
# Used as the basis for the cross-dataset common-core experiment
# ---------------------------------------------------------------------------

COMMON_ROLES = ["temporal", "monetary", "transaction"]

# ---------------------------------------------------------------------------
# Specific roles per dataset
# Used to quantify incremental value of richer data sources
# ---------------------------------------------------------------------------

SPECIFIC_ROLES = {
    "paysim": ["stability"],
    "baf":    ["identity", "velocity", "stability", "device", "entity"],
    "ieee":   ["identity", "velocity", "stability", "device", "relational"],
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_common_features(dataset: str) -> list:
    """Return only features mapped to common roles for a given dataset."""
    roles = FEATURE_ROLES.get(dataset, {})
    return [f for role in COMMON_ROLES for f in roles.get(role, [])]


def get_all_features(dataset: str) -> list:
    """Return all features for a given dataset."""
    roles = FEATURE_ROLES.get(dataset, {})
    return [f for role_features in roles.values() for f in role_features]


def get_features_by_role(dataset: str, role: str) -> list:
    """Return features for a specific semantic role in a dataset."""
    return FEATURE_ROLES.get(dataset, {}).get(role, [])


def summarise():
    """Print a readable summary of the feature role mapping table."""
    all_roles = sorted({role for roles in FEATURE_ROLES.values() for role in roles})
    datasets  = ["paysim", "baf", "ieee"]

    print(f"\n{'Role':<16}" + "".join(f"{ds.upper():>18}" for ds in datasets))
    print("-" * (16 + 18 * len(datasets)))

    for role in all_roles:
        row = f"{role:<16}"
        for ds in datasets:
            feats = FEATURE_ROLES[ds].get(role, [])
            cell  = f"✓ ({len(feats)})" if feats else "❌"
            row  += f"{cell:>18}"
        print(row)

    print(f"\nCommon roles     : {COMMON_ROLES}")
    for ds in datasets:
        print(f"{ds:<8} specific : {SPECIFIC_ROLES[ds]}")


if __name__ == "__main__":
    summarise()
