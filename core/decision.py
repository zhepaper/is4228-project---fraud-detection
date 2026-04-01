"""
Decision Engine
===============
Converts a fraud probability score into an actionable decision.

Real-world fraud systems don't just output a probability — they
translate it into a business action with an explanation.

Decision tiers:
  - BLOCK  : high confidence fraud — transaction rejected immediately
  - REVIEW : ambiguous — flagged for manual review queue
  - PASS   : low risk — transaction approved

Risk factors are generated from the top SHAP contributors,
translated into human-readable strings for the frontend.
"""

from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Decision output structure
# ---------------------------------------------------------------------------

@dataclass
class DecisionResult:
    fraud_probability : float
    decision          : str          # "block" | "review" | "pass"
    risk_level        : str          # "high" | "medium" | "low"
    risk_factors      : list         # human-readable explanation strings
    dataset           : str          # which pipeline produced this result
    threshold_used    : float
    model             : str = "LightGBM"

    def to_dict(self) -> dict:
        """Serialise to dict for API response."""
        return {
            "fraud_probability" : round(self.fraud_probability, 4),
            "decision"          : self.decision,
            "risk_level"        : self.risk_level,
            "risk_factors"      : self.risk_factors,
            "dataset"           : self.dataset,
            "threshold_used"    : round(self.threshold_used, 4),
            "model"             : self.model,
        }


# ---------------------------------------------------------------------------
# Core decision logic
# ---------------------------------------------------------------------------

def make_decision(
    fraud_probability : float,
    threshold         : float,
    dataset           : str = "",
    risk_factors      : list = None,
    model             : str = "LightGBM",
) -> DecisionResult:
    """
    Convert a fraud probability into a tiered decision.

    Thresholds are relative to the optimal F1 threshold found during training:
      - BLOCK  : prob >= threshold * 1.2  (high confidence)
      - REVIEW : prob >= threshold * 0.6  (ambiguous zone)
      - PASS   : prob <  threshold * 0.6  (low risk)

    These multipliers can be tuned to balance precision vs recall
    depending on the business cost of false positives vs false negatives.

    Args:
        fraud_probability : model output probability [0, 1]
        threshold         : optimal F1 threshold from training
        dataset           : dataset name for the response
        risk_factors      : list of human-readable risk factor strings
        model             : model name for the response

    Returns:
        DecisionResult with full decision context
    """
    block_threshold  = threshold * 1.2
    review_threshold = threshold * 0.6

    if fraud_probability >= block_threshold:
        decision   = "block"
        risk_level = "high"
    elif fraud_probability >= review_threshold:
        decision   = "review"
        risk_level = "medium"
    else:
        decision   = "pass"
        risk_level = "low"

    return DecisionResult(
        fraud_probability = fraud_probability,
        decision          = decision,
        risk_level        = risk_level,
        risk_factors      = risk_factors or [],
        dataset           = dataset,
        threshold_used    = threshold,
        model             = model,
    )


# ---------------------------------------------------------------------------
# Risk factor generation (SHAP-based)
# ---------------------------------------------------------------------------

# Human-readable descriptions per feature per dataset
# Maps feature name → plain English explanation when it drives fraud score UP
RISK_FACTOR_LABELS = {

    # --- PaySim ---
    "balanceDrain"         : "Sender account drained to zero after transaction",
    "errorIsSmallOrig"     : "Sender balance change matches transaction exactly (no discrepancy)",
    "errorIsSmallDest"     : "Receiver balance change matches transaction exactly",
    "is_transfer"          : "Transaction type is TRANSFER (fraud-prone type)",
    "is_cashout"           : "Transaction type is CASH_OUT (fraud-prone type)",
    "amountToBalanceRatio" : "Transaction amount is large relative to account balance",
    "log_amount"           : "Unusually large transaction amount",

    # --- BAF ---
    "prev_address_missing"          : "No previous address history on record",
    "bank_months_missing"           : "No banking history on record",
    "foreign_request"               : "Application submitted from a foreign IP address",
    "phone_not_verified"            : "Neither home nor mobile phone has been verified",
    "velocity_ratio_6h_24h"         : "Sudden burst of applications in the last 6 hours",
    "velocity_6h"                   : "High number of applications from same context in last 6h",
    "address_instability"           : "Unstable address history",
    "email_is_free"                 : "Free email provider used",
    "has_other_cards"               : "No other credit cards on record",
    "name_email_similarity"         : "Low similarity between name and email address",
    "credit_risk_score"             : "Elevated credit risk score",
    "keep_alive_session"            : "Session keep-alive is inactive",

    # --- IEEE ---
    "TransactionAmt_log"            : "Unusually large transaction amount",
    "TransactionAmt_decimal"        : "Suspicious transaction amount pattern (round number)",
    "P_emaildomain_is_mainstream"   : "Payer using non-mainstream email domain",
    "R_emaildomain_is_mainstream"   : "Receiver using non-mainstream email domain",
    "is_night"                      : "Transaction occurred during night hours",
    "is_weekend"                    : "Transaction occurred on a weekend",
    "card1_addr1"                   : "Unusual card and billing address combination",
}

# Generic fallback when a feature has no label defined
GENERIC_LABEL = "Anomalous pattern detected in {feature}"


def get_risk_factors(
    feature_names  : list,
    shap_values    : list,
    top_n          : int = 5,
) -> list:
    """
    Generate human-readable risk factor strings from SHAP values.

    Takes the top N features by absolute SHAP contribution and
    translates each into a plain English explanation.

    Args:
        feature_names : list of feature name strings
        shap_values   : SHAP values for a single prediction (same length as feature_names)
        top_n         : number of top risk factors to return

    Returns:
        list of human-readable risk factor strings (only positive contributors)
    """
    # Pair feature names with their SHAP values
    pairs = list(zip(feature_names, shap_values))

    # Keep only features that push score UP (positive SHAP = increases fraud probability)
    positive = [(name, val) for name, val in pairs if val > 0]

    # Sort by magnitude (most impactful first)
    positive.sort(key=lambda x: abs(x[1]), reverse=True)

    # Translate to human-readable strings
    risk_factors = []
    for name, _ in positive[:top_n]:
        label = RISK_FACTOR_LABELS.get(
            name,
            GENERIC_LABEL.format(feature=name.replace('_', ' '))
        )
        risk_factors.append(label)

    return risk_factors


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test decision tiers
    threshold = 0.5
    test_cases = [
        ("High fraud",   0.91),
        ("Borderline",   0.38),
        ("Low risk",     0.10),
    ]

    for label, prob in test_cases:
        result = make_decision(
            fraud_probability = prob,
            threshold         = threshold,
            dataset           = "BAF",
            risk_factors      = ["No address history", "Foreign IP request"],
        )
        print(f"{label:15} prob={prob:.2f}  →  {result.decision.upper():8}  ({result.risk_level})")

    print()

    # Test risk factor translation
    features = ["balanceDrain", "foreign_request", "log_amount", "unknown_feature_xyz"]
    shap_vals = [0.42, 0.31, 0.18, -0.05]
    factors = get_risk_factors(features, shap_vals, top_n=3)
    print("Risk factors:")
    for f in factors:
        print(f"  - {f}")
