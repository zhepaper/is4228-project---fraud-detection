"""
Fraud Detection API
====================
FastAPI application exposing a unified /predict endpoint
for three fraud detection pipelines: PaySim, BAF, and IEEE.

Each dataset has its own preprocessing adapter and trained model(s).
The API loads all models at startup and serves predictions in real time.

Endpoints:
  POST /predict/{dataset_type}   — score a single transaction
  GET  /health                   — check API status
  GET  /datasets                 — list available datasets and their features
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.core.decision import make_decision, get_risk_factors
from backend.adapters import paysim_adapter, baf_adapter, ieee_adapter, finbank_adapter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# Fixed decision thresholds per dataset
# (multiplier approach fails for PaySim where optimal threshold > 0.98)
DECISION_THRESHOLDS = {
    "paysim": {"block": 0.90, "review": 0.60},
    "baf":    {"block": 0.92, "review": 0.70},
    "ieee":   {"block": 0.60, "review": 0.35},
}

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Fraud Detection API",
    description="Multi-dataset fraud detection pipeline: PaySim | BAF | IEEE-CIS",
    version="1.0.0",
)

# Global model store — loaded once at startup, reused for every request
MODELS = {}


@app.on_event("startup")
def load_models():
    """Load all trained models and artifacts into memory at startup."""

    # --- PaySim ---
    paysim_model = joblib.load(os.path.join(MODELS_DIR, "paysim_xgb_fraud_model.joblib"))
    MODELS["paysim"] = {
        "model"     : paysim_model,
        "scaler"    : joblib.load(os.path.join(MODELS_DIR, "paysim_scaler.joblib")),
        "artifacts" : joblib.load(os.path.join(MODELS_DIR, "paysim_artifacts.joblib")),
        "explainer" : shap.TreeExplainer(paysim_model),
    }

    # --- BAF ---
    baf_folds = [
        joblib.load(os.path.join(MODELS_DIR, f"baf_lgbm_fold{i}.joblib"))
        for i in range(1, 6)
    ]
    MODELS["baf"] = {
        "models"    : baf_folds,
        "encoder"   : joblib.load(os.path.join(MODELS_DIR, "baf_encoder.joblib")),
        "artifacts" : joblib.load(os.path.join(MODELS_DIR, "baf_artifacts.joblib")),
        "explainers": [shap.TreeExplainer(m) for m in baf_folds],
    }

    # --- IEEE ---
    ieee_folds = [
        joblib.load(os.path.join(MODELS_DIR, f"ieee_lgbm_fold{i}.joblib"))
        for i in range(1, 6)
    ]
    MODELS["ieee"] = {
        "models"         : ieee_folds,
        "label_encoders" : joblib.load(os.path.join(MODELS_DIR, "ieee_label_encoders.joblib")),
        "drop_cols"      : joblib.load(os.path.join(MODELS_DIR, "ieee_drop_cols.joblib")),
        "artifacts"      : joblib.load(os.path.join(MODELS_DIR, "ieee_artifacts.joblib")),
        "explainers"     : [shap.TreeExplainer(m) for m in ieee_folds],
    }

    print("All models loaded successfully.")
    for ds, content in MODELS.items():
        print(f"  {ds}: {list(content.keys())}")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Raw transaction features as a flat dict."""
    features: dict[str, Any]


class PredictResponse(BaseModel):
    fraud_probability : float
    decision          : str
    risk_level        : str
    risk_factors      : list[str]
    dataset           : str
    model             : str
    shap_explanations : dict[str, float]

# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _get_shap_values(explainers: list | shap.TreeExplainer, X: np.ndarray, feature_names: list) -> dict[str, float]:
    """Calculate and average SHAP values for a given input."""
    if not isinstance(explainers, list):
        explainers = [explainers]
    
    # Calculate SHAP values for each explainer
    #TreeExplainer for XGBoost/LGBM usually returns (N, num_features) or a list of two such arrays (class 0, class 1)
    all_shap = []
    for explainer in explainers:
        s = explainer.shap_values(X)
        if isinstance(s, list):
            # Binary classification usually returns [class 0, class 1]
            s = s[1]
        all_shap.append(s[0]) # Single row prediction
        
    # Average across all explainers (if list)
    mean_shap = np.mean(all_shap, axis=0)
    
    # Map to feature names
    return {name: float(val) for name, val in zip(feature_names, mean_shap)}


def _predict_paysim(features: dict) -> PredictResponse:
    store    = MODELS["paysim"]
    model    = store["model"]
    artifacts = store["artifacts"]
    feature_cols = artifacts["features"]

    df = pd.DataFrame([features])
    df = paysim_adapter.preprocess(df)

    # Ensure all expected features are present
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X    = df[feature_cols].values
    prob = float(model.predict_proba(X)[0, 1])

    # Decision using fixed thresholds
    t = DECISION_THRESHOLDS["paysim"]
    if prob >= t["block"]:
        decision, risk_level = "block", "high"
    elif prob >= t["review"]:
        decision, risk_level = "review", "medium"
    else:
        decision, risk_level = "pass", "low"

    # Risk factors from feature values (SHAP now available)
    risk_factors = _simple_risk_factors_paysim(df)
    
    # Get all SHAP values
    explainer = store["explainer"]
    shap_vals = _get_shap_values(explainer, X, feature_cols)

    return PredictResponse(
        fraud_probability = round(prob, 4),
        decision          = decision,
        risk_level        = risk_level,
        risk_factors      = risk_factors,
        dataset           = "paysim",
        model             = "XGBoost",
        shap_explanations = shap_vals,
    )


def _predict_baf(features: dict) -> PredictResponse:
    store        = MODELS["baf"]
    fold_models  = store["models"]
    encoder      = store["encoder"]
    artifacts    = store["artifacts"]
    feature_cols = artifacts["features"]

    df = pd.DataFrame([features])
    df, _ = baf_adapter.preprocess(df, encoder=encoder, fit=False)

    # Keep only features the model was trained on (some may be engineered)
    available = [col for col in feature_cols if col in df.columns]
    missing   = [col for col in feature_cols if col not in df.columns]
    for col in missing:
        df[col] = 0

    X    = df[feature_cols].values

    # Ensemble: average predictions from all 5 fold models
    prob = float(np.mean([m.predict_proba(X)[0, 1] for m in fold_models]))

    # Decision using fixed thresholds
    t = DECISION_THRESHOLDS["baf"]
    if prob >= t["block"]:
        decision, risk_level = "block", "high"
    elif prob >= t["review"]:
        decision, risk_level = "review", "medium"
    else:
        decision, risk_level = "pass", "low"

    risk_factors = _simple_risk_factors_baf(df)

    # Get average SHAP values across all 5 folds
    explainers = store["explainers"]
    shap_vals  = _get_shap_values(explainers, X, feature_cols)

    return PredictResponse(
        fraud_probability = round(prob, 4),
        decision          = decision,
        risk_level        = risk_level,
        risk_factors      = risk_factors,
        dataset           = "baf",
        model             = "LightGBM (5-fold ensemble)",
        shap_explanations = shap_vals,
    )


def _predict_ieee(features: dict) -> PredictResponse:
    store          = MODELS["ieee"]
    fold_models    = store["models"]
    label_encoders = store["label_encoders"]
    drop_cols      = store["drop_cols"]
    artifacts      = store["artifacts"]
    feature_cols   = artifacts["features"]

    df = pd.DataFrame([features])
    df, _, _ = ieee_adapter.preprocess(
        df,
        label_encoders          = label_encoders,
        drop_cols_from_training = drop_cols,
        fit                     = False,
    )

    # Ensure all expected features are present
    for col in feature_cols:
        if col not in df.columns:
            df[col] = -999   # IEEE uses -999 as missing sentinel

    X    = df[feature_cols].values

    # Ensemble: average predictions from all 5 fold models
    prob = float(np.mean([m.predict_proba(X)[0, 1] for m in fold_models]))

    # Decision using fixed thresholds
    t = DECISION_THRESHOLDS["ieee"]
    if prob >= t["block"]:
        decision, risk_level = "block", "high"
    elif prob >= t["review"]:
        decision, risk_level = "review", "medium"
    else:
        decision, risk_level = "pass", "low"

    risk_factors = _simple_risk_factors_ieee(df)

    # Get average SHAP values across all 5 folds
    explainers = store["explainers"]
    shap_vals  = _get_shap_values(explainers, X, feature_cols)

    return PredictResponse(
        fraud_probability = round(prob, 4),
        decision          = decision,
        risk_level        = risk_level,
        risk_factors      = risk_factors,
        dataset           = "ieee",
        model             = "LightGBM (5-fold ensemble)",
        shap_explanations = shap_vals,
    )


# ---------------------------------------------------------------------------
# Simple rule-based risk factors (fallback without SHAP)
# ---------------------------------------------------------------------------

def _simple_risk_factors_paysim(df: pd.DataFrame) -> list[str]:
    factors = []
    if df.get("balanceDrain", pd.Series([0])).iloc[0] == 1:
        factors.append("Sender account drained to zero after transaction")
    if df.get("errorIsSmallOrig", pd.Series([0])).iloc[0] == 1:
        factors.append("Balance change matches transaction exactly (no discrepancy)")
    if df.get("is_transfer", pd.Series([0])).iloc[0] == 1:
        factors.append("Transaction type is TRANSFER (fraud-prone)")
    if df.get("is_cashout", pd.Series([0])).iloc[0] == 1:
        factors.append("Transaction type is CASH_OUT (fraud-prone)")
    if "amountToBalanceRatio" in df.columns and df["amountToBalanceRatio"].iloc[0] > 0.9:
        factors.append("Transaction amount is nearly equal to full account balance")
    return factors[:3]


def _simple_risk_factors_baf(df: pd.DataFrame) -> list[str]:
    factors = []
    if df.get("prev_address_missing", pd.Series([0])).iloc[0] == 1:
        factors.append("No previous address history on record")
    if df.get("bank_months_missing", pd.Series([0])).iloc[0] == 1:
        factors.append("No banking history on record")
    if df.get("foreign_request", pd.Series([0])).iloc[0] == 1:
        factors.append("Application submitted from a foreign IP address")
    if df.get("phone_not_verified", pd.Series([0])).iloc[0] == 1:
        factors.append("Neither home nor mobile phone has been verified")
    if "velocity_ratio_6h_24h" in df.columns and df["velocity_ratio_6h_24h"].iloc[0] > 2:
        factors.append("Sudden burst of applications in the last 6 hours")
    if df.get("email_is_free", pd.Series([0])).iloc[0] == 1:
        factors.append("Free email provider used")
    return factors[:3]


def _simple_risk_factors_ieee(df: pd.DataFrame) -> list[str]:
    factors = []
    if "P_emaildomain_is_mainstream" in df.columns and df["P_emaildomain_is_mainstream"].iloc[0] == 0:
        factors.append("Payer using non-mainstream email domain")
    if "is_night" in df.columns and df["is_night"].iloc[0] == 1:
        factors.append("Transaction occurred during night hours (00:00-06:00)")
    if "TransactionAmt_decimal" in df.columns and df["TransactionAmt_decimal"].iloc[0] == 0:
        factors.append("Suspicious round transaction amount")
    if "is_weekend" in df.columns and df["is_weekend"].iloc[0] == 1:
        factors.append("Transaction occurred on a weekend")
    return factors[:3]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Check API status and which models are loaded."""
    return {
        "status" : "ok",
        "models" : list(MODELS.keys()),
    }


@app.get("/datasets")
def datasets():
    """List available datasets and their expected input features."""
    return {
        ds: {
            "features"  : MODELS[ds]["artifacts"]["features"],
            "threshold" : float(MODELS[ds]["artifacts"]["threshold"]),
            "decision_thresholds": DECISION_THRESHOLDS[ds],
        }
        for ds in MODELS
    }


@app.post("/predict/{dataset_type}", response_model=PredictResponse)
def predict(dataset_type: str, request: PredictRequest):
    """
    Score a single transaction and return a fraud decision.

    Args:
        dataset_type : one of 'paysim', 'baf', 'ieee'
        request.features : raw transaction fields as key-value pairs

    Returns:
        fraud_probability, decision (block/review/pass),
        risk_level, risk_factors, model used
    """
    # FinBank routes through the BAF pipeline after column renaming
    effective_type = "baf" if dataset_type == "finbank" else dataset_type

    if effective_type not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown dataset '{dataset_type}'. Available: paysim, baf, ieee, finbank"
        )

    try:
        if dataset_type == "finbank":
            # Rename FinBank columns → BAF columns, then score with BAF model
            renamed = finbank_adapter.preprocess(pd.DataFrame([request.features])).iloc[0].to_dict()
            result  = _predict_baf(renamed)
            result.dataset = "finbank (routed → BAF)"
            return result
        elif dataset_type == "paysim":
            return _predict_paysim(request.features)
        elif dataset_type == "baf":
            return _predict_baf(request.features)
        elif dataset_type == "ieee":
            return _predict_ieee(request.features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
