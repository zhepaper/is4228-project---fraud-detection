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
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import sys
import random
import numpy as np
import pandas as pd
import joblib
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Any, Optional

class PredictRequest(BaseModel):
    features: dict[str, Any]

class ExplainRequest(BaseModel):
    fraud_probability: float
    risk_factors: list[str]
    decision: str
    dataset: str
    features: dict[str, Any]

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.core.decision import make_decision, get_risk_factors
from backend.adapters import paysim_adapter, baf_adapter, ieee_adapter, finbank_adapter
from backend.core.explain import generate_explanation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Fraud Detection API",
    description="Multi-dataset fraud detection pipeline: PaySim | BAF | IEEE-CIS",
    version="1.0.0",
)

# Allow the HTML frontend to connect from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend folder as static files
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'frontend')
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

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
    is_dynamic        : bool = False
    adjustment_reason : str = ""
    explanation      : str = ""
    explanation_method : str = "template"

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

    # Decision using Dynamic Engine
    # Note: PaySim optimal F1 threshold is very high (~0.98), 
    # but we use 0.5 as the baseline for the multiplier engine
    base_t = artifacts.get("threshold", 0.5)
    
    # Get all SHAP values
    explainer = store["explainer"]
    shap_vals = _get_shap_values(explainer, X, feature_cols)
    
    # Generate risk factors from SHAP (or fallback)
    risk_factors = get_risk_factors(feature_cols, list(shap_vals.values()))

    # Final tiered decision
    result = make_decision(
        fraud_probability = prob,
        threshold         = base_t,
        features          = features,
        dataset           = "paysim",
        risk_factors      = risk_factors,
        model             = "XGBoost",
    )

    return PredictResponse(
        fraud_probability = result.fraud_probability,
        decision          = result.decision,
        risk_level        = result.risk_level,
        risk_factors      = result.risk_factors,
        dataset           = result.dataset,
        model             = result.model,
        shap_explanations = shap_vals,
        is_dynamic        = result.is_dynamic,
        adjustment_reason = result.adjustment_reason,
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

    # Decision using Dynamic Engine
    base_t = artifacts.get("threshold", 0.5)

    # Get average SHAP values across all 5 folds
    explainers = store["explainers"]
    shap_vals  = _get_shap_values(explainers, X, feature_cols)
    
    # Generate risk factors from SHAP (or fallback)
    risk_factors = get_risk_factors(feature_cols, list(shap_vals.values()))

    # Final tiered decision
    result = make_decision(
        fraud_probability = prob,
        threshold         = base_t,
        features          = features,
        dataset           = "baf",
        risk_factors      = risk_factors,
        model             = "LightGBM (5-fold ensemble)",
    )

    return PredictResponse(
        fraud_probability = result.fraud_probability,
        decision          = result.decision,
        risk_level        = result.risk_level,
        risk_factors      = result.risk_factors,
        dataset           = result.dataset,
        model             = result.model,
        shap_explanations = shap_vals,
        is_dynamic        = result.is_dynamic,
        adjustment_reason = result.adjustment_reason,
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

    # Decision using Dynamic Engine
    base_t = artifacts.get("threshold", 0.5)

    # Get average SHAP values across all 5 folds
    explainers = store["explainers"]
    shap_vals  = _get_shap_values(explainers, X, feature_cols)
    
    # Generate risk factors from SHAP (or fallback)
    risk_factors = get_risk_factors(feature_cols, list(shap_vals.values()))

    # Final tiered decision
    result = make_decision(
        fraud_probability = prob,
        threshold         = base_t,
        features          = features,
        dataset           = "ieee",
        risk_factors      = risk_factors,
        model             = "LightGBM (5-fold ensemble)",
    )

    return PredictResponse(
        fraud_probability = result.fraud_probability,
        decision          = result.decision,
        risk_level        = result.risk_level,
        risk_factors      = result.risk_factors,
        dataset           = result.dataset,
        model             = result.model,
        shap_explanations = shap_vals,
        is_dynamic        = result.is_dynamic,
        adjustment_reason = result.adjustment_reason,
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
        }
        for ds in MODELS
    }


@app.get("/")
def serve_frontend():
    """Serve the HTML frontend at root."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend not found. Place index.html in /frontend/"}


@app.post("/scan/{dataset_type}")
def scan_batch(dataset_type: str, limit: int = 45):
    """
    Scan the demo CSV for a dataset, score every row through the API pipeline,
    and return results ranked by fraud probability (highest first).
    
    This powers the HTML dashboard's transaction table.
    """
    # Map dataset to CSV and label column
    csv_map = {
        "paysim":  {"file": "paysim_demo_data.csv",  "label": "isFraud"},
        "baf":     {"file": "baf_demo_data.csv",     "label": "fraud_bool"},
        "finbank": {"file": "finbank_demo_data.csv", "label": "fraud_label"},
    }

    effective_type = "baf" if dataset_type == "finbank" else dataset_type
    if effective_type not in MODELS:
        raise HTTPException(status_code=404, detail=f"Unknown dataset '{dataset_type}'")

    cfg = csv_map.get(dataset_type)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"No demo CSV for '{dataset_type}'")

    csv_path = os.path.join(FRONTEND_DIR, cfg["file"])
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"CSV not found: {cfg['file']}")

    demo_df = pd.read_csv(csv_path)
    label_col = cfg["label"]
    results = []

    for idx, row in demo_df.head(limit).iterrows():
        features = row.to_dict()
        true_label = int(features.pop(label_col, 0))

        # Convert numpy types to native Python for JSON
        features = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in features.items()
        }

        try:
            if dataset_type == "finbank":
                renamed = finbank_adapter.preprocess(pd.DataFrame([features])).iloc[0].to_dict()
                pred = _predict_baf(renamed)
                pred.dataset = "finbank (routed → BAF)"
            elif dataset_type == "paysim":
                pred = _predict_paysim(features)
            elif dataset_type == "baf":
                pred = _predict_baf(features)
            else:
                continue

            # Default template explanation for lightning fast scan rendering
            expl = generate_explanation(
                fraud_probability=pred.fraud_probability,
                risk_factors=pred.risk_factors,
                decision=pred.decision,
                dataset=effective_type,
                feature_values=features,
                use_llm=False,
            )

            results.append({
                "id":             f"TXN-{idx+1:04d}",
                "index":          idx,
                "dataset":        pred.dataset,
                "decision":       pred.decision,
                "risk_level":     pred.risk_level,
                "fraud_probability": pred.fraud_probability,
                "risk_factors":   pred.risk_factors,
                "model":          pred.model,
                "true_label":     true_label,
                "is_dynamic":     pred.is_dynamic,
                "adjustment_reason": pred.adjustment_reason,
                "explanation_method": expl["method"],
                "features":       features,
            })
        except Exception as e:
            results.append({
                "id":    f"TXN-{idx+1:04d}",
                "index": idx,
                "error": str(e),
            })

    # Sort by fraud probability descending
    results.sort(key=lambda r: r.get("fraud_probability", 0), reverse=True)
    return {"dataset": dataset_type, "total": len(results), "transactions": results}


# Global counters for stream transactions
_stream_counters = {}


@app.post("/stream/reset/{dataset_type}")
def reset_stream(dataset_type: str):
    """Reset the stream counter for a specific dataset."""
    _stream_counters[dataset_type] = 0
    return {"status": "reset", "dataset": dataset_type, "counter": 0}


@app.get("/stream/next/{dataset_type}")
def stream_next(dataset_type: str):
    """
    Return ONE sequential transaction from the demo CSV, scored through the ML pipeline.
    Simulates a real-time transaction arriving at a bank by maintaining chronological order.
    """
    csv_map = {
        "paysim":  {"file": "paysim_demo_data.csv",  "label": "isFraud"},
        "baf":     {"file": "baf_demo_data.csv",     "label": "fraud_bool"},
        "finbank": {"file": "finbank_demo_data.csv", "label": "fraud_label"},
    }

    effective_type = "baf" if dataset_type == "finbank" else dataset_type
    if effective_type not in MODELS:
        raise HTTPException(status_code=404, detail=f"Unknown dataset '{dataset_type}'")

    cfg = csv_map.get(dataset_type)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"No demo CSV for '{dataset_type}'")

    csv_path = os.path.join(FRONTEND_DIR, cfg["file"])
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"CSV not found: {cfg['file']}")

    demo_df = pd.read_csv(csv_path)
    label_col = cfg["label"]

    # Ensure dataset is in stream counters
    if dataset_type not in _stream_counters:
        _stream_counters[dataset_type] = 0

    # Pick a sequential row instead of random to simulate chronological live transactions properly
    idx = _stream_counters[dataset_type] % len(demo_df)
    _stream_counters[dataset_type] += 1

    row = demo_df.iloc[idx]
    features = row.to_dict()
    true_label = int(features.pop(label_col, 0))

    # Convert numpy types
    features = {
        k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
        for k, v in features.items()
    }

    txn_id = f"TXN-{_stream_counters[dataset_type]:05d}"

    try:
        if dataset_type == "finbank":
            renamed = finbank_adapter.preprocess(pd.DataFrame([features])).iloc[0].to_dict()
            pred = _predict_baf(renamed)
            pred.dataset = "finbank (routed → BAF)"
        elif dataset_type == "paysim":
            pred = _predict_paysim(features)
        elif dataset_type == "baf":
            pred = _predict_baf(features)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported dataset type")

        # Generate explanation via LLM/template
        expl = generate_explanation(
            fraud_probability=pred.fraud_probability,
            risk_factors=pred.risk_factors,
            decision=pred.decision,
            dataset=effective_type,
            feature_values=features,
            use_llm=False
        )

        return {
            "id":                txn_id,
            "index":             idx,
            "dataset":           pred.dataset,
            "decision":          pred.decision,
            "risk_level":        pred.risk_level,
            "fraud_probability": pred.fraud_probability,
            "risk_factors":      pred.risk_factors,
            "model":             pred.model,
            "true_label":        true_label,
            "is_dynamic":        pred.is_dynamic,
            "adjustment_reason": pred.adjustment_reason,
            "explanation_method": expl["method"],
            "features":          features,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain_transaction")
def explain_transaction(req: ExplainRequest):
    """
    On-demand API hook to generate an LLM explanation for a specific transaction risk profile.
    """
    try:
        expl = generate_explanation(
            fraud_probability=req.fraud_probability,
            risk_factors=req.risk_factors,
            decision=req.decision,
            dataset=req.dataset,
            feature_values=req.features,
            use_llm=True  # Always force LLM for on-demand fetch
        )
        return expl
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        elif dataset_type == "paysim":
            result = _predict_paysim(request.features)
        elif dataset_type == "baf":
            result = _predict_baf(request.features)
        elif dataset_type == "ieee":
            result = _predict_ieee(request.features)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported dataset type '{dataset_type}'")
        
        #Generate natural language explanation using LLM
        expl = generate_explanation(
            fraud_probability=result.fraud_probability,
            risk_factors=result.risk_factors,
            decision=result.decision,
            dataset=effective_type,
            feature_values=request.features,
        )

        # Attach explanation to response
        result.explanation = expl["explanation"]
        result.explanation_method = expl["method"]
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
