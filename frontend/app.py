"""
Fraud Detection POC — Streamlit Frontend
==========================================
Real-time fraud detection simulation across three datasets:
  - PaySim  : synthetic payment transaction fraud
  - BAF     : bank account application fraud
  - IEEE    : online e-commerce transaction fraud

The app simulates a live transaction stream by replaying
pre-sampled demo data, scoring each transaction via the
FastAPI backend, and displaying real-time alerts.
"""

import time
import requests
import pandas as pd
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8000"

DATASET_CONFIG = {
    "BAF — Bank Account Fraud": {
        "key"       : "baf",
        "data_file" : "baf_demo_data.csv",
        "label_col" : "fraud_bool",
        "desc"      : "Bank account application fraud detection",
        "color"     : "#FF6B6B",
        "new"       : False,
    },
    "PaySim — Payment Fraud": {
        "key"       : "paysim",
        "data_file" : "paysim_demo_data.csv",
        "label_col" : "isFraud",
        "desc"      : "Synthetic mobile money transfer fraud",
        "color"     : "#4ECDC4",
        "new"       : False,
    },
    "FinBank — New Client Dataset ✨": {
        "key"       : "finbank",
        "data_file" : "finbank_demo_data.csv",
        "label_col" : "fraud_label",
        "desc"      : "Unseen dataset — auto-routed via Feature Role Mapping → BAF model",
        "color"     : "#A855F7",
        "new"       : True,
    },
}

DECISION_COLORS = {
    "block"  : "#FF4444",
    "review" : "#FFA500",
    "pass"   : "#44BB44",
}

DECISION_ICONS = {
    "block"  : "🔴",
    "review" : "🟡",
    "pass"   : "🟢",
}

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title = "Fraud Detection POC",
    page_icon  = "🛡️",
    layout     = "wide",
)

st.markdown("""
<style>
    .metric-card {
        background: #1E1E2E;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #333;
    }
    .alert-block  { background: #2D1B1B; border-left: 4px solid #FF4444; padding: 10px; border-radius: 5px; margin: 5px 0; }
    .alert-review { background: #2D2A1B; border-left: 4px solid #FFA500; padding: 10px; border-radius: 5px; margin: 5px 0; }
    .alert-pass   { background: #1B2D1B; border-left: 4px solid #44BB44; padding: 10px; border-radius: 5px; margin: 5px 0; }
    .stProgress > div > div { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "running"       not in st.session_state: st.session_state.running       = False
if "results"       not in st.session_state: st.session_state.results       = []
if "total"         not in st.session_state: st.session_state.total         = 0
if "blocked"       not in st.session_state: st.session_state.blocked       = 0
if "reviewed"      not in st.session_state: st.session_state.reviewed      = 0
if "passed"        not in st.session_state: st.session_state.passed        = 0
if "true_frauds"   not in st.session_state: st.session_state.true_frauds   = 0
if "caught_frauds" not in st.session_state: st.session_state.caught_frauds = 0

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def reset_state():
    st.session_state.results       = []
    st.session_state.total         = 0
    st.session_state.blocked       = 0
    st.session_state.reviewed      = 0
    st.session_state.passed        = 0
    st.session_state.true_frauds   = 0
    st.session_state.caught_frauds = 0


def call_api(dataset_key: str, features: dict) -> dict:
    """Send a transaction to the fraud detection API."""
    try:
        r = requests.post(
            f"{API_BASE}/predict/{dataset_key}",
            json    = {"features": features},
            timeout = 5,
        )
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def row_to_features(row: pd.Series, label_col: str) -> dict:
    """Convert a DataFrame row to API-compatible feature dict."""
    d = row.to_dict()
    d.pop(label_col, None)
    # Convert numpy types to Python native for JSON serialisation
    return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in d.items()}


def format_prob_bar(prob: float) -> str:
    """Return a coloured probability bar string."""
    filled = int(prob * 20)
    bar    = "█" * filled + "░" * (20 - filled)
    color  = "red" if prob > 0.7 else "orange" if prob > 0.4 else "green"
    return f":{color}[{bar}] **{prob:.1%}**"

# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🛡️ Fraud Detection")
    st.markdown("**Multi-Dataset POC**")
    st.divider()

    dataset_name = st.selectbox("Dataset", list(DATASET_CONFIG.keys()))
    cfg          = DATASET_CONFIG[dataset_name]

    speed = st.slider("Transaction speed (seconds)", 0.5, 3.0, 1.5, 0.5)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶ Start", use_container_width=True, type="primary")
    with col2:
        stop_btn  = st.button("⏹ Stop",  use_container_width=True)

    if start_btn:
        reset_state()
        st.session_state.running = True
    if stop_btn:
        st.session_state.running = False

    st.divider()
    st.caption(f"API: `{API_BASE}`")
    st.caption(f"Model: {cfg['key'].upper()}-LightGBM")

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

st.title("🛡️ Real-Time Fraud Detection")
st.markdown(f"**{dataset_name}** — {cfg['desc']}")

# Show feature role mapping explanation for FinBank demo
if cfg.get("new"):
    st.info(
        "✨ **New Dataset Demo** — FinBank uses completely different column names "
        "than any training dataset. The pipeline automatically maps them to shared "
        "semantic roles (monetary, velocity, identity...) and routes to the BAF model. "
        "**No retraining required.**"
    )
    with st.expander("🗺️ See Feature Role Mapping (FinBank → BAF)"):
        mapping_df = pd.DataFrame([
            {"Role": "monetary",  "FinBank column": "annual_income",        "→ BAF column": "income"},
            {"Role": "monetary",  "FinBank column": "credit_limit_requested","→ BAF column": "proposed_credit_limit"},
            {"Role": "temporal",  "FinBank column": "application_month",    "→ BAF column": "month"},
            {"Role": "transaction","FinBank column": "channel_type",         "→ BAF column": "payment_type"},
            {"Role": "identity",  "FinBank column": "overseas_ip",          "→ BAF column": "foreign_request"},
            {"Role": "identity",  "FinBank column": "home_phone_verified",  "→ BAF column": "phone_home_valid"},
            {"Role": "velocity",  "FinBank column": "app_count_1h",         "→ BAF column": "velocity_6h"},
            {"Role": "velocity",  "FinBank column": "app_count_24h",        "→ BAF column": "velocity_24h"},
            {"Role": "stability", "FinBank column": "residence_months",     "→ BAF column": "prev_address_months_count"},
            {"Role": "stability", "FinBank column": "banking_history_months","→ BAF column": "bank_months_count"},
            {"Role": "device",    "FinBank column": "applicant_device",     "→ BAF column": "device_os"},
            {"Role": "entity",    "FinBank column": "risk_score",           "→ BAF column": "credit_risk_score"},
        ])
        st.dataframe(mapping_df, use_container_width=True, hide_index=True)

st.divider()

# KPI metrics row
m1, m2, m3, m4, m5 = st.columns(5)
kpi_total    = m1.metric("Total Processed", st.session_state.total)
kpi_blocked  = m2.metric("🔴 Blocked",      st.session_state.blocked)
kpi_reviewed = m3.metric("🟡 For Review",   st.session_state.reviewed)
kpi_passed   = m4.metric("🟢 Passed",       st.session_state.passed)
kpi_recall   = m5.metric(
    "Fraud Caught",
    f"{st.session_state.caught_frauds}/{st.session_state.true_frauds}"
    if st.session_state.true_frauds > 0 else "—"
)

st.divider()

# Two-column layout: transaction stream + alert log
col_stream, col_alerts = st.columns([3, 2])

with col_stream:
    st.subheader("📡 Live Transaction Stream")
    stream_placeholder = st.empty()

with col_alerts:
    st.subheader("🚨 Alert Log")
    alert_placeholder = st.empty()

# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

if st.session_state.running:
    # Load demo data
    data_path = f"{__file__.replace('app.py', '')}{cfg['data_file']}"
    try:
        demo_df = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"Demo data not found: {data_path}")
        st.stop()

    label_col = cfg["label_col"]

    for idx, row in demo_df.iterrows():
        if not st.session_state.running:
            break

        features   = row_to_features(row, label_col)
        true_label = int(row.get(label_col, 0))
        result     = call_api(cfg["key"], features)

        if result is None:
            st.error("Cannot reach API. Make sure `uvicorn` is running on port 8000.")
            st.session_state.running = False
            break

        # Update counters
        st.session_state.total += 1
        decision = result["decision"]
        if decision == "block":   st.session_state.blocked  += 1
        elif decision == "review": st.session_state.reviewed += 1
        else:                      st.session_state.passed   += 1

        if true_label == 1:
            st.session_state.true_frauds += 1
            if decision in ("block", "review"):
                st.session_state.caught_frauds += 1

        # Store result
        st.session_state.results.insert(0, {
            "idx"       : idx,
            "decision"  : decision,
            "prob"      : result["fraud_probability"],
            "risk"      : result["risk_level"],
            "factors"   : result["risk_factors"],
            "true_label": true_label,
        })

        # --- Update stream panel ---
        with stream_placeholder.container():
            # Show last 8 transactions
            for r in st.session_state.results[:8]:
                icon  = DECISION_ICONS[r["decision"]]
                color = DECISION_COLORS[r["decision"]]
                label = "⚠ FRAUD" if r["true_label"] == 1 else "✓ legit"

                st.markdown(
                    f"{icon} **Txn #{r['idx']}** &nbsp;|&nbsp; "
                    f"Score: `{r['prob']:.3f}` &nbsp;|&nbsp; "
                    f"**{r['decision'].upper()}** &nbsp;|&nbsp; `{label}`"
                )
                if r["factors"]:
                    for f in r["factors"]:
                        st.caption(f"  → {f}")
                st.markdown("---")

        # --- Update alert log (block + review only) ---
        alerts = [r for r in st.session_state.results if r["decision"] in ("block", "review")]
        with alert_placeholder.container():
            if not alerts:
                st.info("No alerts yet.")
            for r in alerts[:10]:
                css_class = f"alert-{r['decision']}"
                factors_html = "".join(f"<li>{f}</li>" for f in r["factors"])
                st.markdown(f"""
<div class="{css_class}">
  <b>{DECISION_ICONS[r['decision']]} Txn #{r['idx']} — {r['decision'].upper()}</b>
  &nbsp; Score: <b>{r['prob']:.3f}</b><br>
  <ul style="margin:4px 0 0 0; font-size:0.85em">{factors_html}</ul>
</div>
""", unsafe_allow_html=True)

        # Update KPI metrics
        m1.metric("Total Processed", st.session_state.total)
        m2.metric("🔴 Blocked",      st.session_state.blocked)
        m3.metric("🟡 For Review",   st.session_state.reviewed)
        m4.metric("🟢 Passed",       st.session_state.passed)
        m5.metric(
            "Fraud Caught",
            f"{st.session_state.caught_frauds}/{st.session_state.true_frauds}"
            if st.session_state.true_frauds > 0 else "—"
        )

        time.sleep(speed)

    st.session_state.running = False
    st.success(f"✅ Simulation complete — {st.session_state.total} transactions processed.")

else:
    # Idle state — show instructions
    with stream_placeholder.container():
        st.info("👈 Select a dataset and press **▶ Start** to begin the simulation.")
    with alert_placeholder.container():
        st.info("Alerts will appear here when fraud is detected.")

    # Show previous results if any
    if st.session_state.results:
        st.divider()
        st.subheader("📊 Last Run Summary")
        total   = st.session_state.total
        blocked = st.session_state.blocked
        reviewed= st.session_state.reviewed

        c1, c2, c3 = st.columns(3)
        c1.metric("Block rate",  f"{blocked/total:.1%}"  if total else "—")
        c2.metric("Review rate", f"{reviewed/total:.1%}" if total else "—")
        c3.metric("Fraud recall",
                  f"{st.session_state.caught_frauds}/{st.session_state.true_frauds}"
                  if st.session_state.true_frauds else "—")
