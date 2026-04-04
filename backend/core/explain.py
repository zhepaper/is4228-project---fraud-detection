"""
Explanation Generator
=====================
Generates human-readable explanations for fraud predictions.
Supports:
  - Template-based explanations (offline, fast)
  - LLM-enhanced explanations via OpenAI/compatible API (richer, configurable)
"""

import os
import json
from typing import Optional
from groq import Groq
#from decision import RISK_FACTOR_LABELS, GENERIC_LABEL
from openai import OpenAI

LLM_ENABLED = os.getenv("ENABLE_LLM_EXPLANATIONS", "true").lower() == "true"
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", None)


SYSTEM_PROMPT = """
You are a fraud analysis assistant for a financial security system. 
Your task is to explain why a transaction was scored as potentially fraudulent or legitimate.

Guidelines:
- Be concise (2-4 sentences max).
- Use plain, non-technical language suitable for a fraud analyst or customer support agent.
- Reference the specific risk factors provided, but translate them into business context.
- If the fraud probability is low, explain why the transaction appears legitimate.
- Never reveal model internals, SHAP values, or thresholds.
- If uncertain, acknowledge limitations rather than speculating.

Output format: Return ONLY the explanation text. No JSON, no markdown, no prefixes.
"""


# Custom Plugins: Template-Based Explanation (Offline Fallback)

def generate_template_explanation(
    fraud_probability: float,
    risk_factors: list[str],
    decision: str,
) -> str:
    """
    Generate a concise explanation using predefined templates.
    Fast, offline, and deterministic.
    """
    if decision == "pass" and fraud_probability < 0.3:
        base = "This transaction appears legitimate."
        if not risk_factors:
            return base + " No significant risk indicators were detected."
        return base + f" Minor signals noted: {', '.join(risk_factors[:2])}."

    if decision == "block":
        intro = "This transaction was blocked due to high fraud risk."
    elif decision == "review":
        intro = "This transaction requires manual review because of ambiguous risk signals."
    else:
        intro = "This transaction shows some risk indicators."

    if not risk_factors:
        return intro + " The model detected anomalous patterns not captured by standard features."

    factors_str = "; ".join(risk_factors[:3])  # Top 3 only for brevity
    return f"{intro} Key factors: {factors_str}."



# ---------------------------------------------------------------------------
# LLM-Enhanced Explanation (Optional)
# ---------------------------------------------------------------------------

def _call_llm_api(
    prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    model: str = LLM_MODEL,
    api_key: str = LLM_API_KEY,
    base_url: Optional[str] = LLM_BASE_URL,
    max_tokens: int = 150,
    temperature: float = 0.1,
) -> Optional[str]:
    """
    Call OpenAI-compatible API to generate explanation.
    Returns cleaned response text or None on error.
    """
    if not api_key:
        return None

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=10,
        )
        text = response.choices[0].message.content.strip()
        # Remove any accidental markdown or prefixes
        text = text.replace("```", "").replace("Explanation:", "").strip()
        return text if text else None
    except Exception as e:
        # Log error in production; here we silently fallback
        print(f"[LLM Explanation Error] {e}")
        return None


def generate_llm_explanation(
    fraud_probability: float,
    risk_factors: list[str],
    decision: str,
    dataset: str,
    feature_values: Optional[dict] = None,  # Optional: raw feature values for richer context
) -> Optional[str]:
    """
    Generate explanation via LLM. Returns None if LLM call fails.
    """
    # Build user prompt
    factors_bullet = "\n".join(f"- {f}" for f in risk_factors[:5]) if risk_factors else "- None detected"
    decision_context = {
        "block": "blocked as high-risk fraud",
        "review": "flagged for manual review",
        "pass": "approved as low-risk",
    }.get(decision, "evaluated")

    feature_context = ""
    if feature_values:
        # Include top 3 numeric features by absolute value for context
        numeric = {k: v for k, v in feature_values.items() if isinstance(v, (int, float))}
        top_feats = sorted(numeric.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        if top_feats:
            feature_context = "\n\nTransaction context:\n" + "\n".join(
                f"- {k}: {v:.2f}" if isinstance(v, float) else f"- {k}: {v}"
                for k, v in top_feats
            )

    user_prompt = f"""
        Transaction assessment:
        - Dataset: {dataset}
        - Fraud probability: {fraud_probability:.1%}
        - Decision: {decision_context}
        - Top risk factors:
        {factors_bullet}
        {feature_context}

        Please explain why this transaction was {decision_context}.
        """.strip()

    return _call_llm_api(user_prompt)


# ---------------------------------------------------------------------------
# Unified Explanation Endpoint Logic
# ---------------------------------------------------------------------------

def generate_explanation(
    fraud_probability: float,
    risk_factors: list[str],
    decision: str,
    dataset: str,
    feature_values: Optional[dict] = None,
    use_llm: bool = None,  # None = use config, True/False = override
) -> dict:
    """
    Generate explanation with fallback strategy.
    
    Returns:
        {
            "explanation": str,
            "method": "template" | "llm",
            "confidence": float,  # heuristic: higher for LLM if successful
            "risk_factors_used": list[str]
        }
    """
    use_llm = use_llm if use_llm is not None else LLM_ENABLED
    
    # Try LLM first if enabled
    if use_llm:
        llm_text = generate_llm_explanation(
            fraud_probability, risk_factors, decision, dataset, feature_values
        )
        if llm_text:
            return {
                "explanation": llm_text,
                "method": "llm",
                "confidence": 0.9,  # heuristic
                "risk_factors_used": risk_factors[:5],
            }

    # Fallback to template
    template_text = generate_template_explanation(
        fraud_probability, risk_factors, decision
    )
    return {
        "explanation": template_text,
        "method": "template",
        "confidence": 0.7,  # heuristic
        "risk_factors_used": risk_factors[:3],
    }