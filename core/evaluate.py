"""
Unified Evaluation Module
==========================
Shared evaluation functions used across all three dataset pipelines.

All three datasets use the same metrics:
  - AUC-ROC  : ranking ability (less sensitive to class imbalance)
  - AUC-PR   : precision-recall tradeoff (primary metric for fraud detection)
  - F1       : harmonic mean at optimal threshold
  - Precision / Recall at optimal threshold

The optimal threshold is found by maximising F1 on the precision-recall curve,
rather than using a fixed 0.5 cutoff — important for imbalanced datasets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = None,
    dataset_name: str = "",
) -> dict:
    """
    Compute full evaluation metrics for a set of fraud predictions.

    Args:
        y_true       : ground truth binary labels (0/1)
        y_pred_proba : predicted fraud probabilities [0, 1]
        threshold    : decision threshold — if None, finds optimal F1 threshold
        dataset_name : label for display purposes

    Returns:
        dict with all metrics and the threshold used
    """
    # Find optimal threshold by maximising F1 if not provided
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_arr = 2 * precision_arr * recall_arr / (precision_arr + recall_arr + 1e-9)

    if threshold is None:
        best_idx  = np.argmax(f1_arr)
        threshold = float(thresholds[best_idx])

    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        "dataset"   : dataset_name,
        "n_samples" : int(len(y_true)),
        "fraud_rate": float(y_true.mean()),
        "auc_roc"   : float(roc_auc_score(y_true, y_pred_proba)),
        "auc_pr"    : float(average_precision_score(y_true, y_pred_proba)),
        "f1"        : float(f1_score(y_true, y_pred)),
        "precision" : float(np.sum((y_pred == 1) & (y_true == 1)) / (np.sum(y_pred == 1) + 1e-9)),
        "recall"    : float(np.sum((y_pred == 1) & (y_true == 1)) / (np.sum(y_true == 1) + 1e-9)),
        "threshold" : threshold,
    }

    return metrics


def find_best_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Return the threshold that maximises F1 score."""
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_arr   = 2 * precision_arr * recall_arr / (precision_arr + recall_arr + 1e-9)
    best_idx = np.argmax(f1_arr)
    return float(thresholds[best_idx])


# ---------------------------------------------------------------------------
# Cross-dataset comparison
# ---------------------------------------------------------------------------

def compare_datasets(results: dict) -> pd.DataFrame:
    """
    Build a comparison table from multiple evaluate() results.

    Args:
        results : dict of {dataset_name: metrics_dict}

    Returns:
        DataFrame with one row per dataset
    """
    rows = []
    for name, m in results.items():
        rows.append({
            "Dataset"   : name,
            "N"         : f"{m['n_samples']:,}",
            "Fraud Rate": f"{m['fraud_rate']:.2%}",
            "AUC-ROC"   : round(m["auc_roc"], 4),
            "AUC-PR"    : round(m["auc_pr"],  4),
            "F1"        : round(m["f1"],       4),
            "Precision" : round(m["precision"],4),
            "Recall"    : round(m["recall"],   4),
            "Threshold" : round(m["threshold"],4),
        })
    return pd.DataFrame(rows).set_index("Dataset")


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metrics: dict,
    title: str = "",
    ax_roc=None,
    ax_pr=None,
):
    """
    Plot ROC and Precision-Recall curves side by side.
    Can plot into existing axes (for multi-dataset subplots) or create new figure.
    """
    if ax_roc is None or ax_pr is None:
        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 4))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    ax_roc.plot(fpr, tpr, lw=2, label=f"AUC-ROC = {metrics['auc_roc']:.4f}")
    ax_roc.fill_between(fpr, tpr, alpha=0.1)
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"ROC Curve — {title}")
    ax_roc.legend()

    # Precision-Recall curve
    precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_pred_proba)
    ax_pr.plot(recall_arr, precision_arr, lw=2,
               color='darkorange', label=f"AUC-PR = {metrics['auc_pr']:.4f}")
    # Mark optimal threshold point
    ax_pr.scatter(metrics["recall"], metrics["precision"],
                  color='red', s=80, zorder=5,
                  label=f"Best F1 = {metrics['f1']:.4f}")
    ax_pr.axvline(metrics["recall"],    color='red', linestyle='--', alpha=0.3)
    ax_pr.axhline(metrics["precision"], color='red', linestyle='--', alpha=0.3)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"Precision-Recall Curve — {title}")
    ax_pr.legend()

    plt.tight_layout()


def plot_score_distribution(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float,
    title: str = "",
):
    """
    Plot the distribution of predicted fraud scores for fraud vs legitimate.
    Shows how well-separated the two classes are.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(y_pred_proba[y_true == 0], bins=100, density=True,
            alpha=0.6, color='steelblue',  label='Legitimate')
    ax.hist(y_pred_proba[y_true == 1], bins=100, density=True,
            alpha=0.6, color='tomato',     label='Fraud')
    ax.axvline(threshold, color='black', linestyle='--',
               label=f'Threshold = {threshold:.3f}')

    ax.set_xlabel("Predicted Fraud Probability")
    ax.set_ylabel("Density")
    ax.set_title(f"Score Distribution — {title}")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_variant_comparison(variant_results: dict, title: str = "Variant Robustness"):
    """
    Bar chart comparing AUC-ROC, AUC-PR, F1 across BAF variants.
    Visualises model performance degradation under distribution shift.
    """
    labels  = list(variant_results.keys())
    aucroc  = [variant_results[k]["auc_roc"] for k in labels]
    aucpr   = [variant_results[k]["auc_pr"]  for k in labels]
    f1_vals = [variant_results[k]["f1"]      for k in labels]

    x     = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 5))
    b1 = ax.bar(x - width, aucroc,  width, label='AUC-ROC', color='steelblue')
    b2 = ax.bar(x,         aucpr,   width, label='AUC-PR',  color='darkorange')
    b3 = ax.bar(x + width, f1_vals, width, label='F1',      color='seagreen')

    for bar in [*b1, *b2, *b3]:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f'{bar.get_height():.3f}',
                ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def print_summary(metrics: dict, model_name: str = "LightGBM", n_folds: int = 5):
    """Print a formatted results summary box."""
    w = 56
    print("═" * w)
    print(f"   {metrics['dataset']} — Pipeline Results")
    print("═" * w)
    print(f"  Model        : {model_name} ({n_folds}-Fold Stratified CV)")
    print(f"  Samples      : {metrics['n_samples']:,}")
    print(f"  Fraud rate   : {metrics['fraud_rate']:.4%}")
    print("─" * w)
    print(f"  AUC-ROC      : {metrics['auc_roc']:.5f}")
    print(f"  AUC-PR       : {metrics['auc_pr']:.5f}")
    print(f"  F1           : {metrics['f1']:.5f}  @ threshold {metrics['threshold']:.4f}")
    print(f"  Precision    : {metrics['precision']:.5f}")
    print(f"  Recall       : {metrics['recall']:.5f}")
    print("═" * w)
