"""
components/model_evaluation/plots.py
=====================================
Visualisation helpers for model evaluation: ROC curve, Precision-Recall
curve, confusion matrix heatmap, calibration curve, and feature importance
(when the underlying model exposes it).

All plots use a non-interactive Agg backend and are saved as PNG files
to config.plots_dir.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

from credit_card_fraud_detection.utils.logging_setup import logger

sns.set_theme(style="whitegrid")
_DPI = 150


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Plot saved -> {path}")


def plot_roc_curve(y_true, y_prob, model_name: str, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax, name=model_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend()
    path = out_dir / f"roc_curve_{model_name}.png"
    _save(fig, path)
    return path


def plot_pr_curve(y_true, y_prob, model_name: str, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax, name=model_name)
    baseline = float(np.mean(y_true))
    ax.axhline(baseline, linestyle="--", color="gray", label=f"Random (={baseline:.3f})")
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    ax.legend()
    path = out_dir / f"pr_curve_{model_name}.png"
    _save(fig, path)
    return path


def plot_confusion_matrix(y_true, y_pred, model_name: str, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"]).plot(
        ax=ax, cmap="Blues", values_format=",d"
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    path = out_dir / f"confusion_matrix_{model_name}.png"
    _save(fig, path)
    return path


def plot_calibration_curve(y_true, y_prob, model_name: str, out_dir: Path, n_bins: int = 10) -> Path:
    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(mean_pred, fraction_pos, marker="o", label=model_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration Curve — {model_name}")
    ax.legend()
    path = out_dir / f"calibration_{model_name}.png"
    _save(fig, path)
    return path


def plot_model_comparison(comparison_table: list, metric: str, out_dir: Path) -> Path:
    """Bar chart comparing all models on a single metric (e.g. 'auprc')."""
    names  = [r["model"] for r in comparison_table]
    values = [r.get(metric, 0) for r in comparison_table]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=sns.color_palette("Set2", len(names)))
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Model Comparison — {metric.upper()}")
    ax.tick_params(axis="x", rotation=30)
    path = out_dir / f"model_comparison_{metric}.png"
    _save(fig, path)
    return path


import shap

def plot_feature_importance(model, X_test: pd.DataFrame, model_name: str, out_dir: Path, top_n: int = 20) -> Path | None:
    """
    Plots model-agnostic feature importances using SHAP values.
    Uses a small background sample to ensure performance.
    """
    try:
        # Sample background dataset for performance
        X_sample = shap.sample(X_test, min(200, len(X_test)))
        
        # Initialize explainer (handles tree, linear, and neural net models)
        explainer = shap.Explainer(model.predict, X_sample)
        shap_values = explainer(X_sample)

        fig, ax = plt.subplots(figsize=(8, max(4, X_test.shape[1] * 0.3)))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=top_n)
        
        plt.title(f"SHAP Feature Importances — {model_name}")
        path = out_dir / f"feature_importance_{model_name}.png"
        
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Plot saved -> {path}")
        return path
        
    except Exception as exc:
        logger.warning(f"  '{model_name}' could not generate SHAP feature importances: {exc}")
        return None