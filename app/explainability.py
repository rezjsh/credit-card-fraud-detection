"""
app/explainability.py
======================
SHAP explainability helpers for the fraud-detection Streamlit app.

Uses a model-agnostic `shap.Explainer` around `model.predict_proba` so it
works uniformly across every model type in ModelFactory (linear, tree-based,
SVM, MLP, ...), mirroring the approach already used in
`components/model_evaluation/plots.py::plot_feature_importance`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import shap


def build_explainer(predict_fn, background: pd.DataFrame, max_background: int = 200) -> shap.Explainer:
    """
    Build a SHAP explainer around a scalar-output prediction function.

    Args:
        predict_fn:   callable(X: DataFrame) -> np.ndarray of P(fraud)
        background:   representative sample of feature rows (e.g. from val/test)
        max_background: cap the background sample size for speed
    """
    if len(background) > max_background:
        background = background.sample(max_background, random_state=42)
    return shap.Explainer(predict_fn, background)


def explain_single(explainer: shap.Explainer, row: pd.DataFrame):
    """Return SHAP values for a single-row DataFrame."""
    return explainer(row)


def top_contributors(shap_values, feature_names: list[str], top_n: int = 10) -> pd.DataFrame:
    """
    Turn a single-row SHAP explanation into a ranked DataFrame of the
    top_n features by absolute contribution to the prediction.
    """
    values = np.asarray(shap_values.values).reshape(-1)
    base = float(np.asarray(shap_values.base_values).reshape(-1)[0])
    df = pd.DataFrame({"feature": feature_names, "shap_value": values})
    df["abs_shap"] = df["shap_value"].abs()
    df = df.sort_values("abs_shap", ascending=False).head(top_n)
    df.attrs["base_value"] = base
    return df.drop(columns="abs_shap")


def waterfall_data(shap_row: pd.DataFrame, base_value: float) -> pd.DataFrame:
    """Prepare cumulative data for a simple bar/waterfall style chart."""
    df = shap_row.copy().sort_values("shap_value")
    df["cumulative"] = base_value + df["shap_value"].cumsum()
    return df