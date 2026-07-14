"""
app/app.py
==========
Streamlit front-end for the Credit Card Fraud Detection project.

Run with:
    streamlit run app/app.py

Pages:
  1. Overview                     — project + pipeline summary
  2. Predict (single transaction) — manual entry, live probability + SHAP
  3. Predict (batch CSV)          — upload a raw-schema CSV, get scored file back
  4. Explainability               — SHAP global importance over a sample
  5. Model Performance Dashboard  — evaluation_report.json / model_comparison.json / plots
"""

from __future__ import annotations

import json
import pathlib
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st

from credit_card_fraud_detection.config.configuration import ConfigurationManager
from inference import FraudInferencePipeline, ArtifactLoadError
import explainability as expl

if sys.platform == 'win32':
    pathlib.PosixPath = pathlib.WindowsPath
st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="wide")


# ──────────────────────────────────────────────────────────────────────────
# Cached resources
# ──────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model artifacts...")
def load_pipeline() -> FraudInferencePipeline:
    return FraudInferencePipeline()


@st.cache_data
def load_json_safe(path: str) -> dict | list | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


@st.cache_data
def load_parquet_safe(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_parquet(p)


def get_config_manager() -> ConfigurationManager:
    if "config_manager" not in st.session_state:
        st.session_state.config_manager = ConfigurationManager()
    return st.session_state.config_manager


# ──────────────────────────────────────────────────────────────────────────
# Sidebar / navigation
# ──────────────────────────────────────────────────────────────────────────

st.sidebar.title("💳 Fraud Detection")
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Predict — Single Transaction",
        "Predict — Batch (CSV)",
        "Explainability",
        "Model Performance Dashboard",
    ],
)

pipeline_error: str | None = None
pipeline: FraudInferencePipeline | None = None
try:
    pipeline = load_pipeline()
    print(pipeline)
except ArtifactLoadError as exc:
    print(exc)
    pipeline_error = str(exc)

if pipeline_error:
    st.sidebar.error("Artifacts not found — see main panel.")
else:
    st.sidebar.success(f"Model loaded: **{pipeline.model_name}**")
    st.sidebar.caption(f"{len(pipeline.selected_features)} selected features")


def _require_pipeline() -> FraudInferencePipeline:
    if pipeline is None:
        st.error(
            "Model artifacts could not be loaded.\n\n"
            f"**Details:** {pipeline_error}\n\n"
            "Run the data_transformation and model_trainer pipeline stages "
            "(see `main.py`) so that `artifacts/data_transformation/artefacts/"
            "{scaler,feature_selector}.pkl` and `artifacts/model_trainer/models/"
            "best_model.pkl` exist, then reload this page."
        )
        st.stop()
    return pipeline


# ──────────────────────────────────────────────────────────────────────────
# Page: Overview
# ──────────────────────────────────────────────────────────────────────────

if page == "Overview":
    st.title("Credit Card Fraud Detection")
    st.write(
        "This app serves the trained model produced by the "
        "`credit_card_fraud_detection` pipeline: data ingestion → EDA → "
        "validation → transformation → training → evaluation."
    )

    cm = get_config_manager()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Pipeline configuration")
        try:
            tcfg = cm.get_data_transformation_config()
            st.json(
                {
                    "target_column": tcfg.split.target_column,
                    "test_size": tcfg.split.test_size,
                    "val_size": tcfg.split.val_size,
                    "scaling_strategy": tcfg.scaling.strategy,
                    "feature_selection_strategy": tcfg.feature_selection.strategy,
                },
                expanded=False,
            )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not read transformation config: {exc}")

    with col2:
        st.subheader("Model status")
        if pipeline:
            st.metric("Active model", pipeline.model_name)
            st.metric("Features used", len(pipeline.selected_features))
            with st.expander("Selected feature list"):
                st.write(pipeline.selected_features)
        else:
            st.info("No model loaded yet — see error above / in other tabs.")


# ──────────────────────────────────────────────────────────────────────────
# Page: Single prediction
# ──────────────────────────────────────────────────────────────────────────

elif page == "Predict — Single Transaction":
    pipe = _require_pipeline()
    st.title("Score a single transaction")

    with st.expander("Prefill from a real test-set row (optional)"):
        cm = get_config_manager()
        try:
            test_path = cm.get_data_transformation_config().split.test_path
        except Exception:
            test_path = None
        sample_df = load_parquet_safe(str(test_path)) if test_path else None
        if sample_df is not None and st.button("Load random example"):
            st.session_state["prefill_row"] = sample_df.sample(1).iloc[0].to_dict()
        elif sample_df is None:
            st.caption("Test split parquet not found yet — run the transformation stage to enable this.")

    prefill = st.session_state.get("prefill_row", {})

    st.subheader("PCA features (V1–V28)")
    v_cols = st.columns(7)
    values = {}
    for i in range(1, 29):
        col_name = f"V{i}"
        default = float(prefill.get(col_name, 0.0)) if prefill else 0.0
        with v_cols[(i - 1) % 7]:
            values[col_name] = st.number_input(col_name, value=default, format="%.4f", key=f"in_{col_name}")

    st.subheader("Transaction amount")
    amount_default = float(prefill.get("Amount", 50.0)) if prefill else 50.0
    values["Amount"] = st.number_input("Amount (EUR)", min_value=0.0, value=amount_default, format="%.2f")

    show_shap = st.checkbox("Explain this prediction with SHAP", value=True)

    if st.button("Predict", type="primary"):
        pred, prob, X = pipe.predict_single(values)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Fraud probability", f"{prob:.2%}")
        with c2:
            st.metric("Predicted class", "🚨 FRAUD" if pred == 1 else "✅ Legitimate")
        st.progress(min(max(prob, 0.0), 1.0))

        if show_shap:
            with st.spinner("Computing SHAP explanation..."):
                background = X  # single row; explainer needs *some* background
                cm = get_config_manager()
                try:
                    test_path = cm.get_data_transformation_config().split.test_path
                    test_df = load_parquet_safe(str(test_path))
                except Exception:
                    test_df = None
                if test_df is not None:
                    bg_cols = [c for c in X.columns if c in test_df.columns]
                    background = test_df[bg_cols].sample(min(100, len(test_df)), random_state=42)

                try:
                    explainer = expl.build_explainer(pipe.model.predict_proba, background)
                    sv = expl.explain_single(explainer, X)
                    top = expl.top_contributors(sv, list(X.columns))
                    st.subheader("Top feature contributions")
                    st.bar_chart(top.set_index("feature")["shap_value"])
                    st.dataframe(top, use_container_width=True)
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"SHAP explanation failed: {exc}")


# ──────────────────────────────────────────────────────────────────────────
# Page: Batch prediction
# ──────────────────────────────────────────────────────────────────────────

elif page == "Predict — Batch (CSV)":
    pipe = _require_pipeline()
    st.title("Score a batch of transactions")
    st.caption(
        "Upload a CSV with columns V1..V28, Amount (Class optional, id optional). "
        f"Expected raw columns: {', '.join(pipe.expected_raw_columns)}"
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read CSV: {exc}")
            st.stop()

        missing = [c for c in pipe.expected_raw_columns if c not in df_raw.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()

        with st.spinner(f"Scoring {len(df_raw):,} rows..."):
            y_pred, y_prob, X = pipe.predict(df_raw)

        result = df_raw.copy()
        result["fraud_probability"] = y_prob
        result["predicted_class"] = y_pred

        n_flagged = int(y_pred.sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows scored", f"{len(result):,}")
        c2.metric("Flagged as fraud", f"{n_flagged:,}")
        c3.metric("Flag rate", f"{n_flagged / len(result):.2%}")

        if "Class" in df_raw.columns:
            from sklearn.metrics import average_precision_score, recall_score, precision_score

            y_true = df_raw["Class"].values
            st.subheader("Ground-truth comparison")
            m1, m2, m3 = st.columns(3)
            m1.metric("AUPRC", f"{average_precision_score(y_true, y_prob):.4f}")
            m2.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.4f}")
            m3.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.4f}")

        st.subheader("Scored data")
        st.dataframe(result.sort_values("fraud_probability", ascending=False), use_container_width=True)

        st.session_state["last_batch_X"] = X
        st.session_state["last_batch_result"] = result

        st.download_button(
            "Download scored CSV",
            data=result.to_csv(index=False).encode("utf-8"),
            file_name="scored_transactions.csv",
            mime="text/csv",
        )


# ──────────────────────────────────────────────────────────────────────────
# Page: Explainability
# ──────────────────────────────────────────────────────────────────────────

elif page == "Explainability":
    pipe = _require_pipeline()
    st.title("Global explainability (SHAP)")

    X_bg = st.session_state.get("last_batch_X")
    if X_bg is None:
        cm = get_config_manager()
        try:
            test_path = cm.get_data_transformation_config().split.test_path
            test_df = load_parquet_safe(str(test_path))
        except Exception:
            test_df = None
        if test_df is not None:
            target = cm.get_data_transformation_config().split.target_column
            feat_cols = [c for c in pipe.selected_features if c in test_df.columns]
            X_bg = test_df[feat_cols]
        else:
            st.info(
                "No data available yet. Score a batch on the 'Predict — Batch (CSV)' "
                "page first, or make sure the test split parquet exists."
            )
            st.stop()

    sample_size = st.slider("Background sample size", min_value=20, max_value=min(500, len(X_bg)), value=min(150, len(X_bg)))
    if st.button("Compute global feature importance", type="primary"):
        with st.spinner("Running SHAP over the sample (this can take a moment)..."):
            sample = X_bg.sample(sample_size, random_state=42) if len(X_bg) > sample_size else X_bg
            explainer = expl.build_explainer(pipe.model.predict_proba, sample, max_background=sample_size)
            sv = explainer(sample)
            mean_abs = pd.DataFrame(
                {"feature": sample.columns, "mean_abs_shap": np.abs(sv.values).mean(axis=0)}
            ).sort_values("mean_abs_shap", ascending=False)
            st.subheader("Feature importance (mean |SHAP value|)")
            st.bar_chart(mean_abs.set_index("feature"))
            st.dataframe(mean_abs, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────
# Page: Dashboard
# ──────────────────────────────────────────────────────────────────────────

elif page == "Model Performance Dashboard":
    st.title("Model performance dashboard")
    cm = get_config_manager()
    try:
        eval_cfg = cm.get_model_evaluation_config()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load evaluation config: {exc}")
        st.stop()

    comparison = load_json_safe(str(eval_cfg.comparison_table_path))
    report = load_json_safe(str(eval_cfg.evaluation_report_path))

    if comparison:
        st.subheader("Model comparison (ranked by AUPRC)")
        st.dataframe(pd.DataFrame(comparison), use_container_width=True)
    else:
        st.info(f"No comparison table found at {eval_cfg.comparison_table_path}. Run the evaluation stage.")

    if report:
        st.subheader("Per-model evaluation report")
        model_names = list(report.keys()) if isinstance(report, dict) else []
        if model_names:
            chosen = st.selectbox("Model", model_names)
            st.json(report[chosen], expanded=False)
        else:
            st.json(report, expanded=False)
    else:
        st.info(f"No evaluation report found at {eval_cfg.evaluation_report_path}. Run the evaluation stage.")

    plots_dir = Path(eval_cfg.plots_dir)
    if plots_dir.exists():
        images = sorted(plots_dir.glob("*.png"))
        if images:
            st.subheader("Saved plots")
            names = [p.name for p in images]
            chosen_plot = st.selectbox("Plot", names)
            st.image(str(plots_dir / chosen_plot), use_container_width=True)
        else:
            st.info("Plots directory exists but is empty. Run the evaluation stage to populate it.")
    else:
        st.info(f"Plots directory not found at {plots_dir}.")