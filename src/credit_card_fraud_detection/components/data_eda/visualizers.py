"""
visualizers.py
--------------
Visualisation components that implement AnalysisComponent so they slot into
the same pipeline as the statistical analyzers.

Design rules:
  - Every visualizer must be dataset-agnostic: no hardcoded column names.
  - Failures on individual plots are logged and skipped; they must not crash
    the pipeline.
  - Each visualizer creates its own output sub-directory and returns a dict
    summarising what was saved.
  - All figures are closed after saving to avoid memory leaks.
  - A configurable DPI and figure-size make outputs suitable for reports.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe in pipelines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np

from credit_card_fraud_detection.components.data_eda.interface import AnalysisComponent
from credit_card_fraud_detection.utils.logging_setup import logger

# ---------------------------------------------------------------------------
# Shared style defaults
# ---------------------------------------------------------------------------
_PALETTE = "Set2"
_DPI = 150
_STYLE = "whitegrid"
sns.set_theme(style=_STYLE)


def _save(fig: plt.Figure, path: str | Path) -> None:
    """Save and unconditionally close a figure."""
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_dir(config, *sub: str) -> Path:
    """Construct and create a plot sub-directory under config.root_dir."""
    directory = Path(config.root_dir) / "plots" / Path(*sub) if sub else Path(config.root_dir) / "plots"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# ---------------------------------------------------------------------------
# Distribution visualizer
# ---------------------------------------------------------------------------

class DistributionVisualizer(AnalysisComponent):
    """
    Histogram + KDE for every numerical column.

    Skips columns that are constant or have only a single unique value since
    those produce degenerate plots.
    """

    def analyze(self, df: pd.DataFrame, config) -> dict:
        logger.info("Visualizing: Numerical Distributions")
        plot_dir = _plot_dir(config, "distributions")
        saved, skipped = [], []

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            series = df[col].dropna()
            if series.nunique() < 2:
                logger.debug(f"  Skipping constant column: {col}")
                skipped.append(col)
                continue
            try:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(series, kde=True, ax=ax, color=sns.color_palette(_PALETTE)[0])
                ax.set_title(f"Distribution of {col}", fontsize=13)
                ax.set_xlabel(col)
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
                out_path = plot_dir / f"{col}_distribution.png"
                _save(fig, out_path)
                saved.append(col)
            except Exception as exc:
                logger.warning(f"  Distribution plot failed for '{col}': {exc}")
                skipped.append(col)

        logger.info(f"  Distributions: {len(saved)} saved, {len(skipped)} skipped.")
        return {
            "visualizations": (
                f"Distribution plots saved to '{plot_dir}'. "
                f"Saved: {len(saved)}, Skipped: {len(skipped)}."
            )
        }


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------

class CorrelationHeatmapVisualizer(AnalysisComponent):
    """
    Pearson correlation heatmap for all numerical features.

    Drops columns with zero variance before computing correlations to avoid
    NaN rows/columns in the heatmap.
    """

    def analyze(self, df: pd.DataFrame, config) -> dict:
        logger.info("Visualizing: Correlation Heatmap")
        plot_dir = _plot_dir(config)

        numeric_df = df.select_dtypes(include=[np.number])
        # Drop constant columns — they yield all-NaN correlation rows
        numeric_df = numeric_df.loc[:, numeric_df.std() > 0]

        if numeric_df.shape[1] < 2:
            return {"visualizations": "Not enough non-constant numeric columns for a heatmap."}

        try:
            corr = numeric_df.corr()
            n = corr.shape[0]
            fig_size = max(10, n * 0.7)
            fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))
            sns.heatmap(
                corr,
                annot=n <= 20,           # only annotate when readable
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                linewidths=0.5,
                ax=ax,
            )
            ax.set_title("Pearson Correlation Heatmap", fontsize=14)
            out_path = plot_dir / "correlation_heatmap.png"
            _save(fig, out_path)
            logger.info(f"  Heatmap saved to '{out_path}'.")
            return {"visualizations": f"Correlation heatmap saved to '{out_path}'."}
        except Exception as exc:
            logger.warning(f"  Heatmap generation failed: {exc}")
            return {"visualizations": f"Heatmap failed: {exc}"}


# ---------------------------------------------------------------------------
# Bivariate visualizer
# ---------------------------------------------------------------------------

class BivariateVisualizer(AnalysisComponent):
    """
    Visual feature-vs-target comparisons.

    Numerical features  → box plots (one per column).
    Categorical features → stacked bar charts (one per column).

    Columns are discovered dynamically — no hardcoded names.  A configurable
    MAX_CATEGORIES limit prevents illegible plots for very high-cardinality
    columns (those are logged and skipped).
    """

    MAX_CATEGORIES = 20   # skip categorical columns with more unique values than this

    def analyze(self, df: pd.DataFrame, config) -> dict:
        logger.info("Visualizing: Bivariate Plots")
        target = config.target_column
        plot_dir = _plot_dir(config, "bivariate")

        if target not in df.columns:
            logger.warning(f"  Target column '{target}' missing — skipping bivariate plots.")
            return {"visualizations": f"Target column '{target}' missing; bivariate plots skipped."}

        saved_box, saved_bar, skipped = [], [], []

        # 1. Numerical vs Target — box plots
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != target and df[c].nunique() > 1
        ]
        for col in numeric_cols:
            try:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.boxplot(
                    data=df, x=target, y=col,
                    hue=target, palette=_PALETTE, legend=False, ax=ax
                )
                ax.set_title(f"{col}  vs  {target}", fontsize=13)
                out_path = plot_dir / f"boxplot_{col}_vs_{target}.png"
                _save(fig, out_path)
                saved_box.append(col)
            except Exception as exc:
                logger.warning(f"  Boxplot failed for '{col}': {exc}")
                skipped.append(col)

        # 2. Categorical vs Target — stacked bar charts
        cat_cols = [
            c for c in df.select_dtypes(include=["object", "category"]).columns
            if c != target
        ]
        for col in cat_cols:
            n_unique = df[col].nunique()
            if n_unique > self.MAX_CATEGORIES:
                logger.debug(
                    f"  Skipping '{col}' — {n_unique} categories exceeds MAX_CATEGORIES={self.MAX_CATEGORIES}"
                )
                skipped.append(col)
                continue
            try:
                cross_tab = pd.crosstab(df[col], df[target], normalize="index")
                fig, ax = plt.subplots(figsize=(max(8, n_unique * 0.6), 5))
                cross_tab.plot(kind="bar", stacked=True, colormap="viridis", ax=ax)
                ax.set_title(f"Target proportions across {col}", fontsize=13)
                ax.set_ylabel("Proportion")
                ax.set_xlabel(col)
                ax.tick_params(axis="x", rotation=45)
                ax.legend(title=target, bbox_to_anchor=(1.05, 1), loc="upper left")
                out_path = plot_dir / f"stackedbar_{col}_vs_{target}.png"
                _save(fig, out_path)
                saved_bar.append(col)
            except Exception as exc:
                logger.warning(f"  Stacked bar failed for '{col}': {exc}")
                skipped.append(col)

        summary = (
            f"Bivariate plots saved to '{plot_dir}'. "
            f"Box plots: {len(saved_box)}, Stacked bars: {len(saved_bar)}, Skipped: {len(skipped)}."
        )
        logger.info(f"  {summary}")
        return {"visualizations": summary}


# ---------------------------------------------------------------------------
# Target distribution visualizer (bonus)
# ---------------------------------------------------------------------------

class TargetDistributionVisualizer(AnalysisComponent):
    """
    Bar chart of the target-class distribution, making class imbalance
    immediately visible to a human reviewer.
    """

    def analyze(self, df: pd.DataFrame, config) -> dict:
        target = config.target_column
        logger.info(f"Visualizing: Target Distribution for '{target}'")
        plot_dir = _plot_dir(config)

        if target not in df.columns:
            return {"visualizations": f"Target column '{target}' missing."}

        try:
            counts = df[target].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(counts.index.astype(str), counts.values,
                          color=sns.color_palette(_PALETTE, len(counts)))
            ax.bar_label(bars, fmt="%d", padding=3)
            ax.set_title(f"Class Distribution — {target}", fontsize=13)
            ax.set_xlabel(target)
            ax.set_ylabel("Count")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
            out_path = plot_dir / f"target_distribution_{target}.png"
            _save(fig, out_path)
            logger.info(f"  Target distribution plot saved to '{out_path}'.")
            return {"visualizations": f"Target distribution plot saved to '{out_path}'."}
        except Exception as exc:
            logger.warning(f"  Target distribution plot failed: {exc}")
            return {"visualizations": f"Target distribution plot failed: {exc}"}