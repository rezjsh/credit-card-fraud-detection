"""
visualizers.py
--------------
Visualisation components that implement AnalysisComponent so they slot into
the same pipeline as the statistical analyzers.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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


class DistributionVisualizer(AnalysisComponent):
    def analyze(self, df: pd.DataFrame, config) -> dict:
        logger.info("Visualizing: Numerical Distributions")
        plot_dir = _plot_dir(config, "distributions")
        saved, skipped = [], []

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            series = df[col].dropna()
            if series.nunique() < 2:
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
            "visualizations_distributions": (
                f"Distribution plots saved to '{plot_dir}'. "
                f"Saved: {len(saved)}, Skipped: {len(skipped)}."
            )
        }


class CorrelationHeatmapVisualizer(AnalysisComponent):
    def analyze(self, df: pd.DataFrame, config) -> dict:
        logger.info("Visualizing: Correlation Heatmap")
        plot_dir = _plot_dir(config)
        numeric_df = df.select_dtypes(include=[np.number])
        numeric_df = numeric_df.loc[:, numeric_df.std() > 0]

        if numeric_df.shape[1] < 2:
            return {"visualizations_heatmap": "Not enough non-constant numeric columns for a heatmap."}

        try:
            corr = numeric_df.corr()
            n = corr.shape[0]
            fig_size = max(10, n * 0.7)
            fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))
            sns.heatmap(
                corr, annot=n <= 20, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5, ax=ax
            )
            ax.set_title("Pearson Correlation Heatmap", fontsize=14)
            out_path = plot_dir / "correlation_heatmap.png"
            _save(fig, out_path)
            return {"visualizations_heatmap": f"Correlation heatmap saved to '{out_path}'."}
        except Exception as exc:
            return {"visualizations_heatmap": f"Heatmap failed: {exc}"}


class BivariateVisualizer(AnalysisComponent):
    MAX_CATEGORIES = 20

    def analyze(self, df: pd.DataFrame, config) -> dict:
        logger.info("Visualizing: Bivariate Plots")
        target = config.target_column
        plot_dir = _plot_dir(config, "bivariate")

        if target not in df.columns:
            return {"visualizations_bivariate": f"Target column '{target}' missing; bivariate plots skipped."}

        saved_box, saved_bar, skipped = [], [], []

        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target and df[c].nunique() > 1]
        for col in numeric_cols:
            try:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.boxplot(data=df, x=target, y=col, hue=target, palette=_PALETTE, legend=False, ax=ax)
                ax.set_title(f"{col}  vs  {target}", fontsize=13)
                out_path = plot_dir / f"boxplot_{col}_vs_{target}.png"
                _save(fig, out_path)
                saved_box.append(col)
            except Exception as exc:
                skipped.append(col)

        cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if c != target]
        for col in cat_cols:
            n_unique = df[col].nunique()
            if n_unique > self.MAX_CATEGORIES:
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
                skipped.append(col)

        summary = f"Bivariate plots saved to '{plot_dir}'. Box plots: {len(saved_box)}, Stacked bars: {len(saved_bar)}, Skipped: {len(skipped)}."
        return {"visualizations_bivariate": summary}


class TargetDistributionVisualizer(AnalysisComponent):
    def analyze(self, df: pd.DataFrame, config) -> dict:
        target = config.target_column
        plot_dir = _plot_dir(config)

        if target not in df.columns:
            return {"visualizations_target": f"Target column '{target}' missing."}

        try:
            counts = df[target].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(counts.index.astype(str), counts.values, color=sns.color_palette(_PALETTE, len(counts)))
            ax.bar_label(bars, fmt="%d", padding=3)
            ax.set_title(f"Class Distribution — {target}", fontsize=13)
            ax.set_xlabel(target)
            ax.set_ylabel("Count")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
            out_path = plot_dir / f"target_distribution_{target}.png"
            _save(fig, out_path)
            return {"visualizations_target": f"Target distribution plot saved to '{out_path}'."}
        except Exception as exc:
            return {"visualizations_target": f"Target distribution plot failed: {exc}"}