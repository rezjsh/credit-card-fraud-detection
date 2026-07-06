"""
components/data_transformation/feature_selector.py
====================================================
Stage 4 — Feature Selection

Removes features that are uninformative, redundant, or near-constant
before the train/val/test split so that every split starts from the
same clean feature set.

Three-pass selection pipeline:
  Pass 1 — Variance filter
      Drops columns whose variance falls below min_variance.
      Near-constant columns add no information and hurt tree-based models.

  Pass 2 — Mutual Information (MI) filter
      Ranks features by MI with the fraud label.
      Drops features below mi_score_threshold OR keeps top_k_features,
      whichever is configured.
      MI is computed on a subsample (mi_sample_size) for speed.

  Pass 3 — Spearman redundancy filter
      Among pairs with |Spearman r| > max_spearman_correlation, the lower-MI
      member is dropped.  This removes redundant engineered features without
      sacrificing predictive power.

The selected feature list is stored in a JSON metadata file so the same
set can be enforced at inference time.

CC Fraud 2023 specifics:
  • V1–V28 are PCA components — orthogonal by design, so Spearman pass
    rarely removes any of them. The variance pass may remove a V-column
    if it was inadvertently zeroed during an upstream transform.
  • Amount_zscore is linearly redundant with Amount after StandardScaler,
    so the Spearman pass typically removes one of them — this is expected.
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from credit_card_fraud_detection.components.data_transformation.interface import (
    FittedTransformationComponent,
)
from credit_card_fraud_detection.entity.config_entity import FeatureSelectionConfig
from credit_card_fraud_detection.utils.logging_setup import logger


class FeatureSelector(FittedTransformationComponent):
    """
    Three-pass feature selector: variance → MI → Spearman redundancy.

    fit_transform() must be called on the training set only.
    transform() applies the already-selected feature list to val/test.
    """

    def __init__(self, config: FeatureSelectionConfig) -> None:
        self.config          = config
        self._selected_cols: List[str] = []
        self._mi_scores:     dict = {}
        self._dropped:       dict = {}
        self._report:        dict = {}
        self._is_fitted:     bool = False

    # ─────────────────────────────────────────────────────────────────
    # FittedTransformationComponent interface
    # ─────────────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg    = self.config
        target = cfg.target_column
        logger.info(f"FeatureSelector: starting with {df.shape[1]} columns")

        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found.")

        # Work only on feature columns
        feature_cols = [c for c in df.columns if c != target]

        # ── Pass 1: variance filter ───────────────────────────────
        feature_cols, dropped_var = self._variance_filter(df, feature_cols)

        # ── Pass 2: MI filter ─────────────────────────────────────
        feature_cols, dropped_mi, mi_scores = self._mi_filter(df, feature_cols, target)
        self._mi_scores = mi_scores

        # ── Pass 3: Spearman redundancy filter ────────────────────
        feature_cols, dropped_corr = self._spearman_filter(df, feature_cols)

        self._selected_cols = feature_cols
        self._dropped = {
            "variance_filter":  dropped_var,
            "mi_filter":        dropped_mi,
            "spearman_filter":  dropped_corr,
        }
        self._is_fitted = True

        self._report = {
            "n_input_features":    len([c for c in df.columns if c != target]),
            "n_selected_features": len(self._selected_cols),
            "n_dropped_variance":  len(dropped_var),
            "n_dropped_mi":        len(dropped_mi),
            "n_dropped_spearman":  len(dropped_corr),
            "selected_features":   self._selected_cols,
            "dropped_features":    self._dropped,
            "top_10_mi_scores":    dict(list(sorted(
                mi_scores.items(), key=lambda x: x[1], reverse=True
            )[:10]),),
        }

        logger.info(
            f"FeatureSelector: {len(self._selected_cols)} features selected "
            f"(dropped variance={len(dropped_var)}, MI={len(dropped_mi)}, "
            f"Spearman={len(dropped_corr)})"
        )

        # Save metadata immediately after fit
        self._save_metadata()
        return df[self._selected_cols + [target]]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Call fit_transform() before transform().")
        target   = self.config.target_column
        present  = [c for c in self._selected_cols if c in df.columns]
        missing  = [c for c in self._selected_cols if c not in df.columns]
        if missing:
            logger.warning(f"FeatureSelector: {len(missing)} expected columns missing: {missing}")
        cols = present + ([target] if target in df.columns else [])
        return df[cols]

    @classmethod
    def load(cls, path: Path) -> "FeatureSelector":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        instance = cls(config=payload["config"])
        instance._selected_cols = payload["selected_cols"]
        instance._mi_scores     = payload["mi_scores"]
        instance._dropped       = payload["dropped"]
        instance._is_fitted     = True
        logger.info(f"FeatureSelector: loaded from {path}")
        return instance

    def save_artefact(self, path: Path | None = None) -> Path:
        if not self._is_fitted:
            raise RuntimeError("Nothing to save — not yet fitted.")
        out = Path(path or self.config.selector_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump({
                "config":        self.config,
                "selected_cols": self._selected_cols,
                "mi_scores":     self._mi_scores,
                "dropped":       self._dropped,
            }, f)
        logger.info(f"FeatureSelector: artefact saved → {out}")
        return out

    def save_data(self, df: pd.DataFrame, path: Path | None = None) -> Path:
        out = Path(path or self.config.scaled_data_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False, compression="snappy")
        logger.info(f"FeatureSelector: selected data saved → {out}")
        return out

    @property
    def report(self) -> dict:
        return dict(self._report)

    @property
    def selected_features(self) -> List[str]:
        return list(self._selected_cols)

    # ─────────────────────────────────────────────────────────────────
    # Private passes
    # ─────────────────────────────────────────────────────────────────

    def _variance_filter(self, df: pd.DataFrame, cols: List[str]):
        threshold = self.config.min_variance
        dropped   = []
        kept      = []
        for col in cols:
            v = float(df[col].var())
            if v < threshold:
                dropped.append(col)
                logger.info(f"  [Variance] Dropped '{col}' (var={v:.6f} < {threshold})")
            else:
                kept.append(col)
        return kept, dropped

    def _mi_filter(self, df: pd.DataFrame, cols: List[str], target: str):
        cfg      = self.config
        sample_n = min(cfg.mi_sample_size, len(df))
        sample   = df[cols + [target]].dropna().sample(sample_n, random_state=42)
        X        = sample[cols].select_dtypes(include=[np.number])
        valid_cols = X.columns.tolist()
        y        = sample[target].values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mi_raw = mutual_info_classif(X, y, random_state=42)

        mi_scores = {col: round(float(s), 6) for col, s in zip(valid_cols, mi_raw)}

        # Apply top_k or threshold
        if cfg.top_k_features is not None:
            sorted_cols = sorted(mi_scores, key=lambda c: mi_scores[c], reverse=True)
            kept        = sorted_cols[:cfg.top_k_features]
        else:
            kept        = [c for c in valid_cols if mi_scores.get(c, 0) >= cfg.mi_score_threshold]

        dropped = [c for c in cols if c not in kept]
        for col in dropped:
            logger.info(f"  [MI] Dropped '{col}' (MI={mi_scores.get(col, 0):.6f})")

        return kept, dropped, mi_scores

    def _spearman_filter(self, df: pd.DataFrame, cols: List[str]):
        if len(cols) < 2:
            return cols, []

        cfg       = self.config
        threshold = cfg.max_spearman_correlation

        sample_n = min(20_000, len(df))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = df[cols].sample(sample_n, random_state=42).corr(method="spearman").abs()

        dropped = set()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if cols[i] in dropped or cols[j] in dropped:
                    continue
                if corr.iloc[i, j] >= threshold:
                    # Drop the one with lower MI score
                    mi_i = self._mi_scores.get(cols[i], 0)
                    mi_j = self._mi_scores.get(cols[j], 0)
                    to_drop = cols[i] if mi_i <= mi_j else cols[j]
                    dropped.add(to_drop)
                    logger.info(
                        f"  [Spearman] Dropped '{to_drop}' "
                        f"(|r|={corr.iloc[i,j]:.4f} ≥ {threshold} "
                        f"with '{cols[j if to_drop == cols[i] else i]}')"
                    )

        kept = [c for c in cols if c not in dropped]
        return kept, list(dropped)

    def _save_metadata(self) -> None:
        """Save selected feature list to JSON for inference-time enforcement."""
        out = Path(self.config.metadata_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "selected_features": self._selected_cols,
            "n_features":        len(self._selected_cols),
            "mi_scores":         self._mi_scores,
            "dropped":           self._dropped,
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"FeatureSelector: metadata saved → {out}")