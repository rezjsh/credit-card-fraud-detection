"""
components/data_transformation/resampler.py
============================================
Stage 6 — Training-Set Resampling (optional)

Applies oversampling / undersampling to the TRAINING SET ONLY to address
class imbalance before model fitting.

CRITICAL RULE: resampling is applied after the train/val/test split.
Never resample before splitting — synthetic minority samples will leak
into the validation and test sets, causing inflated recall and AUPRC.

CC Fraud 2023 specifics:
  • The 2023 dataset is already 50:50 balanced, so resampling is OFF by
    default (config.yaml: run_resampling: false).
  • If you subsample the dataset to simulate real-world fraud rates
    (0.1–0.5%) before running the pipeline, enable this stage and set
    strategy: smote or adasyn.

Supported strategies:
  smote              — Synthetic Minority Over-sampling Technique
  borderline_smote   — SMOTE focused on decision boundary samples
  adasyn             — Adaptive Synthetic Sampling
  random_oversample  — Naive random duplication of minority samples
  random_undersample — Naive random removal of majority samples
  none               — Pass-through (no resampling)

Requires:  pip install imbalanced-learn
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from credit_card_fraud_detection.entity.config_entity import ResamplingConfig
from credit_card_fraud_detection.utils.logging_setup import logger


class Resampler:
    """
    Applies a configurable resampling strategy to the training set.

    Usage:
        resampler = Resampler(config)
        X_res, y_res = resampler.resample(X_train, y_train)
        df_resampled = resampler.to_dataframe(X_res, y_res, feature_names)
        resampler.save(df_resampled)
    """

    def __init__(self, config: ResamplingConfig) -> None:
        self.config  = config
        self._report: dict = {}

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def resample(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the configured resampling strategy to (X, y).

        Args:
            X: Feature matrix (training set only — no val/test!)
            y: Target vector aligned with X.

        Returns:
            (X_resampled, y_resampled) as numpy arrays.
        """
        cfg      = self.config
        strategy = cfg.strategy.lower()
        n_before = len(y)
        n_min    = int(y.sum())
        n_maj    = n_before - n_min

        logger.info(
            f"Resampler: strategy='{strategy}' | "
            f"before: total={n_before:,}, minority={n_min:,}, majority={n_maj:,}"
        )

        if strategy == "none":
            logger.info("  Resampling disabled — returning original data")
            self._report = {"strategy": "none", "unchanged": True}
            return X.values, y.values

        # Safety guard
        if n_min < cfg.min_minority_samples:
            raise ValueError(
                f"Minority class has only {n_min} samples — below "
                f"min_minority_samples={cfg.min_minority_samples}. "
                "Resampling aborted."
            )

        sampler  = self._build_sampler(strategy, n_min, n_maj)
        X_res, y_res = sampler.fit_resample(X, y)

        n_after      = len(y_res)
        n_min_after  = int(y_res.sum())
        n_maj_after  = n_after - n_min_after

        self._report = {
            "strategy":       strategy,
            "before": {"total": n_before, "minority": n_min, "majority": n_maj,
                       "ratio": round(n_maj / n_min, 4) if n_min > 0 else None},
            "after":  {"total": n_after, "minority": n_min_after, "majority": n_maj_after,
                       "ratio": round(n_maj_after / n_min_after, 4) if n_min_after > 0 else None},
            "synthetic_samples_added": max(0, n_min_after - n_min),
            "samples_removed":         max(0, n_maj - n_maj_after),
        }

        logger.info(
            f"  After resampling: total={n_after:,}, "
            f"minority={n_min_after:,}, majority={n_maj_after:,}"
        )
        return X_res, y_res

    def to_dataframe(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
    ) -> pd.DataFrame:
        """Reconstruct a DataFrame from resampled arrays."""
        df = pd.DataFrame(X, columns=feature_names)
        df[self.config.target_column] = y.astype(int)
        return df

    def save(self, df: pd.DataFrame, path: Path | None = None) -> Path:
        out = Path(path or self.config.resampled_train_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False, compression="snappy")
        logger.info(f"Resampler: saved resampled train → {out}")
        return out

    @property
    def report(self) -> dict:
        return dict(self._report)

    # ─────────────────────────────────────────────────────────────────
    # Private — sampler factory
    # ─────────────────────────────────────────────────────────────────

    def _build_sampler(self, strategy: str, n_min: int, n_maj: int):
        """Construct and return the appropriate imbalanced-learn sampler."""
        try:
            from imblearn.over_sampling import (
                ADASYN,
                BorderlineSMOTE,
                RandomOverSampler,
                SMOTE,
            )
            from imblearn.under_sampling import RandomUnderSampler
        except ImportError as exc:
            raise ImportError(
                "imbalanced-learn is required for resampling. "
                "Install with: pip install imbalanced-learn"
            ) from exc

        cfg = self.config
        # Compute sampling_strategy dict: {minority_label: target_count}
        target_minority = int(n_maj * cfg.sampling_ratio)

        common = {"random_state": cfg.random_state}

        if strategy == "smote":
            k = min(cfg.k_neighbors, n_min - 1)
            return SMOTE(sampling_strategy=cfg.sampling_ratio, k_neighbors=k, **common)

        if strategy == "borderline_smote":
            k = min(cfg.k_neighbors, n_min - 1)
            return BorderlineSMOTE(sampling_strategy=cfg.sampling_ratio, k_neighbors=k, **common)

        if strategy == "adasyn":
            k = min(cfg.k_neighbors, n_min - 1)
            return ADASYN(sampling_strategy=cfg.sampling_ratio, n_neighbors=k, **common)

        if strategy == "random_oversample":
            return RandomOverSampler(sampling_strategy=cfg.sampling_ratio, **common)

        if strategy == "random_undersample":
            return RandomUnderSampler(sampling_strategy=cfg.sampling_ratio, **common)

        raise ValueError(
            f"Unknown resampling strategy: '{strategy}'. "
            "Choose from: smote | borderline_smote | adasyn | "
            "random_oversample | random_undersample | none"
        )