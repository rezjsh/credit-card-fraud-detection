"""
components/data_transformation/feature_engineer.py
====================================================
Stage 2 — Feature Engineering

Generates interpretable and model-ready features from the raw CC Fraud 2023
schema (V1–V28, Amount, Class).

Feature families produced:
  A. Amount transformations
       Amount_log1p       — log1p to reduce right skew
       Amount_zscore      — standardised without sklearn (dataset-level z)
       Amount_risk_bucket — ordinal risk tier (micro → extreme)
       is_micro_transaction — binary flag (Amount < threshold)
       is_round_amount    — binary flag (Amount ≈ round number)

  B. PCA feature interactions
       V{a}_x_V{b}       — product of configurable pairs
                           (captures non-linear joint fraud signals)

  C. PCA aggregate features
       V_l2_norm          — Euclidean norm of V1–V28 vector
                           (distance from PCA origin; fraud clusters may
                            be further from origin than legit)

All feature names are driven by FeatureEngineeringConfig — no hardcoded
strings except the pca_prefix pattern.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from credit_card_fraud_detection.components.data_transformation.interface import (
    FittedTransformationComponent,
)
from credit_card_fraud_detection.entity.config_entity import FeatureEngineeringConfig
from credit_card_fraud_detection.utils.logging_setup import logger


class FeatureEngineer(FittedTransformationComponent):
    """
    Transforms a cleaned DataFrame into a feature-rich DataFrame.

    All transformations are fit-free (no sklearn fit step) — they are
    computed purely from the current DataFrame values, making them safe
    to apply identically to train, val, and test sets as long as the
    same config is used.

    Exception: Amount_zscore uses dataset-level mean/std.  These are
    computed on the training set and stored in the report for re-use
    on val/test, and persisted to disk for inference.
    """

    def __init__(self, config: FeatureEngineeringConfig) -> None:
        self.config  = config
        self._report: dict = {}
        # Populated during fit on train; reused for val/test/inference
        self._amount_mean: float | None = None
        self._amount_std:  float | None = None
        self._is_fitted: bool = False

    # ─────────────────────────────────────────────────────────────────
    # Public API (FittedTransformationComponent Interface)
    # ─────────────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute dataset-level statistics on *df*, then transform.
        Call this strictly on the training set.
        """
        cfg = self.config
        if cfg.add_amount_zscore and cfg.amount_column in df.columns:
            self._amount_mean = float(df[cfg.amount_column].mean())
            self._amount_std  = float(df[cfg.amount_column].std())
        
        self._is_fitted = True
        return self._engineer(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform *df* using statistics learned during fit_transform.
        Call this on val/test sets and during real-time inference.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit_transform() before transform().")
        return self._engineer(df)

    @classmethod
    def load(cls, path: Path) -> "FeatureEngineer":
        """Load a previously fitted FeatureEngineer from disk for inference."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
            
        instance = cls(config=payload["config"])
        instance._amount_mean = payload["amount_mean"]
        instance._amount_std  = payload["amount_std"]
        instance._is_fitted   = True
        
        logger.info(f"FeatureEngineer: loaded artefact from {path}")
        return instance

    def save_artefact(self, path: Path | None = None) -> Path:
        """Persist fitted parameters to disk for inference use."""
        if not self._is_fitted:
            raise RuntimeError("Nothing to save — not yet fitted.")
            
        # Safely attempt to get the path from config, or use a sensible fallback
        default_path = getattr(self.config, 'engineered_artefact_path', 'artifacts/data_transformation/feature_engineer.pkl')
        out = Path(path or default_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out, "wb") as f:
            pickle.dump({
                "config": self.config,
                "amount_mean": self._amount_mean,
                "amount_std": self._amount_std
            }, f)
            
        logger.info(f"FeatureEngineer: artefact saved → {out}")
        return out

    def save_data(self, df: pd.DataFrame, path: Path | None = None) -> Path:
        """Persist the engineered DataFrame to disk."""
        out = Path(path or self.config.engineered_data_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False, compression="snappy")
        
        logger.info(f"FeatureEngineer: saved engineered data → {out}")
        return out

    @property
    def report(self) -> dict:
        return dict(self._report)

    @property
    def amount_stats(self) -> dict:
        """Return amount mean/std learned on training set (for re-use on val/test)."""
        return {"mean": self._amount_mean, "std": self._amount_std}

    # ─────────────────────────────────────────────────────────────────
    # Private — main dispatcher
    # ─────────────────────────────────────────────────────────────────

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        df  = df.copy()
        features_added: List[str] = []

        logger.info(f"FeatureEngineer: starting with {df.shape[1]} columns")

        # A. Amount features
        if cfg.amount_column in df.columns:
            if cfg.log_transform_amount:
                df, cols = self._log_transform_amount(df)
                features_added.extend(cols)

            if cfg.add_amount_zscore:
                df, cols = self._compute_amount_zscore(df)
                features_added.extend(cols)

            if cfg.amount_bin_labels:
                df, cols = self._bin_amount(df)
                features_added.extend(cols)

            df, cols = self._flag_micro_transactions(df)
            features_added.extend(cols)

            if cfg.flag_round_amounts:
                df, cols = self._flag_round_amounts(df)
                features_added.extend(cols)

        # B. PCA interaction features
        if cfg.pca_interaction_pairs:
            df, cols = self._add_pca_interactions(df)
            features_added.extend(cols)

        # C. PCA aggregate features
        pca_cols = [c for c in df.columns if c.startswith(cfg.pca_prefix) and c[len(cfg.pca_prefix):].isdigit()]
        if pca_cols and cfg.add_pca_l2_norm:
            df, cols = self._add_pca_l2_norm(df, pca_cols)
            features_added.extend(cols)

        self._report = {
            "features_added":  features_added,
            "n_features_added": len(features_added),
            "final_shape":     list(df.shape),
        }
        logger.info(
            f"FeatureEngineer: added {len(features_added)} features → "
            f"{df.shape[1]} total columns"
        )
        return df

    # ─────────────────────────────────────────────────────────────────
    # A. Amount transformations
    # ─────────────────────────────────────────────────────────────────

    def _log_transform_amount(self, df: pd.DataFrame):
        col = self.config.log_amount_col_name
        df[col] = np.log1p(df[self.config.amount_column].clip(lower=0))
        logger.info(f"  + {col} (log1p of Amount)")
        return df, [col]

    def _compute_amount_zscore(self, df: pd.DataFrame):
        col  = self.config.amount_zscore_col_name
        mean = self._amount_mean if self._amount_mean is not None else float(df[self.config.amount_column].mean())
        std  = self._amount_std  if self._amount_std  is not None else float(df[self.config.amount_column].std())

        if std == 0:
            logger.warning("  Amount std = 0 — zscore will be all zeros")
            df[col] = 0.0
        else:
            df[col] = (df[self.config.amount_column] - mean) / std
        logger.info(f"  + {col} (Amount z-score, mean={mean:.4f}, std={std:.4f})")
        return df, [col]

    def _bin_amount(self, df: pd.DataFrame):
        cfg = self.config
        col = cfg.amount_bin_col_name
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df[col] = pd.cut(
                df[cfg.amount_column],
                bins=cfg.amount_bin_edges,
                labels=cfg.amount_bin_labels,
                right=True,
                include_lowest=True,
            ).astype(str)
        # Ordinal-encode to int (preserves order for tree models)
        label_order = {lbl: i for i, lbl in enumerate(cfg.amount_bin_labels)}
        df[col] = df[col].map(label_order).fillna(-1).astype(int)
        logger.info(f"  + {col} (ordinal risk bucket: {cfg.amount_bin_labels})")
        return df, [col]

    def _flag_micro_transactions(self, df: pd.DataFrame):
        cfg = self.config
        col = cfg.micro_transaction_col_name
        df[col] = (df[cfg.amount_column] < cfg.micro_transaction_threshold).astype(int)
        
        n_micro = int(df[col].sum())
        logger.info(f"  + {col} ({n_micro:,} micro-transactions flagged)")
        return df, [col]

    def _flag_round_amounts(self, df: pd.DataFrame):
        cfg = self.config
        col = cfg.round_amount_col_name
        remainder = df[cfg.amount_column] % 1.0
        is_round  = (remainder <= cfg.round_amount_tolerance) | (remainder >= (1.0 - cfg.round_amount_tolerance))
        df[col]   = is_round.astype(int)
        n_round   = int(df[col].sum())
        logger.info(f"  + {col} ({n_round:,} round-amount transactions flagged)")
        return df, [col]

    # ─────────────────────────────────────────────────────────────────
    # B. PCA interaction features
    # ─────────────────────────────────────────────────────────────────

    def _add_pca_interactions(self, df: pd.DataFrame):
        added = []
        for pair in self.config.pca_interaction_pairs:
            if len(pair) != 2:
                logger.warning(f"  Skipping malformed interaction pair: {pair}")
                continue
            a, b = pair
            if a not in df.columns or b not in df.columns:
                logger.warning(f"  Interaction pair '{a}×{b}': one or both columns missing — skipped")
                continue
            col_name = f"{a}_x_{b}"
            df[col_name] = df[a] * df[b]
            added.append(col_name)
            logger.info(f"  + {col_name} (product interaction)")
        return df, added

    # ─────────────────────────────────────────────────────────────────
    # C. PCA aggregate features
    # ─────────────────────────────────────────────────────────────────

    def _add_pca_l2_norm(self, df: pd.DataFrame, pca_cols: List[str]):
        col        = self.config.pca_l2_norm_col_name
        pca_matrix = df[pca_cols].values
        df[col]    = np.linalg.norm(pca_matrix, axis=1)
        logger.info(f"  + {col} (L2 norm of {len(pca_cols)} PCA features)")
        return df, [col]