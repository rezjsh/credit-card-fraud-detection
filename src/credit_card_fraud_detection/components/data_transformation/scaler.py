"""
components/data_transformation/scaler.py
=========================================
Stage 3 — Feature Scaling

Applies column-wise scaling with configurable strategy.

CC Fraud 2023 specifics:
  • V1–V28 are already PCA-standardised (≈ N(0,1)) but StandardScaler
    is still applied to re-centre any residual drift from feature engineering.
  • Amount and its derived columns (log1p, zscore) are raw-scale and need
    robust or standard scaling.
  • Binary flag columns (is_micro_transaction, is_round_amount) and the
    ordinal risk bucket are excluded from scaling.
  • The fitted scaler is persisted to disk so it can be loaded at inference
    time and applied to new transactions without refitting.

Scaler strategy selection:
  standard  → sklearn StandardScaler (z-score normalisation)
  robust    → sklearn RobustScaler (median/IQR — outlier-resistant)
  minmax    → sklearn MinMaxScaler ([0,1] range)
  none      → pass-through (no scaling applied)

Columns in robust_scale_columns always use RobustScaler regardless of
the global strategy — this lets you mix strategies per column type.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from credit_card_fraud_detection.entity.config_entity import ScalingConfig
from credit_card_fraud_detection.utils.logging_setup import logger


# Type alias for any sklearn scaler
_Scaler = StandardScaler | RobustScaler | MinMaxScaler


class FeatureScaler:
    """
    Fits scalers on a training DataFrame and transforms any DataFrame.

    Maintains separate fitted scaler instances per column group so that
    the correct scaler is always applied at inference time.

    Persists a dict of {col: fitted_scaler} to disk via pickle.
    """

    def __init__(self, config: ScalingConfig) -> None:
        self.config = config
        self._scalers:  Dict[str, _Scaler] = {}   # col → fitted scaler
        self._report:   dict = {}
        self._is_fitted: bool = False

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit scalers on *df* (training set) and return transformed copy."""
        self._fit(df)
        return self._transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scalers to *df* (val / test / inference)."""
        if not self._is_fitted:
            raise RuntimeError("Call fit_transform() before transform().")
        return self._transform(df)

    def save_scaler(self, path: Path | None = None) -> Path:
        """Persist fitted scalers dict to disk as a pickle file."""
        if not self._is_fitted:
            raise RuntimeError("Nothing to save — scalers not yet fitted.")
        out = Path(path or self.config.scaler_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump({"scalers": self._scalers, "config": self.config}, f)
        logger.info(f"FeatureScaler: scalers saved → {out}")
        return out

    @classmethod
    def load(cls, path: Path) -> "FeatureScaler":
        """Load a previously saved FeatureScaler from disk."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        instance = cls(config=payload["config"])
        instance._scalers   = payload["scalers"]
        instance._is_fitted = True
        logger.info(f"FeatureScaler: loaded from {path}")
        return instance

    def save_data(self, df: pd.DataFrame, path: Path | None = None) -> Path:
        out = Path(path or self.config.scaled_data_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False, compression="snappy")
        logger.info(f"FeatureScaler: scaled data saved → {out}")
        return out

    @property
    def report(self) -> dict:
        return dict(self._report)

    @property
    def scaled_columns(self) -> List[str]:
        return list(self._scalers.keys())

    # ─────────────────────────────────────────────────────────────────
    # Private
    # ─────────────────────────────────────────────────────────────────

    def _columns_to_scale(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Returns {col: scaler_type} for every column that should be scaled.

        Priority:
          1. Columns in exclude_columns → skip
          2. Columns in robust_scale_columns → "robust"
          3. All remaining numeric columns → global strategy
        """
        cfg         = self.config
        exclude_set = set(cfg.exclude_columns)
        robust_set  = set(cfg.robust_scale_columns)
        col_scaler  = {}

        for col in df.select_dtypes(include=[np.number]).columns:
            if col in exclude_set:
                continue
            if col in robust_set:
                col_scaler[col] = "robust"
            else:
                col_scaler[col] = cfg.strategy

        return col_scaler

    def _make_scaler(self, strategy: str) -> _Scaler | None:
        if strategy == "standard":
            return StandardScaler()
        if strategy == "robust":
            return RobustScaler()
        if strategy == "minmax":
            return MinMaxScaler()
        if strategy == "none":
            return None
        raise ValueError(f"Unknown scaling strategy: '{strategy}'")

    def _fit(self, df: pd.DataFrame) -> None:
        col_scaler_map = self._columns_to_scale(df)
        fitted_report  = {}

        for col, strategy in col_scaler_map.items():
            scaler = self._make_scaler(strategy)
            if scaler is None:
                continue
            values = df[[col]].values
            scaler.fit(values)
            self._scalers[col] = scaler

            # Record pre-fit stats for the report
            fitted_report[col] = {
                "strategy":  strategy,
                "pre_mean":  round(float(df[col].mean()), 6),
                "pre_std":   round(float(df[col].std()),  6),
                "pre_min":   round(float(df[col].min()),  6),
                "pre_max":   round(float(df[col].max()),  6),
            }

        self._is_fitted = True
        self._report    = {
            "n_columns_scaled": len(self._scalers),
            "excluded_columns": self.config.exclude_columns,
            "column_details":   fitted_report,
        }
        logger.info(
            f"FeatureScaler: fitted {len(self._scalers)} scalers "
            f"(strategy='{self.config.strategy}', "
            f"robust override for {len(self.config.robust_scale_columns)} cols)"
        )

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, scaler in self._scalers.items():
            if col not in df.columns:
                logger.warning(f"  Column '{col}' not found in DataFrame — skipping")
                continue
            df[col] = scaler.transform(df[[col]].values).ravel()
        return df