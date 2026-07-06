"""
components/data_transformation/cleaner.py
==========================================
Stage 1 — Data Cleaning

Responsibilities:
  1. Drop non-feature columns (id, etc.)
  2. Validate schema (correct columns, dtypes, class values)
  3. Handle missing values with configurable strategy
  4. Add missing-value indicator flags (optional)
  5. Winsorise outliers on Amount (capping at configurable quantiles)
  6. Remove duplicate rows
  7. Validate row count after cleaning

CC Fraud 2023 specifics:
  • Dataset is typically clean (no missing values in the Kaggle version).
    The missing-value logic is defensive and handles edge cases.
  • Only Amount is Winsorised; V1–V28 are PCA-standardised and should
    not be capped (capping distorts their orthogonal geometry).
  • The 'id' column is a row index with no predictive signal — always dropped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from credit_card_fraud_detection.entity.config_entity import CleaningConfig
from credit_card_fraud_detection.utils.logging_setup import logger


class DataCleaner:
    """
    Stateless cleaning component.

    All cleaning decisions are driven by CleaningConfig; no hardcoded
    column names except through the config object.
    """

    def __init__(self, config: CleaningConfig) -> None:
        self.config = config
        self._report: dict = {}   # accumulates cleaning statistics

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full cleaning pipeline on *df* and return the cleaned DataFrame.

        Steps are run in order and each step's stats are recorded in self.report.
        """
        cfg = self.config
        logger.info(f"DataCleaner: starting with {df.shape[0]:,} rows × {df.shape[1]} columns")
        self._report = {"initial_shape": list(df.shape)}

        df = self._drop_columns(df)
        df = self._validate_schema(df)
        df = self._remove_duplicates(df)
        df = self._handle_missing(df)
        df = self._winsorise_outliers(df)
        df = self._validate_class_column(df)
        self._check_min_rows(df)

        self._report["final_shape"] = list(df.shape)
        logger.info(
            f"DataCleaner: finished → {df.shape[0]:,} rows × {df.shape[1]} columns"
        )
        return df

    def save(self, df: pd.DataFrame, path: Path | None = None) -> Path:
        """Write the cleaned DataFrame to parquet and return the path."""
        out = Path(path or self.config.cleaned_data_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False, compression="snappy")
        logger.info(f"DataCleaner: saved cleaned data → {out}")
        return out

    @property
    def report(self) -> dict:
        return dict(self._report)

    # ─────────────────────────────────────────────────────────────────
    # Private steps
    # ─────────────────────────────────────────────────────────────────

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        to_drop = [c for c in self.config.drop_columns if c in df.columns]
        if to_drop:
            df = df.drop(columns=to_drop)
            logger.info(f"  Dropped columns: {to_drop}")
        self._report["dropped_columns"] = to_drop
        return df

    def _validate_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Log non-numeric columns; attempt coercion, warn if still non-numeric."""
        non_numeric = [
            c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c])
        ]
        if non_numeric:
            logger.warning(f"  Non-numeric columns detected: {non_numeric}. Attempting coercion.")
            for col in non_numeric:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        self._report["non_numeric_columns"] = non_numeric
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        n_before = len(df)
        df = df.drop_duplicates()
        n_removed = n_before - len(df)
        self._report["duplicates_removed"] = n_removed
        if n_removed > 0:
            logger.info(f"  Removed {n_removed:,} duplicate rows")
        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg      = self.config
        strategy = cfg.missing_strategy_numeric
        missing  = df.isnull().sum()
        cols_with_missing = missing[missing > 0].index.tolist()

        self._report["missing_value_counts"] = missing[missing > 0].to_dict()

        if not cols_with_missing:
            logger.info("  No missing values — skipping imputation")
            return df

        logger.info(f"  Imputing {len(cols_with_missing)} columns using strategy='{strategy}'")

        for col in cols_with_missing:
            # Add indicator flag before imputation
            if cfg.flag_missing:
                flag_col = f"{col}_was_missing"
                df[flag_col] = df[col].isna().astype(int)

            if strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "zero":
                df[col] = df[col].fillna(0.0)
            elif strategy == "none":
                raise ValueError(
                    f"Missing values found in '{col}' but strategy='none'. "
                    "Fix the data source or change missing_strategy_numeric."
                )
            else:
                raise ValueError(f"Unknown missing_strategy_numeric: '{strategy}'")

        return df

    def _winsorise_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap outliers at configurable quantile bounds (Winsorisation)."""
        cfg = self.config
        cap_report = {}

        for col in cfg.outlier_cap_columns:
            if col not in df.columns:
                logger.warning(f"  Outlier cap column '{col}' not found — skipping")
                continue

            lo = df[col].quantile(cfg.outlier_lower_quantile)
            hi = df[col].quantile(cfg.outlier_upper_quantile)

            n_capped_lo = int((df[col] < lo).sum())
            n_capped_hi = int((df[col] > hi).sum())

            df[col] = df[col].clip(lower=lo, upper=hi)
            cap_report[col] = {
                "lower_bound":   round(float(lo), 4),
                "upper_bound":   round(float(hi), 4),
                "n_capped_low":  n_capped_lo,
                "n_capped_high": n_capped_hi,
            }
            logger.info(
                f"  Winsorised '{col}': [{lo:.4f}, {hi:.4f}] — "
                f"capped low={n_capped_lo:,}, high={n_capped_hi:,}"
            )

        self._report["outlier_capping"] = cap_report
        return df

    def _validate_class_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure target is binary {0, 1}; cast if needed."""
        target = self.config.target_column
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found after cleaning.")

        unique_vals = set(df[target].unique())
        if not unique_vals.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(
                f"Target column '{target}' contains unexpected values: {unique_vals}. "
                "Expected only 0 and 1."
            )
        df[target] = df[target].astype(int)
        self._report["class_distribution"] = df[target].value_counts().to_dict()
        return df

    def _check_min_rows(self, df: pd.DataFrame) -> None:
        if len(df) < self.config.min_rows_after_cleaning:
            raise RuntimeError(
                f"Only {len(df):,} rows after cleaning — below minimum threshold "
                f"of {self.config.min_rows_after_cleaning:,}. Check the data source."
            )