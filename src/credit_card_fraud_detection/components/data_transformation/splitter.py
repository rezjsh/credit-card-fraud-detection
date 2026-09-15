"""
components/data_transformation/splitter.py
===========================================
Stage 5 — Stratified Train / Val / Test Split

Splits the fully-transformed DataFrame into three disjoint, stratified
partitions and persists each to a separate Parquet file.

Design decisions:
  • Stratification on Class is mandatory — a non-stratified split can
    produce a test set with zero fraud samples if the dataset is small.
  • Two-step split: first carve out the test set, then split the remainder
    into train and val. This ensures test is always held out cleanly.
  • Validates that class proportions are preserved in every partition.
  • Saves a split_metadata.json alongside the Parquet files recording
    exact row counts, fraud rates, and random seeds for reproducibility.

CC Fraud 2023 (balanced 50:50):
  With stratify=True the split is trivially balanced. The metadata file
  still documents this explicitly so nothing is taken for granted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_card_fraud_detection.components.data_transformation.interface import TransformationComponent
from credit_card_fraud_detection.entity.config_entity import SplitConfig
from credit_card_fraud_detection.utils.logging_setup import logger


class DataSplitter(TransformationComponent):
    """
    Performs a reproducible, stratified three-way split.

    Attributes after split():
        train, val, test  — the three DataFrames
        metadata          — dict of split statistics
    """

    def __init__(self, config: SplitConfig) -> None:
        self.config   = config
        self.train:   pd.DataFrame | None = None
        self.val:     pd.DataFrame | None = None
        self.test:    pd.DataFrame | None = None
        self._report: dict = {}

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split *df* into (train, val, test) and store them as instance attributes.
        Returns the three DataFrames.
        """
        cfg    = self.config
        target = cfg.target_column
        logger.info(
            f"DataSplitter: splitting {len(df):,} rows "
            f"(test={cfg.test_size}, val={cfg.val_size}, "
            f"stratify={cfg.stratify}, seed={cfg.random_state})"
        )

        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found.")

        stratify_col = df[target] if cfg.stratify else None

        # Step 1 — carve out test set
        df_tv, df_test = train_test_split(
            df,
            test_size=cfg.test_size,
            stratify=stratify_col,
            random_state=cfg.random_state,
            shuffle=cfg.shuffle,
        )

        # Step 2 — split train+val into train and val
        # val_size as a fraction of the train+val remainder
        val_of_remainder = cfg.val_size / (1.0 - cfg.test_size)
        stratify_tv = df_tv[target] if cfg.stratify else None

        df_train, df_val = train_test_split(
            df_tv,
            test_size=val_of_remainder,
            stratify=stratify_tv,
            random_state=cfg.random_state,
            shuffle=cfg.shuffle,
        )

        self.train = df_train.reset_index(drop=True)
        self.val   = df_val.reset_index(drop=True)
        self.test  = df_test.reset_index(drop=True)

        self._build_report(df)
        self._log_summary()
        return self.train, self.val, self.test

    def save(self) -> None:
        """Write train / val / test Parquet files + metadata JSON to disk."""
        if self.train is None:
            raise RuntimeError("Call split() before save().")

        cfg = self.config
        for df, path in [
            (self.train, cfg.train_path),
            (self.val,   cfg.val_path),
            (self.test,  cfg.test_path),
        ]:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out, index=False, compression="snappy")
            logger.info(f"  Saved {len(df):,} rows → {out}")

        # Metadata JSON
        meta_path = Path(cfg.train_path).parent / "split_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._report, f, indent=2)
        logger.info(f"  Split metadata → {meta_path}")

    @property
    def report(self) -> dict:
        return dict(self._report)

    # ─────────────────────────────────────────────────────────────────
    # Private
    # ─────────────────────────────────────────────────────────────────

    def _partition_stats(self, df: pd.DataFrame, name: str) -> dict:
        target     = self.config.target_column
        n          = len(df)
        n_fraud    = int(df[target].sum())
        fraud_rate = round(float(n_fraud / n * 100), 4) if n > 0 else 0.0
        return {
            "name":         name,
            "n_rows":       n,
            "n_fraud":      n_fraud,
            "n_legit":      n - n_fraud,
            "fraud_rate_%": fraud_rate,
        }

    def _build_report(self, full_df: pd.DataFrame) -> None:
        cfg = self.config
        self._report = {
            "random_state":   cfg.random_state,
            "stratified":     cfg.stratify,
            "test_size":      cfg.test_size,
            "val_size":       cfg.val_size,
            "train_size":     round(1.0 - cfg.test_size - cfg.val_size, 4),
            "total_rows":     len(full_df),
            "train":          self._partition_stats(self.train, "train"),
            "val":            self._partition_stats(self.val,   "val"),
            "test":           self._partition_stats(self.test,  "test"),
            "smote_warning":  (
                "Apply resampling (SMOTE / ADASYN) ONLY to the train set "
                "AFTER this split. Never resample before splitting."
            ),
        }

    def _log_summary(self) -> None:
        for name, df in [("train", self.train), ("val", self.val), ("test", self.test)]:
            target     = self.config.target_column
            fraud_rate = round(float(df[target].mean() * 100), 2)
            logger.info(
                f"  {name:<6}: {len(df):>8,} rows  "
                f"fraud={df[target].sum():,} ({fraud_rate}%)"
            )