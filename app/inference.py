"""
app/inference.py
=================
Loads the persisted artifacts produced by the training pipeline
(FeatureScaler, FeatureSelector, best model) and reproduces the exact
feature-engineering -> scaling -> selection chain used at training time,
so a raw transaction row (V1-V28, Amount[, Class]) can be scored.

Known limitation
-----------------
`FeatureEngineer` (components/data_transformation/feature_engineer.py) computes
Amount_zscore using the TRAIN split's mean/std, but never persists those two
numbers to disk. Since we don't have them at serving time, this module
approximates them from the full raw dataset (data_ingestion output) the first
time it runs, and caches the result to `artifacts/data_transformation/artefacts/
inference_amount_stats.json`. This is a reasonable proxy (same distribution,
slightly larger sample) but is NOT bit-identical to the stats used during
training. If you need bit-identical reproduction, persist `engineer.amount_stats`
at the end of stage_04 and load it here instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from credit_card_fraud_detection.config.configuration import ConfigurationManager
from credit_card_fraud_detection.components.data_transformation.feature_engineer import FeatureEngineer
from credit_card_fraud_detection.components.data_transformation.scaler import FeatureScaler
from credit_card_fraud_detection.components.data_transformation.feature_selector import FeatureSelector
from credit_card_fraud_detection.components.model_trainer.model_factory import ModelFactory
from credit_card_fraud_detection.components.model_trainer.interface import TrainableModel


class ArtifactLoadError(RuntimeError):
    """Raised when a required artifact file is missing from disk."""


class FraudInferencePipeline:
    """
    End-to-end scorer for raw transactions.

    Usage:
        pipe = FraudInferencePipeline()
        y_pred, y_prob, X_features = pipe.predict(df_raw)
    """

    def __init__(self, config_manager: Optional[ConfigurationManager] = None) -> None:
        self.cm = config_manager or ConfigurationManager()
        self.transform_cfg = self.cm.get_data_transformation_config()
        self.trainer_cfg = self.cm.get_model_trainer_config()

        self.engineer = FeatureEngineer(self.transform_cfg.feature_engineering)
        self._inject_amount_stats(self.engineer)

        self.scaler = self._load_scaler()
        self.selector = self._load_selector()
        self.model, self.model_name = self._load_best_model()

    # ------------------------------------------------------------------
    # Artifact loading
    # ------------------------------------------------------------------

    def _load_scaler(self) -> FeatureScaler:
        path = self.transform_cfg.scaling.scaler_path
        if not Path(path).exists():
            raise ArtifactLoadError(
                f"Scaler artifact not found at {path}. Run the data_transformation "
                "stage first (it saves scaler.pkl there)."
            )
        return FeatureScaler.load(path)

    def _load_selector(self) -> FeatureSelector:
        path = self.transform_cfg.feature_selection.selector_path
        if not Path(path).exists():
            raise ArtifactLoadError(
                f"Feature selector artifact not found at {path}. Run the "
                "data_transformation stage first (it saves feature_selector.pkl there)."
            )
        return FeatureSelector.load(path)

    def _load_best_model(self) -> Tuple[TrainableModel, str]:
        """
        Load best_model.pkl.

        Note: every `_BaseModel` subclass in `model_factory.py` shares the exact
        same generic `load()` implementation (it just unpickles {"name", "pipeline"}
        and doesn't validate the name), so trying every registered model class in
        a loop is pointless -- if the file is corrupted, all of them fail with the
        identical error. We load once, directly, and surface real diagnostics.
        """
        import pickle

        best_path = Path(self.trainer_cfg.best_model_path)
        if not best_path.exists():
            raise ArtifactLoadError(f"best_model.pkl not found at {best_path}. Run model training first.")

        size = best_path.stat().st_size
        if size == 0:
            raise ArtifactLoadError(
                f"{best_path} exists but is empty (0 bytes). The training run that "
                "wrote it was likely interrupted -- re-run stage_05_model_trainer."
            )

        try:
            with open(best_path, "rb") as f:
                payload = pickle.load(f)
        except Exception as exc:  # noqa: BLE001
            raise ArtifactLoadError(
                f"{best_path} ({size:,} bytes) is not a valid/complete pickle file: {exc!r}. "
                "This usually means the file was truncated during a copy/write, or it was "
                "produced by a different (incompatible) library version than what's installed "
                "now -- check `xgboost`/`lightgbm` versions if the best model was one of those. "
                "Re-run stage_05_model_trainer to regenerate it in this environment."
            ) from exc

        name = payload.get("name", "unknown")

        # Reconstruct a TrainableModel wrapper around the already-deserialized pipeline
        # without re-pickling (avoids re-triggering the same failure a second time).
        from credit_card_fraud_detection.components.model_trainer.model_factory import _MODEL_REGISTRY

        cls = _MODEL_REGISTRY.get(name)
        if cls is None:
            raise ArtifactLoadError(
                f"best_model.pkl was saved with unknown model name '{name}'. "
                f"Known models: {ModelFactory.available_models()}"
            )
        instance = cls()
        instance.set_fitted_pipeline(payload["pipeline"])
        return instance, name

    def _inject_amount_stats(self, engineer: FeatureEngineer) -> None:
        """Populate engineer._amount_mean / _amount_std from a cached or freshly-computed proxy."""
        cache_path = Path(self.transform_cfg.root_dir) / "artefacts" / "inference_amount_stats.json"
        amount_col = self.transform_cfg.feature_engineering.amount_column

        if cache_path.exists():
            stats = json.loads(cache_path.read_text())
        else:
            raw_path = self.transform_cfg.raw_data_path
            if not Path(raw_path).exists():
                stats = {"mean": 0.0, "std": 1.0}
            else:
                raw = pd.read_csv(raw_path, usecols=lambda c: c == amount_col)
                stats = {"mean": float(raw[amount_col].mean()), "std": float(raw[amount_col].std())}
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(stats))

        engineer._amount_mean = stats["mean"]
        engineer._amount_std = stats["std"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def expected_raw_columns(self) -> list[str]:
        cfg = self.transform_cfg.feature_engineering
        pca_count = 28
        return [f"{cfg.pca_prefix}{i}" for i in range(1, pca_count + 1)] + [cfg.amount_column]

    @property
    def selected_features(self) -> list[str]:
        return self.selector.selected_features

    def build_features(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Raw columns (V1..V28, Amount[, Class]) -> final model-ready feature matrix."""
        df = df_raw.copy()
        for c in ("id", "Id", "ID"):
            if c in df.columns:
                df = df.drop(columns=[c])

        df = self.engineer.transform(df)
        df = self.scaler.transform(df)
        df = self.selector.transform(df)

        target = self.transform_cfg.split.target_column
        if target in df.columns:
            df = df.drop(columns=[target])
        return df

    def predict(self, df_raw: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Returns (hard predictions, fraud probabilities, feature matrix used)."""
        X = self.build_features(df_raw)
        y_prob = self.model.predict_proba(X)
        y_pred = (y_prob >= 0.5).astype(int)
        return y_pred, y_prob, X

    def predict_single(self, record: dict) -> Tuple[int, float, pd.DataFrame]:
        df_raw = pd.DataFrame([record])
        y_pred, y_prob, X = self.predict(df_raw)
        return int(y_pred[0]), float(y_prob[0]), X