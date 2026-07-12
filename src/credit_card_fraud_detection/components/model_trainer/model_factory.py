"""
components/model_trainer/model_factory.py
==========================================
Concrete TrainableModel implementations for every supported algorithm,
plus a ModelFactory registry that the orchestrator uses to instantiate
models purely by string key (driven by params.yaml: models_to_train).

Supported models:
  logistic_regression  — fast linear baseline, highly interpretable
  random_forest         — robust bagged trees, handles non-linearity well
  xgboost                — gradient boosting, typically best AUPRC on this dataset
  lightgbm               — faster gradient boosting, similar performance to XGBoost
  svm                     — kernel SVM, strong on small/medium balanced subsets
  knn                     — distance-based baseline, sensitive to scaling (already scaled)
  naive_bayes            — extremely fast probabilistic baseline
  mlp                     — small feed-forward neural network

Each wrapper builds a sklearn Pipeline with a 'clf' step so that
GridSearchCV / RandomizedSearchCV param grids using the 'clf__' prefix
(as defined in params.yaml) work uniformly across every model.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Type

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from credit_card_fraud_detection.components.model_trainer.interface import TrainableModel
from credit_card_fraud_detection.utils.logging_setup import logger


# ─────────────────────────────────────────────────────────────────────────────
# Base implementation shared by every concrete wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _BaseModel(TrainableModel):
    """Shared fit/predict/save/load logic. Subclasses only define the pipeline + grid."""

    name: str = "base"

    def __init__(self, param_grid: Dict[str, list] | None = None, random_state: int = 42) -> None:
        self.random_state = random_state
        self._param_grid_override = param_grid
        self.pipeline: Pipeline | None = None
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TrainableModel":
        if self.pipeline is None:
            self.pipeline = self.build_pipeline()
        logger.info(f"  Fitting {self.name} on {len(X):,} rows × {X.shape[1]} features")
        self.pipeline.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.pipeline.predict_proba(X)[:, 1]

    def save(self, path: Path) -> Path:
        self._check_fitted()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"name": self.name, "pipeline": self.pipeline}, f)
        logger.info(f"  {self.name}: saved → {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "TrainableModel":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        instance = cls()
        instance.pipeline = payload["pipeline"]
        instance._fitted  = True
        return instance

    @property
    def best_params(self) -> Dict[str, Any]:
        self._check_fitted()
        clf = self.pipeline.named_steps.get("clf")
        return clf.get_params() if clf is not None else {}

    def get_param_grid(self) -> Dict[str, list]:
        return self._param_grid_override or {}

    def set_fitted_pipeline(self, pipeline: Pipeline) -> None:
        """Used by the tuner to inject a pipeline found via CV search."""
        self.pipeline = pipeline
        self._fitted  = True

    def _check_fitted(self) -> None:
        if not self._fitted or self.pipeline is None:
            raise RuntimeError(f"{self.name}: call fit() (or set_fitted_pipeline()) first.")


# ─────────────────────────────────────────────────────────────────────────────
# Concrete model wrappers
# ─────────────────────────────────────────────────────────────────────────────

class LogisticRegressionModel(_BaseModel):
    """
    Fast linear baseline. Highly interpretable coefficients.
    Strong starting point and useful as a sanity-check against tree models.
    """
    name = "logistic_regression"

    def build_pipeline(self) -> Pipeline:
        from sklearn.preprocessing import StandardScaler
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                random_state=self.random_state,
                class_weight="balanced",
            )),
        ])


class RandomForestModel(_BaseModel):
    """
    Bagged decision trees. Robust to feature scale and outliers,
    captures non-linear interactions natively (e.g. V14×V12).
    """
    name = "random_forest"

    def build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("clf", RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                class_weight="balanced",
            )),
        ])


class XGBoostModel(_BaseModel):
    """
    Gradient-boosted trees via XGBoost. Typically the strongest single model
    on this dataset due to its ability to model sharp decision boundaries
    around the highly separable PCA features (V14, V12, V10, V4...).

    scale_pos_weight should be tuned near n_majority/n_minority for best recall.
    """
    name = "xgboost"

    def build_pipeline(self) -> Pipeline:
        from xgboost import XGBClassifier
        return Pipeline([
            ("clf", XGBClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric="aucpr",
                use_label_encoder=False,
                tree_method="hist",
            )),
        ])


class LightGBMModel(_BaseModel):
    """
    Gradient-boosted trees via LightGBM. Faster training than XGBoost at
    comparable accuracy; leaf-wise growth can overfit on small minority
    classes — monitor with the OverfitDetector validator after training.
    """
    name = "lightgbm"

    def build_pipeline(self) -> Pipeline:
        from lightgbm import LGBMClassifier
        return Pipeline([
            ("clf", LGBMClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=-1,
            )),
        ])


class SVMModel(_BaseModel):
    """
    Kernel SVM. Strong on small-to-medium balanced subsets but scales poorly
    (O(n^2)-O(n^3)) — not recommended above ~50k training rows without
    subsampling. Included for completeness / smaller experiments.
    """
    name = "svm"

    def build_pipeline(self) -> Pipeline:
        from sklearn.preprocessing import StandardScaler
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                probability=True,
                random_state=self.random_state,
                class_weight="balanced",
            )),
        ])


class KNNModel(_BaseModel):
    """
    Distance-based baseline. Since V1-V28 are already PCA-standardised and
    Amount-derived features are scaled, KNN can work reasonably well, but
    is the slowest at inference time (no model compression).
    """
    name = "knn"

    def build_pipeline(self) -> Pipeline:
        from sklearn.preprocessing import StandardScaler
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_jobs=-1)),
        ])


class NaiveBayesModel(_BaseModel):
    """
    Gaussian Naive Bayes. Extremely fast, useful as a quick sanity baseline
    and for ensembling, but the independence assumption is violated by the
    PCA interaction features deliberately added during feature engineering.
    """
    name = "naive_bayes"

    def build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("clf", GaussianNB()),
        ])


class MLPModel(_BaseModel):
    """
    Small feed-forward neural network. Can capture complex non-linear
    boundaries but is more prone to overfitting on a dataset this size
    without careful regularisation (alpha) and early stopping.
    """
    name = "mlp"

    def build_pipeline(self) -> Pipeline:
        from sklearn.preprocessing import StandardScaler
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                random_state=self.random_state,
                max_iter=500,
                early_stopping=True,
            )),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Registry + Factory
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_REGISTRY: Dict[str, Type[_BaseModel]] = {
    "logistic_regression": LogisticRegressionModel,
    "random_forest":       RandomForestModel,
    "xgboost":             XGBoostModel,
    "lightgbm":            LightGBMModel,
    "svm":                 SVMModel,
    "knn":                 KNNModel,
    "naive_bayes":         NaiveBayesModel,
    "mlp":                 MLPModel,
}


class ModelFactory:
    """
    Instantiates TrainableModel wrappers by string key.

    Usage:
        model = ModelFactory.create("xgboost", param_grid=grid, random_state=42)
    """

    @staticmethod
    def create(
        name: str,
        param_grid: Dict[str, list] | None = None,
        random_state: int = 42,
    ) -> TrainableModel:
        if name not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY.keys())}"
            )
        cls = _MODEL_REGISTRY[name]
        return cls(param_grid=param_grid, random_state=random_state)

    @staticmethod
    def available_models() -> list[str]:
        return list(_MODEL_REGISTRY.keys())

    @staticmethod
    def load_model(name: str, path: Path) -> TrainableModel:
        if name not in _MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{name}'.")
        return _MODEL_REGISTRY[name].load(path)