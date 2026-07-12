"""
components/model_trainer/interface.py
======================================
Abstract base contract for trainable model wrappers.

Every model candidate (LogisticRegression, RandomForest, XGBoost, LightGBM,
SVM, KNN, NaiveBayes, MLP) is wrapped in a sklearn Pipeline behind a common
interface so the TrainerOrchestrator can treat them uniformly: fit, tune,
predict, score, save, load.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


class TrainableModel(ABC):
    """
    Common contract for every model candidate.

    Implementations wrap a scikit-learn-compatible estimator (or Pipeline)
    and expose a uniform fit / tune / predict / save / load surface so the
    orchestrator never needs model-specific branching logic.
    """

    name: str   # short registry key, e.g. "random_forest"

    @abstractmethod
    def build_pipeline(self) -> Any:
        """Construct and return an untrained sklearn Pipeline for this model."""

    @abstractmethod
    def get_param_grid(self) -> Dict[str, list]:
        """Return the hyperparameter search space for this model."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TrainableModel":
        """Fit the (possibly already-tuned) pipeline on (X, y)."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return hard class predictions for X."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(class=1) probability estimates for X."""

    @abstractmethod
    def save(self, path: Path) -> Path:
        """Persist the fitted pipeline to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "TrainableModel":
        """Load a previously fitted model wrapper from disk."""

    @property
    @abstractmethod
    def best_params(self) -> Dict[str, Any]:
        """Return the hyperparameters used by the currently fitted pipeline."""