"""
components/data_transformation/interface.py
============================================
Abstract base contracts for every transformation component.

Two interfaces are provided:

  TransformationComponent  — for stateless or fit-free transforms that
                              accept a DataFrame and return a DataFrame.
                              (e.g. DataCleaner, FeatureEngineer)

  FittedTransformationComponent — for transforms that learn statistics
                                   from training data and must be persisted
                                   so the same transform can be applied at
                                   inference time without refitting.
                                   (e.g. FeatureScaler, FeatureSelector)

All components must also expose:
  report   — dict of statistics / decisions made during the last run
  save()   — persist output data or fitted artefacts to disk
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class TransformationComponent(ABC):
    """
    Base contract for stateless or fit-free transformation stages.

    A component receives a DataFrame, applies its transformation, and
    returns the transformed DataFrame.  It must never retain state that
    would make it unsafe to call transform() on a different DataFrame
    from the one used during construction.
    """

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the transformation to *df* and return the result.

        Args:
            df: Input DataFrame (will not be mutated — components must copy).

        Returns:
            Transformed DataFrame.
        """

    @property
    @abstractmethod
    def report(self) -> dict:
        """Return a dict of statistics / decisions made during the last run."""

    @abstractmethod
    def save(self, df: pd.DataFrame, path: Path | None = None) -> Path:
        """Persist output DataFrame to *path* (or a config-driven default)."""


class FittedTransformationComponent(ABC):
    """
    Base contract for transformation stages that learn from training data.

    Workflow:
        # Training time
        component.fit_transform(X_train)   → transformed X_train
        component.save_artefact()           → persists fitted params to disk

        # Inference / val / test time
        component = ComponentClass.load(path)
        component.transform(X_new)          → transformed X_new
    """

    @abstractmethod
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit on *df* (training data) and return the transformed result.
        Must populate all internal state required by transform().
        """

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the already-fitted transformation to *df*.
        Raises RuntimeError if called before fit_transform().
        """

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "FittedTransformationComponent":
        """Load a previously fitted component from *path*."""

    @abstractmethod
    def save_artefact(self, path: Path | None = None) -> Path:
        """Persist fitted parameters / artefacts to disk."""

    @abstractmethod
    def save_data(self, df: pd.DataFrame, path: Path | None = None) -> Path:
        """Persist the transformed DataFrame to disk."""

    @property
    @abstractmethod
    def report(self) -> dict:
        """Return a dict of statistics from the last fit_transform() call."""