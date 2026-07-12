"""
components/model_evaluation/interface.py
=========================================
Abstract base contract for evaluation components.

Every evaluator receives a fitted model, test features/target, and the
evaluation config, and returns a results dict that the orchestrator merges
into the final evaluation report.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class EvaluationComponent(ABC):
    """
    Base contract for a single evaluation check (metrics, threshold tuning,
    calibration, subgroup analysis, etc).
    """

    @abstractmethod
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        config,
        **kwargs,
    ) -> dict:
        """
        Run the evaluation check and return a results dict.

        Args:
            y_true: Ground-truth labels.
            y_pred: Hard predicted labels.
            y_prob: Predicted probability of the positive (fraud) class.
            config: ModelEvaluationConfig instance.
            kwargs: Optional extra context (e.g. raw DataFrame for subgroup analysis).
        """