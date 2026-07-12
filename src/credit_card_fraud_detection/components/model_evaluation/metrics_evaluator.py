"""
components/model_evaluation/metrics_evaluator.py
=================================================
Core classification metric suite + threshold optimisation + cost analysis.

Three evaluation components:
  MetricsEvaluator      — full metric suite at the default 0.5 threshold
  ThresholdOptimizer    — sweeps thresholds on a validation-like split to
                           find the best operating point for F1/F2/recall floor
  CostAnalyzer          — translates confusion matrix into financial cost
                           using FP/FN cost assumptions from config
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from credit_card_fraud_detection.components.model_evaluation.interface import EvaluationComponent
from credit_card_fraud_detection.entity.config_entity import ModelEvaluationConfig
from credit_card_fraud_detection.utils.logging_setup import logger


def _fbeta_from_pr(precision: float, recall: float, beta: float) -> float:
    b2 = beta ** 2
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0

def _calculate_core_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy":    round(float(accuracy_score(y_true, y_pred)), 6),
        "precision":   round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall":      round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1":          round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "f2":          round(float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)), 6),
        "mcc":         round(float(matthews_corrcoef(y_true, y_pred)), 6),
        "roc_auc":     round(float(roc_auc_score(y_true, y_prob)), 6),
        "auprc":       round(float(average_precision_score(y_true, y_prob)), 6),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 6),
    }

class MetricsEvaluator(EvaluationComponent):
    """Full classification metric suite with Stratified Bootstrapping."""

    def evaluate(self, y_true, y_pred, y_prob, config: ModelEvaluationConfig, **kwargs) -> dict:
        logger.info("Evaluating: Core Metric Suite (with Stratified Bootstrapping)")
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        base_metrics = _calculate_core_metrics(y_true, y_pred, y_prob)
        
        confusion = {
            "true_negatives":  int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives":  int(tp),
        }

        n_total = len(y_true)
        n_fraud = int(np.sum(y_true))
        dummy_acc = round((n_total - n_fraud) / n_total * 100, 4) if n_total else None

        results = {
            "metrics": base_metrics,
            "confusion_matrix": confusion,
            "dummy_baseline": {
                "accuracy_%": dummy_acc,
                "interpretation": (
                    f"A model predicting 'legitimate' for every transaction achieves "
                    f"{dummy_acc}% accuracy but 0% recall. Clear this baseline decisively on recall and AUPRC."
                ),
            },
        }

        # Stratified Bootstrapping for Confidence Intervals
        if getattr(config, "compute_confidence_intervals", False):
            n_boot = getattr(config, "n_bootstrap", 1000)
            alpha = getattr(config, "confidence_level", 0.95)
            rng = np.random.default_rng(getattr(config, "random_state", 42))
            
            boot_store: Dict[str, List[float]] = {m: [] for m in base_metrics.keys()}
            pos_idx = np.where(y_true == 1)[0]
            neg_idx = np.where(y_true == 0)[0]
            
            for _ in range(n_boot):
                boot_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
                boot_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
                idx = np.concatenate([boot_pos, boot_neg])
                
                try:
                    boot_m = _calculate_core_metrics(y_true[idx], y_pred[idx], y_prob[idx])
                    for k, v in boot_m.items():
                        boot_store[k].append(v)
                except ValueError:
                    continue

            lower_p = (1.0 - alpha) / 2.0 * 100
            upper_p = (alpha + (1.0 - alpha) / 2.0) * 100
            
            ci_results = {}
            for m, vals in boot_store.items():
                if vals:
                    ci_results[m] = {
                        "lower": round(float(np.percentile(vals, lower_p)), 6),
                        "upper": round(float(np.percentile(vals, upper_p)), 6)
                    }
            results["confidence_intervals"] = ci_results

        return results


class ThresholdOptimizer(EvaluationComponent):
    """Finds optimal decision boundaries, including direct financial cost minimization."""

    def evaluate(self, y_true, y_pred, y_prob, config: ModelEvaluationConfig, **kwargs) -> dict:
        logger.info(f"Evaluating: Threshold Optimisation (metric={config.threshold_metric})")
        df = kwargs.get("df")
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

        best = {"threshold": 0.5, "score": -float('inf'), "precision": None, "recall": None}
        
        amounts = df["Amount"].to_numpy() if (df is not None and "Amount" in df.columns) else None
        fp_cost = getattr(config, "fp_cost_eur", 3.0)
        fn_multiplier = getattr(config, "fn_cost_multiplier", 1.0)

        for thresh, prec, rec in zip(thresholds, precisions[:-1], recalls[:-1]):
            current_preds = (y_prob >= thresh).astype(int)
            
            if config.threshold_metric == "minimum_cost":
                if amounts is None:
                    raise ValueError("Cannot optimize for 'minimum_cost' without 'Amount' column in test data.")
                
                fn_mask = (y_true == 1) & (current_preds == 0)
                fp_mask = (y_true == 0) & (current_preds == 1)
                
                cost = (np.sum(amounts[fn_mask]) * fn_multiplier) + (np.sum(fp_mask) * fp_cost)
                score = -cost 
                
            elif config.threshold_metric == "f1":
                score = _fbeta_from_pr(prec, rec, beta=1)
            elif config.threshold_metric == "f2":
                score = _fbeta_from_pr(prec, rec, beta=2)
            elif config.threshold_metric == "recall_at_precision":
                score = prec if rec >= getattr(config, "target_recall_floor", 0.90) else -1
            else:
                raise ValueError(f"Unknown threshold_metric: {config.threshold_metric}")

            if score > best["score"]:
                best = {
                    "threshold": round(float(thresh), 4),
                    "score":     round(float(score), 4),
                    "precision": round(float(prec), 4),
                    "recall":    round(float(rec), 4),
                }

        return {
            "threshold_optimization": {
                "metric_optimized":  config.threshold_metric,
                "default_threshold": 0.5,
                "optimal_threshold": best["threshold"],
                "optimal_score":     best["score"],
                "precision_at_optimal": best["precision"],
                "recall_at_optimal":    best["recall"],
            }
        }

class CostAnalyzer(EvaluationComponent):
    """
    Financial cost interpretation of the confusion matrix.

    FN cost defaults to the mean Amount of fraudulent transactions in the
    evaluated set if an 'Amount' column is supplied via kwargs['df'];
    otherwise falls back to config.fn_cost_default_eur.
    """

    def evaluate(self, y_true, y_pred, y_prob, config: ModelEvaluationConfig, **kwargs) -> dict:
        logger.info("Evaluating: Financial Cost Analysis")
        df = kwargs.get("df")

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        if df is not None and "Amount" in df.columns:
            fraud_mask = (y_true == 1)
            fraud_amounts = df.loc[fraud_mask, "Amount"] if fraud_mask.any() else pd.Series([config.fn_cost_default_eur])
            fn_cost = float(fraud_amounts.mean()) if len(fraud_amounts) else config.fn_cost_default_eur
        else:
            fn_cost = config.fn_cost_default_eur

        fp_cost = config.fp_cost_eur

        model_cost = round(fn * fn_cost + fp * fp_cost, 2)
        dummy_cost = round(int(np.sum(y_true)) * fn_cost, 2)
        savings    = round(dummy_cost - model_cost, 2)

        return {
            "cost_analysis": {
                "assumptions": {
                    "fn_cost_eur": round(fn_cost, 2),
                    "fp_cost_eur": fp_cost,
                },
                "model_total_cost_eur": model_cost,
                "dummy_total_cost_eur": dummy_cost,
                "cost_savings_eur":     savings,
                "cost_reduction_%":     round(savings / dummy_cost * 100, 2) if dummy_cost > 0 else None,
                "breakdown": {
                    "false_negatives_cost_eur": round(fn * fn_cost, 2),
                    "false_positives_cost_eur": round(fp * fp_cost, 2),
                },
            }
        }