from __future__ import annotations

import time
from typing import List

import pandas as pd

from credit_card_fraud_detection.entity.config_entity import EDAConfig
from credit_card_fraud_detection.components.data_eda.interface import AnalysisComponent
from credit_card_fraud_detection.components.data_eda.strategies import ReportStrategy
from credit_card_fraud_detection.utils.logging_setup import logger


class ComprehensiveEDAReport:
    """
    Orchestrates a configurable sequence of EDA analysis and visualisation
    components, collects their results, and exports them via a pluggable
    strategy (text, JSON, …).

    Usage — default pipeline (recommended):
        report  = ComprehensiveEDAReport.build_default_pipeline(config)
        results = report.run_pipeline(df)
        report.export_report(JsonReportStrategy(), "eda_report.json")

    Usage — manual / custom pipeline:
        report = ComprehensiveEDAReport(config)
        report.add_analyzer(OverviewAnalyzer()).add_analyzer(FraudAmountAnalyzer())
        results = report.run_pipeline(df)
    """

    def __init__(self, config: EDAConfig) -> None:
        self.config = config
        self._analyzers: List[AnalysisComponent] = []
        self._raw_results: dict = {}
        self._run_metadata: dict = {}

    # ------------------------------------------------------------------
    # Builder API
    # ------------------------------------------------------------------

    def add_analyzer(self, analyzer: AnalysisComponent) -> "ComprehensiveEDAReport":
        """Append a component and return self for chaining."""
        self._analyzers.append(analyzer)
        return self

    # ------------------------------------------------------------------
    # Factory — full CC Fraud 2023 pipeline
    # ------------------------------------------------------------------

    @classmethod
    def build_default_pipeline(cls, config: EDAConfig) -> "ComprehensiveEDAReport":
        """
        Returns a ComprehensiveEDAReport pre-loaded with every analyzer
        appropriate for the Credit Card Fraud Detection 2023 dataset.

        Execution order is intentional:
          1. Health checks first (overview, missing, outliers)
          2. Target profile
          3. Univariate + bivariate statistics
          4. Statistical separability tests (KS, MW, point-biserial)
          5. Feature importance (MI)
          6. Fraud-specific structural analyzers
          7. Visualizers last (slowest, non-blocking to results dict)
        """
        from credit_card_fraud_detection.components.data_eda.analyzers import (
            # Section 1 — health
            OverviewAnalyzer,
            MissingDataAnalyzer,
            OutlierAnalyzer,
            # Section 2 — target
            TargetDistributionAnalyzer,
            # Section 3 — univariate
            UnivariateAnalyzer,
            # Section 4 — bivariate
            BivariateAnalyzer,
            FraudAmountAnalyzer,
            # Section 5 — statistical tests
            KolmogorovSmirnovAnalyzer,
            MannWhitneyAnalyzer,
            PointBiserialCorrelationAnalyzer,
            # Section 6 — importance & redundancy
            MutualInformationAnalyzer,
            SpearmanCorrelationAnalyzer,
            # Section 7 — fraud-specific structural
            PCAFeatureSeparabilityAnalyzer,
            AmountScalingAnalyzer,
            FraudTransactionPatternAnalyzer,
            FeatureScaleConsistencyAnalyzer,
        )
        from credit_card_fraud_detection.components.data_eda.visualizers import (
            DistributionVisualizer,
            CorrelationHeatmapVisualizer,
            BivariateVisualizer,
            TargetDistributionVisualizer,
        )

        report = cls(config)
        (
            report
            # --- Health ---
            .add_analyzer(OverviewAnalyzer())
            .add_analyzer(MissingDataAnalyzer())
            .add_analyzer(OutlierAnalyzer())
            # --- Target ---
            .add_analyzer(TargetDistributionAnalyzer())
            # --- Univariate / Bivariate ---
            .add_analyzer(UnivariateAnalyzer())
            .add_analyzer(BivariateAnalyzer())
            .add_analyzer(FraudAmountAnalyzer())
            # --- Statistical Tests ---
            .add_analyzer(KolmogorovSmirnovAnalyzer())
            .add_analyzer(MannWhitneyAnalyzer())
            .add_analyzer(PointBiserialCorrelationAnalyzer())
            # --- Importance & Redundancy ---
            .add_analyzer(MutualInformationAnalyzer())
            .add_analyzer(SpearmanCorrelationAnalyzer())
            # --- Fraud-Specific Structural ---
            .add_analyzer(PCAFeatureSeparabilityAnalyzer())
            .add_analyzer(AmountScalingAnalyzer())
            .add_analyzer(FraudTransactionPatternAnalyzer())
            .add_analyzer(FeatureScaleConsistencyAnalyzer())
            # --- Visualizers (always last) ---
            .add_analyzer(TargetDistributionVisualizer())
            .add_analyzer(DistributionVisualizer())
            .add_analyzer(CorrelationHeatmapVisualizer())
            .add_analyzer(BivariateVisualizer())
        )
        return report

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def run_pipeline(self, df: pd.DataFrame) -> dict:
        """
        Run every registered component in order.

        A failed component logs a warning and is skipped rather than crashing
        the entire pipeline.  Timing and failure metadata are stored in
        '_pipeline_metadata' inside the results dict.
        """
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty or None.")

        logger.info(
            f"Starting EDA pipeline — {len(self._analyzers)} components | "
            f"{df.shape[0]:,} rows × {df.shape[1]} columns"
        )

        start = time.perf_counter()
        failed: List[str] = []

        for analyzer in self._analyzers:
            name = type(analyzer).__name__
            t0 = time.perf_counter()
            try:
                result = analyzer.analyze(df, self.config)
                if not isinstance(result, dict):
                    raise TypeError(f"Expected dict, got {type(result).__name__}")
                self._raw_results.update(result)
                logger.info(f"  [DONE] {name} ({round(time.perf_counter() - t0, 2)}s)")
            except Exception as exc:
                logger.warning(f"  [FAILED] {name} failed ({round(time.perf_counter() - t0, 2)}s): {exc}")
                failed.append(name)

        total = round(time.perf_counter() - start, 2)
        self._run_metadata = {
            "total_components": len(self._analyzers),
            "failed_components": failed,
            "pipeline_duration_seconds": total,
        }
        self._raw_results["_pipeline_metadata"] = self._run_metadata
        logger.info(f"Pipeline finished in {total}s. Failed: {failed or 'none'}")
        return self._raw_results

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_report(self, strategy: ReportStrategy, filepath: str) -> None:
        if not self._raw_results:
            raise RuntimeError("Call run_pipeline() before export_report().")
        strategy.generate(self._raw_results, filepath)
        logger.info(f"Report written to: {filepath}")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def results(self) -> dict:
        return dict(self._raw_results)

    @property
    def metadata(self) -> dict:
        return dict(self._run_metadata)