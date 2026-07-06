import time
import pandas as pd
from typing import List
from credit_card_fraud_detection.components.data_validation.interface import ValidationStrategy, ValidatorRegistry
from credit_card_fraud_detection.entity.config_entity import ValidationConfig
from credit_card_fraud_detection.components.data_eda.strategies import ReportStrategy
from credit_card_fraud_detection.entity.config_entity import ValidationConfig
from credit_card_fraud_detection.components.data_validation.context import ValidationContext
import credit_card_fraud_detection.components.data_validation.validators   # Note: We must import validators so the decorators execute and register the classes
from credit_card_fraud_detection.utils.common import logger

class ValidationPipeline:
    def __init__(self, config: ValidationConfig) -> None:
        self.config = config
        self._validators: List[ValidationStrategy] = []
        self._results: dict = {}

    def add(self, validator: ValidationStrategy) -> "ValidationPipeline":
        self._validators.append(validator)
        return self

    @classmethod
    def build(cls, config: ValidationConfig) -> "ValidationPipeline":
        vp = cls(config)
        
        for val_name in config.active_validators:
            try:
                validator_instance = ValidatorRegistry.create(val_name)
                vp.add(validator_instance)
            except ValueError as e:
                logger.error(f"Failed to load validator: {e}")
        return vp

    def run(self, df: pd.DataFrame) -> dict:
        if df is None or df.empty:
            raise ValueError("DataFrame is empty.")

        logger.info(f"Starting Validation Pipeline — {len(self._validators)} validators")
        start = time.perf_counter()
        failed = []

        # ── Initialize the Context Cache Once ──
        context = ValidationContext(df, self.config)

        for validator in self._validators:
            name = type(validator).__name__
            t0 = time.perf_counter()
            try:
                result = validator.analyze(context)
                self._results.update(result)
                logger.info(f"  [DONE] {name} ({round(time.perf_counter() - t0, 2)}s)")
            except Exception as exc:
                logger.warning(f"  [FAILED] {name} failed: {exc}")
                failed.append(name)

        total = round(time.perf_counter() - start, 2)
        self._results["_validation_metadata"] = {
            "total_validators": len(self._validators),
            "failed_validators": failed,
            "duration_seconds": total,
        }
        return self._results

    def export(self, strategy: ReportStrategy, filepath: str) -> None:
        if not self._results:
            raise RuntimeError("Call run() before export().")
        strategy.generate(self._results, filepath)
        logger.info(f"Validation report written to: {filepath}")