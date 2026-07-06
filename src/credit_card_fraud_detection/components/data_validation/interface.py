from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Type

from credit_card_fraud_detection.components.data_validation.context import ValidationContext

class ValidationStrategy(ABC):
    """Abstract base class for validation strategies in the credit card fraud detection pipeline."""
    @abstractmethod
    def analyze(self, context: ValidationContext) -> dict:
        """Analyze the given context and return validation results."""
        pass

class ValidatorRegistry:
    """Factory registry for dynamic instantiation of ValidationStrategies."""
    _registry: Dict[str, Type[ValidationStrategy]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a validator class."""
        def inner_wrapper(wrapped_class: Type[ValidationStrategy]) -> Type[ValidationStrategy]:
            cls._registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @classmethod
    def create(cls, name: str) -> ValidationStrategy:
        """Instantiate a validator by its registered name."""
        if name not in cls._registry:
            raise ValueError(f"Validator '{name}' is not registered.")
        return cls._registry[name]()