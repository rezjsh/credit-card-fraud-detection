from credit_card_fraud_detection.utils.logging_setup import logger
from credit_card_fraud_detection.components.data_ingestion.interface import IDataIngestionStrategy
from credit_card_fraud_detection.components.data_ingestion.strategies import (
    LocalCSVIngestionStrategy, 
    KaggleAPIIngestionStrategy
)

class DataIngestionFactory:
    """
    Factory class to dynamically fetch the required ingestion strategy.
    """
    @staticmethod
    def get_strategy(strategy_type: str) -> IDataIngestionStrategy:
        logger.info(f"Resolving ingestion strategy for type: '{strategy_type}'")
        
        strategies = {
            "local_csv": LocalCSVIngestionStrategy,
            "kaggle_api": KaggleAPIIngestionStrategy
        }
        
        if strategy_type not in strategies:
            raise ValueError(f"Unsupported Ingestion Strategy: '{strategy_type}' requested.")
            
        return strategies[strategy_type]()