from credit_card_fraud_detection.utils.logging_setup import logger
from credit_card_fraud_detection.entity.config_entity import DataIngestionConfig
from credit_card_fraud_detection.components.data_ingestion.factory import DataIngestionFactory

class DataIngestionOrchestrator:
    """
    Orchestrates the entire execution lifecycle of the data ingestion pipeline component.
    """
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.strategy = DataIngestionFactory.get_strategy(config.ingestion_strategy)

    def execute_ingestion(self) -> None:
        """
        Runs sequentially: download -> extract.
        """
        try:
            logger.info("--- Beginning Ingestion Pipeline Stage ---")
            self.strategy.download_data(config=self.config)
            self.strategy.extract_data(config=self.config)
            logger.info("--- Data Ingestion Stage Completed Successfully ---")
        except Exception as e:
            logger.error(f"Ingestion lifecycle failed abruptly: {e}")
            raise e