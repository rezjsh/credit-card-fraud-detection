from credit_card_fraud_detection.utils.logging_setup import logger
from credit_card_fraud_detection.config.configuration import ConfigurationManager
from credit_card_fraud_detection.components.data_ingestion.orchestrator import DataIngestionOrchestrator

class DataIngestionPipeline:
    """
    Executes the ingestion stage within the global MLOps workflow context.
    """
    def __init__(self, config_manager: ConfigurationManager) -> None:
        self.config_manager = config_manager

    def run_pipeline(self) -> None:
        data_ingestion_config = self.config_manager.get_data_ingestion_config()
        
        orchestrator = DataIngestionOrchestrator(config=data_ingestion_config)
        orchestrator.execute_ingestion()