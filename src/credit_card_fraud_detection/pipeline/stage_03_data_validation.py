import sys
import pandas as pd
from credit_card_fraud_detection.config.configuration import ConfigurationManager
from credit_card_fraud_detection.components.data_validation.orchestrator import ValidationPipeline
from credit_card_fraud_detection.components.data_eda.strategies import JsonReportStrategy
from credit_card_fraud_detection.utils.common import logger

class DataValidationTrainingPipeline:
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager

    def run(self):
        logger.info(">>>>>> Starting Validation Pipeline Stage <<<<<<")
        config = self.config_manager.get_validation_config()
        
        df = pd.read_csv(config.raw_data_path)
        pipeline = ValidationPipeline.build(config)
        results = pipeline.run(df)
        
        pipeline.export(JsonReportStrategy(), str(config.validation_report_json))
        
        if results.get("_validation_metadata", {}).get("failed_validators"):
            logger.error("Pipeline halted due to validation failures.")
            sys.exit(1)
            
        logger.info(">>>>>> Validation Pipeline Completed Successfully <<<<<<")

if __name__ == '__main__':
    pipeline = DataValidationTrainingPipeline()
    pipeline.main()