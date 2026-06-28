from credit_card_fraud_detection.config.configuration import ConfigurationManager
from credit_card_fraud_detection.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from credit_card_fraud_detection.utils.logging_setup import logger



def main():
    """
    Main execution function to orchestrate the Data Ingestion stage.
    """
    try:
        config_manager = ConfigurationManager()

        # --- Stage 1: Data Ingestion ---
        STAGE_NAME = "Stage 01: Data Ingestion"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_ingestion_pipeline = DataIngestionPipeline(config_manager)
        data_ingestion_pipeline.run_pipeline()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")

   
    except Exception as e:
        logger.exception(e)
        raise e

if __name__ == "__main__":
   main()