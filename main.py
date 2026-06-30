from credit_card_fraud_detection.config.configuration import ConfigurationManager
from credit_card_fraud_detection.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from credit_card_fraud_detection.pipeline.stage_02_data_eda import DataEDAPipeline
from credit_card_fraud_detection.pipeline.stage_03_data_validation import DataValidationTrainingPipeline
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

        #   # --- Stage 2: Data EDA ---
        # STAGE_NAME = "Stage 02: Data EDA"
        # logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        # eda_pipeline = DataEDAPipeline(config_manager)
        # eda_pipeline.run_pipeline()
        # logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")

        # --- Stage 3: Data Validation ---
        STAGE_NAME = "Stage 03: Data Validation"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        validation_pipeline = DataValidationTrainingPipeline()
        validation_pipeline.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")


   
    except Exception as e:
        logger.exception(e)
        raise e

if __name__ == "__main__":
   main()