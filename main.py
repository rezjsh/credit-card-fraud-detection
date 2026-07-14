from credit_card_fraud_detection.config.configuration import ConfigurationManager
from credit_card_fraud_detection.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from credit_card_fraud_detection.pipeline.stage_02_data_eda import DataEDAPipeline
from credit_card_fraud_detection.pipeline.stage_03_data_validation import DataValidationTrainingPipeline

from credit_card_fraud_detection.pipeline.stage_04_data_transformation import DataTransformationPipeline
from credit_card_fraud_detection.pipeline.stage_06_model_evaluation import EvaluationPipeline
from credit_card_fraud_detection.pipeline.stage_05_model_trainer import ModelTrainerPipeline
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
        data_ingestion_pipeline.run()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")

          # --- Stage 2: Data EDA ---
        STAGE_NAME = "Stage 02: Data EDA"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        eda_pipeline = DataEDAPipeline(config_manager)
        eda_pipeline.run()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")

        # --- Stage 3: Data Validation ---
        STAGE_NAME = "Stage 03: Data Validation"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        validation_pipeline = DataValidationTrainingPipeline(config_manager)
        validation_pipeline.run()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")


        # --- Stage 4: Data Transformation ---
        STAGE_NAME = "Stage 04: Data Transformation"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        transformation_pipeline = DataTransformationPipeline(config_manager)
        transformation_pipeline.run()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")


        # # --- Stage 5: Model Trainer ---
        STAGE_NAME = "Stage 05: Model Trainer"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_trainer_pipeline = ModelTrainerPipeline(config_manager)
        model_trainer_pipeline.run()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")

        
        ## # --- Stage 6: Model Evaluation ---
        STAGE_NAME = "Stage 06: Model Evaluation"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_evaluation_pipeline = EvaluationPipeline(config=config_manager.get_model_evaluation_config(), val_path=config_manager.get_data_transformation_config().split.val_path)
        model_evaluation_pipeline.run()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
        
    except Exception as e:
        logger.exception(e)
        raise e

if __name__ == "__main__":
   main()