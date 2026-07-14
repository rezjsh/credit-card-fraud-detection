import sys
import time
from credit_card_fraud_detection.config.configuration import ConfigurationManager
from credit_card_fraud_detection.components.model_trainer.trainer import ModelTrainer
from credit_card_fraud_detection.utils.common import load_data
from credit_card_fraud_detection.utils.logging_setup import logger

class ModelTrainerPipeline:
    def __init__(self, config_manager: ConfigurationManager) -> None:
        self.config_manager = config_manager

    def run(self) -> None:
        logger.info(">>>>>> Commencing Pipeline Execution Stage 06: Model Trainer <<<<<<")
        start_time = time.perf_counter()
        
        # 1. Load Configurations
        trainer_config = self.config_manager.get_model_trainer_config()
        # Safely extract the param_grids from the parsed params dictionary
        param_grids = self.config_manager.params.model_trainer.get("param_grids", {})

        # 2. Determine Data Source
        train_source = trainer_config.train_resampled_path if trainer_config.use_resampled_train else trainer_config.train_path
        logger.info(f"Loading training data from: {train_source}")
        
        train_df = load_data(train_source)
        val_df = load_data(trainer_config.val_path)

        # 3. Split Features and Target
        target_col = trainer_config.target_column
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_val = val_df.drop(columns=[target_col])
        y_val = val_df[target_col]

        logger.info(f"Train footprint: {X_train.shape} | Val footprint: {X_val.shape}")

        # 4. Orchestrate Training
        orchestrator = ModelTrainer(trainer_config)
        orchestrator.train_all(X_train, y_train, X_val, y_val, param_grids)

        # 5. Persist Artefacts
        orchestrator.save_all_models()
        orchestrator.save_best_model()
        orchestrator.save_metrics()
        orchestrator.save_tuning_results()

        logger.info(f">>>>>> Stage 06 Completed Successfully in {round(time.perf_counter() - start_time, 2)}s <<<<<<\n")

if __name__ == "__main__":
    config_manager = ConfigurationManager()
    pipeline = ModelTrainerPipeline(config_manager)
    pipeline.run()