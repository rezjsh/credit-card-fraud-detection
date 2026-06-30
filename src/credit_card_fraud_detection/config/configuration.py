from credit_card_fraud_detection.utils.logging_setup import logger
from credit_card_fraud_detection.utils.common import create_directory, read_yaml_file
from credit_card_fraud_detection.constants.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH,SCHEMA_FILE_PATH
from credit_card_fraud_detection.entity.config_entity import DataIngestionConfig, EDAConfig, ValidationConfig
from pathlib import Path

class ConfigurationManager:
    def __init__(self, config_filepath = CONFIG_FILE_PATH, params_filepath = PARAMS_FILE_PATH, schema_filepath = SCHEMA_FILE_PATH):
        
        self.config = read_yaml_file(Path(config_filepath))
        self.params = read_yaml_file(Path(params_filepath))
        self.schema = read_yaml_file(Path(schema_filepath))

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        dirs_to_create = [config.root_dir, config.unzip_dir]
        create_directory(dirs_to_create)
        
        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_url_or_path=str(config.source_url_or_path),
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
            ingestion_strategy=str(config.ingestion_strategy)
        )

    def get_eda_config(self) -> EDAConfig:
        config = self.config.data_eda
        params = self.params.eda_params

        create_directory([config.root_dir])

        return EDAConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            json_report_path=Path(config.json_report_path),
            text_report_path=Path(config.text_report_path),
            target_column=params.target_column,
            correlation_threshold=params.correlation_threshold,
            outlier_method=params.outlier_method
        )
    
    def get_validation_config(self) -> ValidationConfig:
        config = self.config.data_validation
        params = self.params.ValidationSuite
        schema = self.schema
        
        dirs_to_create = [config["root_dir"]]
        create_directory(dirs_to_create)
        
        return ValidationConfig(
            root_dir=Path(config.root_dir),
            raw_data_path=Path(config.raw_data_path),
            validation_report_json=Path(config.validation_report_json),
            eda_report_json=Path(config.eda_report_json),
            target_column=schema.target_column,
            amount_column=schema.amount_column,
            pca_prefix=schema.pca_prefix,
            pca_count=schema.pca_count,
            min_expected_rows=schema.min_expected_rows,
            allowed_classes=schema.allowed_classes,
            cv_max_rows=params.cv_max_rows,
            stability_max_rows=params.stability_max_rows,
            importance_max_rows=params.importance_max_rows,
            test_size=params.test_size,
            train_size=params.train_size,
            val_size=params.val_size,
            drift_alpha=params.drift_alpha,
            split_tolerance=params.split_tolerance,
            n_splits=params.n_splits,
            n_repeats=params.n_repeats,
            target_recall=params.target_recall,
            fp_cost_eur=params.fp_cost_eur,
            decision_threshold=params.decision_threshold,
            calibration_bins=params.calibration_bins,
            noise_sigmas=params.noise_sigmas,
            subsample_fractions=params.subsample_fractions,
            importance_estimators=params.importance_estimators
        )