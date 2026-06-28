from credit_card_fraud_detection.utils.logging_setup import logger
from credit_card_fraud_detection.utils.common import create_directory, read_yaml_file
from credit_card_fraud_detection.constants.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from credit_card_fraud_detection.entity.config_entity import DataIngestionConfig
from pathlib import Path

class ConfigurationManager:
    def __init__(self, config_filepath = CONFIG_FILE_PATH, params_filepath = PARAMS_FILE_PATH):
        
        self.config = read_yaml_file(Path(config_filepath))
        self.params = read_yaml_file(Path(params_filepath))
  

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