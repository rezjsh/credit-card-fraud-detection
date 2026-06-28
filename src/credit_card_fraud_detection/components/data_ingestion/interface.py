from abc import ABC, abstractmethod
from credit_card_fraud_detection.entity.config_entity import DataIngestionConfig

class IDataIngestionStrategy(ABC):
    """
    Interface for data ingestion strategies.
    """
    @abstractmethod
    def download_data(self, config: DataIngestionConfig) -> None:
        """
        Fetch data from a specific source target and save it locally.
        """
        pass

    @abstractmethod
    def extract_data(self, config: DataIngestionConfig) -> None:
        """
        Handle decompression or formatting into the raw data landing zone.
        """
        pass