from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url_or_path: str
    local_data_file: Path
    unzip_dir: Path
    ingestion_strategy: str  # e.g., "local_csv", "kaggle_api", "gcs_bucket"