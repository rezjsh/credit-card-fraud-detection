from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url_or_path: str
    local_data_file: Path
    unzip_dir: Path
    ingestion_strategy: str  # e.g., "local_csv", "kaggle_api", "gcs_bucket"



@dataclass(frozen=True)
class EDAConfig:
    root_dir: Path
    data_path: Path
    json_report_path: Path
    text_report_path: Path
    target_column: str
    correlation_threshold: float
    outlier_method: str


@dataclass(frozen=True)
class ValidationConfig:
    # Paths
    root_dir: Path
    raw_data_path: Path
    validation_report_json: Path
    eda_report_json: Path
    
    # Schema properties
    target_column: str
    amount_column: str
    pca_prefix: str
    pca_count: int
    min_expected_rows: int
    allowed_classes: List[int]

    # Run parameters
    cv_max_rows: int
    stability_max_rows: int
    importance_max_rows: int
    test_size: float
    train_size: float
    val_size: float
    drift_alpha: float
    split_tolerance: float
    n_splits: int
    n_repeats: int
    target_recall: float
    fp_cost_eur: float
    decision_threshold: float
    calibration_bins: int
    noise_sigmas: List[float]
    subsample_fractions: List[float]
    importance_estimators: int