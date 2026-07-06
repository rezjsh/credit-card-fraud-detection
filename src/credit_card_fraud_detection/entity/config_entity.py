from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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
    active_validators: List[str]
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



@dataclass(frozen=True)
class CleaningConfig:
    drop_columns: List[str]
    missing_strategy_numeric: str
    flag_missing: bool
    outlier_cap_columns: List[str]
    outlier_lower_quantile: float
    outlier_upper_quantile: float
    min_rows_after_cleaning: int
    target_column: str
    cleaned_data_path: Path

@dataclass(frozen=True)
class FeatureEngineeringConfig:
    log_transform_amount: bool
    log_amount_col_name: str
    amount_bin_edges: List[float]
    amount_bin_labels: List[str]
    amount_bin_col_name: str
    pca_interaction_pairs: List[List[str]]
    add_pca_l2_norm: bool
    pca_l2_norm_col_name: str
    add_amount_zscore: bool
    amount_zscore_col_name: str
    micro_transaction_threshold: float
    micro_transaction_col_name: str
    flag_round_amounts: bool
    round_amount_col_name: str
    round_amount_tolerance: float
    amount_column: str
    pca_prefix: str
    engineered_data_path: Path

@dataclass(frozen=True)
class ScalingConfig:
    strategy: str
    exclude_columns: List[str]
    robust_scale_columns: List[str]
    scaler_path: Path
    scaled_data_path: Path

@dataclass(frozen=True)
class FeatureSelectionConfig:
    strategy: str
    top_k_features: Optional[int]
    mi_score_threshold: float
    max_spearman_correlation: float
    min_variance: float
    mi_sample_size: int
    target_column: str
    selector_path: Path
    scaled_data_path: Path
    metadata_path: Path

@dataclass(frozen=True)
class SplitConfig:
    test_size: float
    val_size: float
    stratify: bool
    random_state: int
    shuffle: bool
    target_column: str
    train_path: Path
    val_path: Path
    test_path: Path

@dataclass(frozen=True)
class ResamplingConfig:
    strategy: str
    k_neighbors: int
    random_state: int
    sampling_ratio: float
    min_minority_samples: int
    target_column: str
    resampled_train_path: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    raw_data_path: Path
    run_cleaning: bool
    run_feature_engineering: bool
    run_scaling: bool
    run_feature_selection: bool
    run_split: bool
    run_resampling: bool
    cleaning: CleaningConfig
    feature_engineering: FeatureEngineeringConfig
    scaling: ScalingConfig
    feature_selection: FeatureSelectionConfig
    split: SplitConfig
    resampling: ResamplingConfig