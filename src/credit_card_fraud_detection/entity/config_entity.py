from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from attrs import field

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



 
# ─────────────────────────────────────────────────────────────────────────────
# Model Training & Evaluation configs
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelTrainerConfig:
    """Unified Configuration for the ModelTrainer stage."""
    # Paths
    root_dir: Path
    train_path: Path
    val_path: Path
    train_resampled_path: Path
    use_resampled_train: bool
    model_dir: Path
    best_model_path: Path
    metrics_path: Path
    tuning_results_path: Path
    
    # Column roles
    target_column: str
    
    # Models and Tuning
    models_to_train: List[str]
    tuning_enabled: bool
    tuning_strategy: str
    tuning_n_iter: int
    tuning_cv_folds: int
    tuning_scoring: str
    tuning_n_jobs: int
    tuning_random_state: int
    tuning_sample_size: int
    
    # Selection
    model_selection_metric: str
    use_class_weight: bool

    random_state: int


@dataclass(frozen=True)
class ModelEvaluationConfig:
    """Unified Configuration for the ModelEvaluator stage."""
    # Paths
    root_dir: Path
    test_path: Path
    model_dir: Path
    best_model_path: Path
    evaluation_report_path: Path
    comparison_table_path: Path
    plots_dir: Path
    
    # Target
    target_column: str
    
    # Evaluation settings
    evaluate_all_models: bool
    optimize_threshold: bool
    threshold_metric: str
    target_recall_floor: float
    decision_threshold: Optional[float]
    
    # Cost assumptions
    fp_cost_eur: float
    fn_cost_default_eur: float
    fn_cost_multiplier: float
    
    # Metrics
    metrics: List[str]
    compute_confidence_intervals: bool
    n_bootstrap: int
    confidence_level: float
    
    # Robustness checks
    run_calibration_check: bool
    run_subgroup_analysis: bool
    random_state: int