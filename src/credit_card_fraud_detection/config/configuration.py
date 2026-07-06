from credit_card_fraud_detection.utils.logging_setup import logger
from credit_card_fraud_detection.utils.common import create_directory, read_yaml_file
from credit_card_fraud_detection.constants.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH,SCHEMA_FILE_PATH
from credit_card_fraud_detection.entity.config_entity import CleaningConfig, DataIngestionConfig, DataTransformationConfig, EDAConfig, FeatureEngineeringConfig, FeatureSelectionConfig, ResamplingConfig, ScalingConfig, SplitConfig, ValidationConfig
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
            active_validators=params.active_validators,
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
    

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        params = self.params.transformation
        
        # Create root and split directories
        create_directory([
            config.root_dir,
            Path(config.train_path).parent,
            Path(config.scaler_path).parent
        ])

        cleaning_cfg = CleaningConfig(
            drop_columns=params.cleaning.drop_columns,
            missing_strategy_numeric=params.cleaning.missing_strategy.numeric,
            flag_missing=params.cleaning.missing_strategy.flag_missing,
            outlier_cap_columns=params.cleaning.outlier_cap_columns,
            outlier_lower_quantile=params.cleaning.outlier_lower_quantile,
            outlier_upper_quantile=params.cleaning.outlier_upper_quantile,
            min_rows_after_cleaning=params.cleaning.min_rows_after_cleaning,
            target_column=config.target_column,
            cleaned_data_path=Path(config.cleaned_data_path)
        )

        engineering_cfg = FeatureEngineeringConfig(
            log_transform_amount=params.feature_engineering.log_transform_amount,
            log_amount_col_name=params.feature_engineering.log_amount_col_name,
            amount_bin_edges=params.feature_engineering.amount_bin_edges,
            amount_bin_labels=params.feature_engineering.amount_bin_labels,
            amount_bin_col_name=params.feature_engineering.amount_bin_col_name,
            pca_interaction_pairs=params.feature_engineering.pca_interaction_pairs,
            add_pca_l2_norm=params.feature_engineering.add_pca_l2_norm,
            pca_l2_norm_col_name=params.feature_engineering.pca_l2_norm_col_name,
            add_amount_zscore=params.feature_engineering.add_amount_zscore,
            amount_zscore_col_name=params.feature_engineering.amount_zscore_col_name,
            micro_transaction_threshold=params.feature_engineering.micro_transaction_threshold,
            micro_transaction_col_name=params.feature_engineering.micro_transaction_col_name,
            flag_round_amounts=params.feature_engineering.flag_round_amounts,
            round_amount_col_name=params.feature_engineering.round_amount_col_name,
            round_amount_tolerance=params.feature_engineering.round_amount_tolerance,
            amount_column=config.amount_column,
            pca_prefix=config.pca_prefix,
            engineered_data_path=Path(config.engineered_data_path)
        )

        scaling_cfg = ScalingConfig(
            strategy=params.scaling.strategy,
            exclude_columns=params.scaling.exclude_columns,
            robust_scale_columns=params.scaling.robust_scale_columns,
            scaler_path=Path(config.scaler_path),
            scaled_data_path=Path(config.scaled_data_path)
        )

        selection_cfg = FeatureSelectionConfig(
            strategy=params.feature_selection.strategy,
            top_k_features=params.feature_selection.top_k_features,
            mi_score_threshold=params.feature_selection.mi_score_threshold,
            max_spearman_correlation=params.feature_selection.max_spearman_correlation,
            min_variance=params.feature_selection.min_variance,
            mi_sample_size=params.feature_selection.mi_sample_size,
            target_column=config.target_column,
            selector_path=Path(config.selector_path),
            scaled_data_path=Path(config.scaled_data_path),
            metadata_path=Path(config.metadata_path)
        )

        split_cfg = SplitConfig(
            test_size=params.split.test_size,
            val_size=params.split.val_size,
            stratify=params.split.stratify,
            random_state=params.split.random_state,
            shuffle=params.split.shuffle,
            target_column=config.target_column,
            train_path=Path(config.train_path),
            val_path=Path(config.val_path),
            test_path=Path(config.test_path)
        )

        resampling_cfg = ResamplingConfig(
            strategy=params.resampling.strategy,
            k_neighbors=params.resampling.k_neighbors,
            random_state=params.resampling.random_state,
            sampling_ratio=params.resampling.sampling_ratio,
            min_minority_samples=params.resampling.min_minority_samples,
            target_column=config.target_column,
            resampled_train_path=Path(config.train_path) # Overwrites train partition with resampled version
        )

        return DataTransformationConfig(
            root_dir=Path(config.root_dir),
            raw_data_path=Path(config.raw_data_path),
            run_cleaning=config.run_cleaning,
            run_feature_engineering=config.run_feature_engineering,
            run_scaling=config.run_scaling,
            run_feature_selection=config.run_feature_selection,
            run_split=config.run_split,
            run_resampling=config.run_resampling,
            cleaning=cleaning_cfg,
            feature_engineering=engineering_cfg,
            scaling=scaling_cfg,
            feature_selection=selection_cfg,
            split=split_cfg,
            resampling=resampling_cfg
        )