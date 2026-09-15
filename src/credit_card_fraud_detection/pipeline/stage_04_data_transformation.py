import sys
import pandas as pd
from credit_card_fraud_detection.config.configuration import ConfigurationManager
from credit_card_fraud_detection.components.data_transformation.cleaner import DataCleaner
from credit_card_fraud_detection.components.data_transformation.feature_engineer import FeatureEngineer
from credit_card_fraud_detection.components.data_transformation.scaler import FeatureScaler
from credit_card_fraud_detection.components.data_transformation.feature_selector import FeatureSelector
from credit_card_fraud_detection.components.data_transformation.splitter import DataSplitter
from credit_card_fraud_detection.components.data_transformation.resampler import Resampler
from credit_card_fraud_detection.utils.logging_setup import logger

class DataTransformationPipeline:
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager

    def run(self):
        logger.info(">>>>>> Starting Data Transformation Pipeline Stage <<<<<<")
        config = self.config_manager.get_data_transformation_config()
        
        # Load raw data
        df = pd.read_csv(config.raw_data_path)

        # 1. Cleaning (Stateless - Safe to run on full df)
        if config.run_cleaning:
            cleaner = DataCleaner(config.cleaning)
            df = cleaner.clean(df)
            cleaner.save(df)

        # 2. Split (Must happen before stateful transformations to prevent leakage)
        if config.run_split:
            splitter = DataSplitter(config.split)
            train, val, test = splitter.split(df)
        else:
            logger.error("Pipeline requires run_split=True to proceed without leakage.")
            sys.exit(1)

        # 3. Feature Engineering
        if config.run_feature_engineering:
            engineer = FeatureEngineer(config.feature_engineering)
            # Fit strictly on train; transform all
            train = engineer.fit_transform(train)
            val = engineer.transform(val)
            test = engineer.transform(test)

        # 4. Scaling
        if config.run_scaling:
            scaler = FeatureScaler(config.scaling)
            train = scaler.fit_transform(train)
            val = scaler.transform(val)
            test = scaler.transform(test)
            scaler.save_scaler()

        # 5. Feature Selection
        if config.run_feature_selection:
            selector = FeatureSelector(config.feature_selection)
            train = selector.fit_transform(train)
            val = selector.transform(val)
            test = selector.transform(test)
            selector.save_artefact()

        # 6. Resampling (Strictly Train Only)
        if config.run_resampling:
            resampler = Resampler(config.resampling)
            X_train = train.drop(columns=[config.resampling.target_column])
            y_train = train[config.resampling.target_column]
            X_res, y_res = resampler.resample(X_train, y_train)
            train = resampler.to_dataframe(X_res, y_res, X_train.columns.tolist())

        # Final Save of the independent, processed partitions
        logger.info("Saving fully transformed partitions...")
        train.to_parquet(config.split.train_path, index=False, compression="snappy")
        val.to_parquet(config.split.val_path, index=False, compression="snappy")
        test.to_parquet(config.split.test_path, index=False, compression="snappy")

        logger.info(">>>>>> Data Transformation Pipeline Completed Successfully <<<<<<")

if __name__ == '__main__':
    config_manager = ConfigurationManager()
    pipeline = DataTransformationPipeline(config_manager=config_manager)
    pipeline.run()