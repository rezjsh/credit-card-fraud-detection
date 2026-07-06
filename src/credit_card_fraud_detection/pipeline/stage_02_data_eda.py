import pandas as pd
from credit_card_fraud_detection.components.data_eda.visualizers import BivariateVisualizer, CorrelationHeatmapVisualizer, DistributionVisualizer, TargetDistributionVisualizer
from credit_card_fraud_detection.components.data_eda.orchestrator import ComprehensiveEDAReport
from credit_card_fraud_detection.components.data_eda.analyzers import (
    AmountScalingAnalyzer, BivariateAnalyzer, FeatureScaleConsistencyAnalyzer, FraudAmountAnalyzer, FraudTransactionPatternAnalyzer, KolmogorovSmirnovAnalyzer, MannWhitneyAnalyzer, MutualInformationAnalyzer, OverviewAnalyzer, MissingDataAnalyzer, OutlierAnalyzer, PCAFeatureSeparabilityAnalyzer, PointBiserialCorrelationAnalyzer, SpearmanCorrelationAnalyzer, TargetDistributionAnalyzer, UnivariateAnalyzer
)
from credit_card_fraud_detection.components.data_eda.strategies import TextReportStrategy, JsonReportStrategy
from credit_card_fraud_detection.utils.logging_setup import logger

class DataEDAPipeline:
    def __init__(self, config_manager):
        self.config_manager = config_manager

    def run_pipeline(self):
        eda_config = self.config_manager.get_eda_config()
        
        # Load the data generated from Stage 01
        df = pd.read_csv(eda_config.data_path)

        # 1. Drop unique identifiers to keep analyzers focused
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
        
        # Initialize Orchestrator
        eda_engine = ComprehensiveEDAReport(config=eda_config)
        
        # Register components
        eda_engine.add_analyzer(OverviewAnalyzer()) \
            .add_analyzer(MissingDataAnalyzer()) \
            .add_analyzer(OutlierAnalyzer()) \
            .add_analyzer(TargetDistributionAnalyzer())  \
            .add_analyzer(UnivariateAnalyzer()) \
            .add_analyzer(BivariateAnalyzer()) \
            .add_analyzer(FraudAmountAnalyzer()) \
            .add_analyzer(KolmogorovSmirnovAnalyzer()) \
            .add_analyzer(MannWhitneyAnalyzer()) \
            .add_analyzer(PointBiserialCorrelationAnalyzer()) \
            .add_analyzer(MutualInformationAnalyzer()) \
            .add_analyzer(SpearmanCorrelationAnalyzer()) \
            .add_analyzer(PCAFeatureSeparabilityAnalyzer()) \
            .add_analyzer(AmountScalingAnalyzer()) \
            .add_analyzer(FraudTransactionPatternAnalyzer()) \
            .add_analyzer(FeatureScaleConsistencyAnalyzer()) \
            .add_analyzer(TargetDistributionVisualizer()) \
            .add_analyzer(DistributionVisualizer())  \
            .add_analyzer(CorrelationHeatmapVisualizer()) \
            .add_analyzer(BivariateVisualizer()) 
                  
        # Execute analysis
        eda_engine.run_pipeline(df)
        
        # Export using dual strategies
        eda_engine.export_report(TextReportStrategy(), eda_config.text_report_path)
        eda_engine.export_report(JsonReportStrategy(), eda_config.json_report_path)
        
        logger.info("Data EDA Pipeline completed successfully.")