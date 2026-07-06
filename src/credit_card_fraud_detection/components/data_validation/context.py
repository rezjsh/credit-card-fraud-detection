import pandas as pd
from typing import Optional

class ValidationContext:
    """Stateful context to cache expensive Pandas operations for the validation pipeline."""
    def __init__(self, df: pd.DataFrame, config: 'ValidationConfig'):
        self.df = df
        self.config = config
        self._clean_df: Optional[pd.DataFrame] = None
        self._cv_sample: Optional[pd.DataFrame] = None
        self._stability_sample: Optional[pd.DataFrame] = None
        self._importance_sample: Optional[pd.DataFrame] = None

    @property
    def clean_df(self) -> pd.DataFrame:
        if self._clean_df is None:
            self._clean_df = self.df.dropna()
        return self._clean_df

    def get_cv_sample(self) -> pd.DataFrame:
        if self._cv_sample is None:
            max_rows = min(self.config.cv_max_rows, len(self.clean_df))
            self._cv_sample = self.clean_df.sample(max_rows, random_state=42)
        return self._cv_sample

    def get_stability_sample(self) -> pd.DataFrame:
        if self._stability_sample is None:
            max_rows = min(self.config.stability_max_rows, len(self.clean_df))
            self._stability_sample = self.clean_df.sample(max_rows, random_state=42)
        return self._stability_sample
        
    def get_importance_sample(self) -> pd.DataFrame:
        if self._importance_sample is None:
            max_rows = min(self.config.importance_max_rows, len(self.clean_df))
            self._importance_sample = self.clean_df.sample(max_rows, random_state=42)
        return self._importance_sample