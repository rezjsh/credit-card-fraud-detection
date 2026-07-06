from abc import ABC, abstractmethod
import pandas as pd


class AnalysisComponent(ABC):
    """
    Base contract for all EDA analysis and visualization components.

    Every component receives the full DataFrame and the EDA config object,
    and must return a plain dict that can be merged into the master results
    store by the orchestrator.
    """

    @abstractmethod
    def analyze(self, df: pd.DataFrame, config) -> dict:
        """
        Run the analysis and return a results dict.

        Args:
            df:     The dataset to analyse.
            config: An EDAConfig instance carrying target column, paths, etc.

        Returns:
            A flat dict whose keys are unique across all components so that
            results can be safely merged with dict.update().
        """