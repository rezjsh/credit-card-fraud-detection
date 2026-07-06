from __future__ import annotations

import warnings
from typing import List

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp, mannwhitneyu, pointbiserialr
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

from credit_card_fraud_detection.components.data_eda.interface import AnalysisComponent
from credit_card_fraud_detection.utils.logging_setup import logger


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _pca_cols(df: pd.DataFrame) -> List[str]:
    """Return V1–V28 columns present in the DataFrame."""
    return [c for c in df.columns if c.upper().startswith("V") and c[1:].isdigit()]


def _get_target(df: pd.DataFrame, config) -> str | None:
    t = config.target_column
    if t not in df.columns:
        logger.warning(f"Target column '{t}' not found in DataFrame.")
        return None
    return t


# ============================================================================
# SECTION 1 — Basic Dataset Health
# ============================================================================

class OverviewAnalyzer(AnalysisComponent):
    """
    High-level shape, memory, type, and data-quality summary.

    Also identifies the 'id' column (if present) as a non-feature column
    so downstream analyzers can exclude it automatically.
    """

    def analyze(self, df: pd.DataFrame, config) -> dict:
        logger.info("Analyzing: Dataset Overview")
        n_rows, n_cols = df.shape
        dup = int(df.duplicated().sum())
        non_feature_cols = [c for c in ["id", "Id", "ID"] if c in df.columns]

        return {
            "num_rows": n_rows,
            "num_cols": n_cols,
            "memory_usage_mb": round(float(df.memory_usage(deep=True).sum() / 1024**2), 4),
            "duplicate_rows": dup,
            "duplicate_percentage": round(dup / n_rows * 100, 4) if n_rows else 0.0,
            "non_feature_columns": non_feature_cols,
            "pca_feature_count": len(_pca_cols(df)),
            "dtypes": df.dtypes.astype(str).to_dict(),
        }


class MissingDataAnalyzer(AnalysisComponent):
    """Column-level missing-value profile sorted by severity."""

    def analyze(self, df: pd.DataFrame, config) -> dict:
        logger.info("Analyzing: Missing Data Profile")
        null_counts = df.isnull().sum()
        profile = {
            col: {
                "count": int(count),
                "percentage": round(count / len(df) * 100, 4),
            }
            for col, count in null_counts.items()
            if count > 0
        }
        profile = dict(sorted(profile.items(), key=lambda x: x[1]["count"], reverse=True))
        return {
            "missing_data": profile,
            "has_missing": len(profile) > 0,
        }


class OutlierAnalyzer(AnalysisComponent):
    """
    IQR-based outlier detection with boundary values.

    For the fraud dataset this is especially informative on 'Amount'
    (right-skewed raw feature).  PCA columns rarely show extreme outliers
    since they are already standardized, but it is still reported.
    """

    def analyze(self, df: pd.DataFrame, config) -> dict:
        logger.info("Analyzing: Outliers (IQR method)")
        target = config.target_column
        numeric_df = df.select_dtypes(include=[np.number]).drop(
            columns=[c for c in ["id", "Id", "ID", target] if c in df.columns],
            errors="ignore",
        )
        profile = {}
        for col in numeric_df.columns:
            s = numeric_df[col].dropna()
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            IQR = Q3 - Q1
            lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            n_out = int(((s < lo) | (s > hi)).sum())
            if n_out > 0:
                profile[col] = {
                    "outlier_count": n_out,
                    "percentage": round(n_out / len(df) * 100, 4),
                    "lower_bound": round(float(lo), 6),
                    "upper_bound": round(float(hi), 6),
                    "min_value": round(float(s.min()), 6),
                    "max_value": round(float(s.max()), 6),
                }

        profile = dict(sorted(profile.items(), key=lambda x: x[1]["outlier_count"], reverse=True))
        return {"outliers": profile}


# ============================================================================
# SECTION 2 — Target Analysis (Fraud-Specific)
# ============================================================================

class TargetDistributionAnalyzer(AnalysisComponent):
    """
    Class balance profile.

    The 2023 dataset is artificially balanced (50/50).  This analyzer makes
    that explicit so the user knows whether SMOTE or class weights are needed.
    """

    def analyze(self, df: pd.DataFrame, config) -> dict:
        target = _get_target(df, config)
        logger.info(f"Analyzing: Target Distribution for '{target}'")
        if target is None:
            return {"target_distribution": "Target column not found."}

        counts = df[target].value_counts().sort_index()
        pcts = (df[target].value_counts(normalize=True) * 100).sort_index().round(4)

        distribution = {
            str(lbl): {"count": int(counts[lbl]), "percentage": float(pcts[lbl])}
            for lbl in counts.index
        }
        ratio = round(float(counts.max() / counts.min()), 4) if counts.min() > 0 else None
        is_balanced = ratio is not None and ratio < 1.5

        return {
            "target_distribution": distribution,
            "class_imbalance_ratio": ratio,
            "is_balanced": is_balanced,
            "balance_note": (
                "Dataset is approximately balanced — class weighting or SMOTE may not be needed."
                if is_balanced
                else f"Dataset is imbalanced ({ratio}:1) — consider SMOTE, class weights, or AUPRC as evaluation metric."
            ),
        }


# ============================================================================
# SECTION 3 — Univariate Feature Statistics
# ============================================================================

class UnivariateAnalyzer(AnalysisComponent):
    """
    Per-feature descriptive statistics for all numeric columns.

    Reports mean, std, median, min, max, skewness, and excess kurtosis.
    Flags highly skewed features (|skew| > 1) — 'Amount' is typically the
    main offender and will need log-transformation before modelling.
    """

    SKEW_THRESHOLD = 1.0

    def analyze(self, df: pd.DataFrame, config) -> dict:
        logger.info("Analyzing: Univariate Statistics")
        target = config.target_column
        exclude = {c for c in ["id", "Id", "ID", target] if c in df.columns}
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

        stats_dict = {}
        highly_skewed = []

        for col in numeric_cols:
            s = df[col].dropna()
            skew = round(float(s.skew()), 4)
            if abs(skew) > self.SKEW_THRESHOLD:
                highly_skewed.append(col)
            stats_dict[col] = {
                "mean":   round(float(s.mean()), 6),
                "std":    round(float(s.std()), 6),
                "median": round(float(s.median()), 6),
                "min":    round(float(s.min()), 6),
                "max":    round(float(s.max()), 6),
                "skewness": skew,
                "kurtosis": round(float(s.kurtosis()), 4),
            }

        return {
            "univariate_analysis": stats_dict,
            "highly_skewed_features": highly_skewed,
            "skew_threshold_used": self.SKEW_THRESHOLD,
        }


# ============================================================================
# SECTION 4 — Bivariate / Fraud-vs-Legit Comparisons
# ============================================================================

class BivariateAnalyzer(AnalysisComponent):
    """
    Per-feature mean and standard deviation broken down by fraud / legit class.

    Computes the absolute mean difference between classes, which is a quick
    proxy for predictive power before running statistical tests.
    """

    def analyze(self, df: pd.DataFrame, config) -> dict:
        target = _get_target(df, config)
        logger.info(f"Analyzing: Bivariate Statistics vs '{target}'")
        if target is None:
            return {"bivariate_analysis": "Target column missing."}

        exclude = {c for c in ["id", "Id", "ID"] if c in df.columns}
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                        if c not in exclude and c != target]

        grouped = df.groupby(target)[numeric_cols]
        mean_by_class = grouped.mean().round(6).to_dict()
        std_by_class  = grouped.std().round(6).to_dict()

        # Absolute mean difference per feature (useful ranking signal)
        classes = sorted(df[target].unique())
        mean_diff = {}
        if len(classes) == 2:
            c0, c1 = str(classes[0]), str(classes[1])
            for col in numeric_cols:
                m0 = mean_by_class.get(col, {}).get(classes[0], np.nan)
                m1 = mean_by_class.get(col, {}).get(classes[1], np.nan)
                mean_diff[col] = round(abs(float(m1) - float(m0)), 6)
            mean_diff = dict(sorted(mean_diff.items(), key=lambda x: x[1], reverse=True))

        return {
            "bivariate_analysis": {
                "mean_by_class": mean_by_class,
                "std_by_class":  std_by_class,
                "abs_mean_difference_by_feature": mean_diff,
            }
        }


class FraudAmountAnalyzer(AnalysisComponent):
    """
    Deep analysis of 'Amount' split by fraud / legit class.

    'Amount' is the only interpretable raw financial feature.  Fraud
    transactions tend to cluster at specific amount ranges — this analyzer
    surfaces those patterns with percentile profiles and log-scale statistics.
    """

    AMOUNT_COL = "Amount"

    def analyze(self, df: pd.DataFrame, config) -> dict:
        target = _get_target(df, config)
        logger.info("Analyzing: Transaction Amount by Fraud Class")

        if target is None or self.AMOUNT_COL not in df.columns:
            return {"fraud_amount_analysis": "Amount column or target column not found."}

        result = {}
        for cls, label in [(0, "legitimate"), (1, "fraud")]:
            subset = df.loc[df[target] == cls, self.AMOUNT_COL].dropna()
            if subset.empty:
                result[label] = "No records."
                continue

            log_vals = np.log1p(subset)
            percentiles = subset.quantile([0.25, 0.5, 0.75, 0.90, 0.95, 0.99])
            result[label] = {
                "count":       int(len(subset)),
                "mean":        round(float(subset.mean()), 4),
                "median":      round(float(subset.median()), 4),
                "std":         round(float(subset.std()), 4),
                "min":         round(float(subset.min()), 4),
                "max":         round(float(subset.max()), 4),
                "p25":         round(float(percentiles[0.25]), 4),
                "p75":         round(float(percentiles[0.75]), 4),
                "p90":         round(float(percentiles[0.90]), 4),
                "p99":         round(float(percentiles[0.99]), 4),
                "log1p_mean":  round(float(log_vals.mean()), 4),
                "log1p_std":   round(float(log_vals.std()), 4),
                "skewness":    round(float(subset.skew()), 4),
            }

        # KS test to check if Amount distributions differ significantly
        legit  = df.loc[df[target] == 0, self.AMOUNT_COL].dropna()
        fraud  = df.loc[df[target] == 1, self.AMOUNT_COL].dropna()
        if len(legit) > 0 and len(fraud) > 0:
            ks_stat, ks_p = ks_2samp(legit, fraud)
            result["ks_test_amount"] = {
                "statistic": round(float(ks_stat), 6),
                "p_value":   round(float(ks_p), 6),
                "distributions_differ": bool(ks_p < 0.05),
            }

        return {"fraud_amount_analysis": result}


# ============================================================================
# SECTION 5 — Statistical Separability Tests
# ============================================================================

class KolmogorovSmirnovAnalyzer(AnalysisComponent):
    """
    Two-sample KS test for every feature — fraud vs legitimate.

    The KS statistic measures the maximum distributional gap between the two
    classes.  Features with high KS statistics and p < 0.05 are the most
    promising for a downstream classifier.  This is especially relevant for
    PCA features since we cannot interpret them semantically.
    """

    def analyze(self, df: pd.DataFrame, config) -> dict:
        target = _get_target(df, config)
        logger.info("Analyzing: Kolmogorov-Smirnov Tests (fraud vs legit per feature)")
        if target is None:
            return {"ks_tests": "Target column missing."}

        exclude = {c for c in ["id", "Id", "ID"] if c in df.columns}
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                        if c not in exclude and c != target]

        legit_df  = df[df[target] == 0]
        fraud_df  = df[df[target] == 1]
        ks_results = {}

        for col in numeric_cols:
            a = legit_df[col].dropna().values
            b = fraud_df[col].dropna().values
            if len(a) < 5 or len(b) < 5:
                continue
            try:
                stat, p = ks_2samp(a, b)
                ks_results[col] = {
                    "ks_statistic": round(float(stat), 6),
                    "p_value":      round(float(p), 6),
                    "significant":  bool(p < 0.05),
                }
            except Exception as exc:
                ks_results[col] = f"Error: {exc}"

        # Sort by KS statistic descending
        sorted_results = dict(
            sorted(
                {k: v for k, v in ks_results.items() if isinstance(v, dict)}.items(),
                key=lambda x: x[1]["ks_statistic"],
                reverse=True,
            )
        )
        sorted_results.update({k: v for k, v in ks_results.items() if not isinstance(v, dict)})

        n_sig = sum(1 for v in ks_results.values() if isinstance(v, dict) and v["significant"])
        return {
            "ks_tests": sorted_results,
            "ks_significant_features_count": n_sig,
        }


class MannWhitneyAnalyzer(AnalysisComponent):
    """
    Mann-Whitney U test (non-parametric) for each feature — fraud vs legit.

    Complements the KS test: MW detects whether one class tends to have
    larger values than the other (stochastic dominance), while KS detects
    any distributional difference.  Together they give a fuller picture.

    Note: For large samples (550k rows), nearly every test will be
    significant; the rank-biserial correlation (effect size) is therefore
    more informative than the p-value alone.
    """

    # Subsample to this size per class to keep runtime reasonable
    MAX_SAMPLE = 10_000

    def analyze(self, df: pd.DataFrame, config) -> dict:
        target = _get_target(df, config)
        logger.info("Analyzing: Mann-Whitney U Tests (fraud vs legit per feature)")
        if target is None:
            return {"mann_whitney_tests": "Target column missing."}

        exclude = {c for c in ["id", "Id", "ID"] if c in df.columns}
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                        if c not in exclude and c != target]

        legit_full = df[df[target] == 0]
        fraud_full = df[df[target] == 1]

        rng = np.random.default_rng(42)
        legit_samp = legit_full.sample(min(self.MAX_SAMPLE, len(legit_full)), random_state=42)
        fraud_samp = fraud_full.sample(min(self.MAX_SAMPLE, len(fraud_full)), random_state=42)

        mw_results = {}
        for col in numeric_cols:
            a = legit_samp[col].dropna().values
            b = fraud_samp[col].dropna().values
            if len(a) < 5 or len(b) < 5:
                continue
            try:
                stat, p = mannwhitneyu(a, b, alternative="two-sided")
                # Rank-biserial correlation as effect size (ranges −1 to +1)
                n1, n2 = len(a), len(b)
                r = 1 - (2 * stat) / (n1 * n2)
                mw_results[col] = {
                    "u_statistic":    round(float(stat), 2),
                    "p_value":        round(float(p), 6),
                    "effect_size_r":  round(float(r), 4),
                    "significant":    bool(p < 0.05),
                }
            except Exception as exc:
                mw_results[col] = f"Error: {exc}"

        sorted_results = dict(
            sorted(
                {k: v for k, v in mw_results.items() if isinstance(v, dict)}.items(),
                key=lambda x: abs(x[1]["effect_size_r"]),
                reverse=True,
            )
        )
        return {"mann_whitney_tests": sorted_results}


class PointBiserialCorrelationAnalyzer(AnalysisComponent):
    """
    Point-biserial correlation between each numeric feature and the binary target.

    This is the correct correlation measure when one variable is continuous and
    the other is binary (0/1 fraud label).  It is mathematically equivalent to
    Pearson's r in this case, and gives a signed effect direction:
      +  → higher feature value associated with fraud
      −  → lower feature value associated with fraud

    Subsamples large datasets for speed while remaining statistically valid.
    """

    MAX_SAMPLE = 50_000

    def analyze(self, df: pd.DataFrame, config) -> dict:
        target = _get_target(df, config)
        logger.info("Analyzing: Point-Biserial Correlation vs fraud label")
        if target is None:
            return {"point_biserial_correlation": "Target column missing."}

        working = df.dropna().copy()
        if len(working) > self.MAX_SAMPLE:
            working = working.sample(self.MAX_SAMPLE, random_state=42)

        exclude = {"id", "Id", "ID", target}
        numeric_cols = [c for c in working.select_dtypes(include=[np.number]).columns
                        if c not in exclude]

        y = working[target].values
        pb_results = {}
        for col in numeric_cols:
            try:
                r, p = pointbiserialr(working[col].values, y)
                pb_results[col] = {
                    "r": round(float(r), 6),
                    "p_value": round(float(p), 6),
                    "significant": bool(p < 0.05),
                    "direction": "positive (↑ → fraud)" if r > 0 else "negative (↑ → legit)",
                }
            except Exception as exc:
                pb_results[col] = f"Error: {exc}"

        sorted_results = dict(
            sorted(
                {k: v for k, v in pb_results.items() if isinstance(v, dict)}.items(),
                key=lambda x: abs(x[1]["r"]),
                reverse=True,
            )
        )
        return {"point_biserial_correlation": sorted_results}


# ============================================================================
# SECTION 6 — Feature Importance & Redundancy
# ============================================================================

class MutualInformationAnalyzer(AnalysisComponent):
    """
    Non-linear mutual information between each feature and the fraud label.

    MI captures relationships missed by linear correlation.  Particularly
    useful for PCA features whose relationship with fraud may be non-monotonic.
    Subsamples for speed; results are stable at 50k+ rows.
    """

    MAX_SAMPLE = 50_000

    def analyze(self, df: pd.DataFrame, config) -> dict:
        target = _get_target(df, config)
        logger.info("Analyzing: Mutual Information Scores")
        if target is None:
            return {"mutual_information_scores": "Target column missing."}

        working = df.dropna().copy()
        if len(working) > self.MAX_SAMPLE:
            working = working.sample(self.MAX_SAMPLE, random_state=42)

        exclude = {"id", "Id", "ID", target}
        X = working[[c for c in working.select_dtypes(include=[np.number]).columns
                     if c not in exclude]]
        y = working[target].values

        try:
            mi = mutual_info_classif(X, y, random_state=42)
        except Exception as exc:
            logger.error(f"Mutual Information failed: {exc}")
            return {"mutual_information_scores": f"Error: {exc}"}

        mi_dict = dict(sorted(
            {col: round(float(s), 6) for col, s in zip(X.columns, mi)}.items(),
            key=lambda x: x[1], reverse=True
        ))
        return {"mutual_information_scores": mi_dict}


class SpearmanCorrelationAnalyzer(AnalysisComponent):
    """
    Spearman rank-correlation matrix for all numeric features.

    REPLACES MulticollinearityAnalyzer (VIF) for this dataset because:
      • V1–V28 are PCA components and are orthogonal by construction → VIF ≈ 1
        for all of them, which conveys no useful information.
      • Spearman detects any monotonic relationship (linear or not) and is
        robust to the heavy tails present in fraud datasets.
      • Research on this dataset (Mejia et al., 2024) found Spearman highlights
        residual monotonic links between V21/V22 missed by Pearson.

    Flags pairs with |r| > threshold as potentially redundant.
    """

    HIGH_CORR_THRESHOLD = 0.70
    MAX_SAMPLE = 20_000  # Spearman is O(n log n) but still slow at 550k

    def analyze(self, df: pd.DataFrame, config) -> dict:
        logger.info("Analyzing: Spearman Rank Correlation")
        target = config.target_column
        exclude = {c for c in ["id", "Id", "ID", target] if c in df.columns}
        numeric_df = df[[c for c in df.select_dtypes(include=[np.number]).columns
                         if c not in exclude]].dropna()

        if len(numeric_df) > self.MAX_SAMPLE:
            numeric_df = numeric_df.sample(self.MAX_SAMPLE, random_state=42)

        if numeric_df.shape[1] < 2:
            return {"spearman_correlation": "Not enough numeric columns."}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = numeric_df.corr(method="spearman").round(4)

        # Find highly correlated pairs
        high_pairs = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = abs(corr.iloc[i, j])
                if val >= self.HIGH_CORR_THRESHOLD:
                    high_pairs.append({
                        "feature_a": cols[i],
                        "feature_b": cols[j],
                        "spearman_r": round(float(corr.iloc[i, j]), 4),
                    })
        high_pairs.sort(key=lambda x: abs(x["spearman_r"]), reverse=True)

        return {
            "spearman_correlation_matrix": corr.to_dict(),
            "high_correlation_pairs": high_pairs,
            "high_correlation_threshold": self.HIGH_CORR_THRESHOLD,
        }


# ============================================================================
# SECTION 7 — Fraud-Specific Structural Analyzers
# ============================================================================

class PCAFeatureSeparabilityAnalyzer(AnalysisComponent):
    """
    Standardized mean difference (Cohen's d) per PCA feature between classes.

    Cohen's d = (mean_fraud − mean_legit) / pooled_std

    Interpretation for fraud detection:
      |d| < 0.2  →  negligible separation
      |d| 0.2–0.5 → small
      |d| 0.5–0.8 → medium
      |d| > 0.8  →  large (most useful for the classifier)

    Research shows V14, V10, V12, V4 are typically the most separable
    in this dataset — this analyzer will confirm or challenge that for
    the specific 2023 version.
    """

    def analyze(self, df: pd.DataFrame, config) -> dict:
        target = _get_target(df, config)
        logger.info("Analyzing: PCA Feature Separability (Cohen's d)")
        if target is None:
            return {"pca_separability": "Target column missing."}

        pca_cols = _pca_cols(df)
        if not pca_cols:
            return {"pca_separability": "No V1–V28 columns found."}

        legit = df[df[target] == 0]
        fraud = df[df[target] == 1]
        results = {}

        for col in pca_cols:
            a = legit[col].dropna().values
            b = fraud[col].dropna().values
            if len(a) < 2 or len(b) < 2:
                continue
            pooled_std = np.sqrt((a.std()**2 + b.std()**2) / 2)
            if pooled_std == 0:
                continue
            d = (b.mean() - a.mean()) / pooled_std
            abs_d = abs(d)
            magnitude = (
                "large"    if abs_d >= 0.8 else
                "medium"   if abs_d >= 0.5 else
                "small"    if abs_d >= 0.2 else
                "negligible"
            )
            results[col] = {
                "cohens_d":  round(float(d), 4),
                "abs_d":     round(float(abs_d), 4),
                "magnitude": magnitude,
                "fraud_mean":  round(float(b.mean()), 4),
                "legit_mean":  round(float(a.mean()), 4),
            }

        sorted_results = dict(sorted(results.items(), key=lambda x: x[1]["abs_d"], reverse=True))
        top_separable = [k for k, v in sorted_results.items() if v["magnitude"] in ("large", "medium")]

        return {
            "pca_separability": sorted_results,
            "top_separable_features": top_separable,
        }


class AmountScalingAnalyzer(AnalysisComponent):
    """
    Checks whether 'Amount' needs scaling and quantifies the scaling impact.

    V1–V28 are already PCA-standardized (zero mean, unit variance).  'Amount'
    is raw and right-skewed, which will distort distance-based models.  This
    analyzer computes the coefficient of variation, skewness, and compares
    raw vs log1p-transformed distributions to guide preprocessing decisions.
    """

    AMOUNT_COL = "Amount"

    def analyze(self, df: pd.DataFrame, config) -> dict:
        logger.info("Analyzing: Amount Scaling Requirements")
        if self.AMOUNT_COL not in df.columns:
            return {"amount_scaling": f"'{self.AMOUNT_COL}' column not found."}

        s = df[self.AMOUNT_COL].dropna()
        log_s = np.log1p(s)
        z_s = (s - s.mean()) / s.std()

        raw_range  = float(s.max() - s.min())
        pca_sample = df[[c for c in _pca_cols(df)]].iloc[:, 0].dropna() if _pca_cols(df) else None
        pca_range  = float(pca_sample.max() - pca_sample.min()) if pca_sample is not None else None

        return {
            "amount_scaling": {
                "raw": {
                    "mean":     round(float(s.mean()), 4),
                    "std":      round(float(s.std()), 4),
                    "min":      round(float(s.min()), 4),
                    "max":      round(float(s.max()), 4),
                    "range":    round(raw_range, 4),
                    "skewness": round(float(s.skew()), 4),
                    "cv_%":     round(float(s.std() / s.mean() * 100), 2) if s.mean() != 0 else None,
                },
                "log1p_transformed": {
                    "mean":     round(float(log_s.mean()), 4),
                    "std":      round(float(log_s.std()), 4),
                    "skewness": round(float(log_s.skew()), 4),
                },
                "z_score_normalized": {
                    "mean":  round(float(z_s.mean()), 6),
                    "std":   round(float(z_s.std()), 6),
                    "range": round(float(z_s.max() - z_s.min()), 4),
                },
                "pca_feature_range_for_comparison": round(pca_range, 4) if pca_range else "N/A",
                "scaling_recommendation": (
                    "Amount is right-skewed. Apply log1p transform then StandardScaler "
                    "to align it with the PCA features before modelling."
                    if abs(s.skew()) > 1 else
                    "Amount skewness is acceptable. StandardScaler alone should suffice."
                ),
            }
        }


class FraudTransactionPatternAnalyzer(AnalysisComponent):
    """
    Higher-level fraud pattern fingerprinting using the top separable features.

    Computes the 10th/90th percentile ranges for fraud vs legit on each feature
    to identify characteristic value ranges that 'smell like fraud'.  Useful
    for building simple rule-based pre-filters alongside the ML model.
    """

    def analyze(self, df: pd.DataFrame, config) -> dict:
        target = _get_target(df, config)
        logger.info("Analyzing: Fraud Transaction Patterns (percentile ranges)")
        if target is None:
            return {"fraud_patterns": "Target column missing."}

        exclude = {"id", "Id", "ID", target}
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                        if c not in exclude]

        legit = df[df[target] == 0]
        fraud = df[df[target] == 1]
        patterns = {}

        for col in numeric_cols:
            f = fraud[col].dropna()
            l = legit[col].dropna()
            if f.empty or l.empty:
                continue
            patterns[col] = {
                "fraud":  {
                    "p10": round(float(f.quantile(0.10)), 4),
                    "p25": round(float(f.quantile(0.25)), 4),
                    "p50": round(float(f.quantile(0.50)), 4),
                    "p75": round(float(f.quantile(0.75)), 4),
                    "p90": round(float(f.quantile(0.90)), 4),
                },
                "legit": {
                    "p10": round(float(l.quantile(0.10)), 4),
                    "p25": round(float(l.quantile(0.25)), 4),
                    "p50": round(float(l.quantile(0.50)), 4),
                    "p75": round(float(l.quantile(0.75)), 4),
                    "p90": round(float(l.quantile(0.90)), 4),
                },
            }

        return {"fraud_patterns": patterns}


class FeatureScaleConsistencyAnalyzer(AnalysisComponent):
    """
    Checks that all non-PCA features are on a comparable scale to V1–V28.

    PCA features have zero mean and unit variance by construction.  Any other
    feature (Amount, engineered features) with a wildly different scale will
    distort distance-based algorithms (KNN, SVM, neural nets).

    Returns a flag and recommended action for each out-of-scale column.
    """

    def analyze(self, df: pd.DataFrame, config) -> dict:
        logger.info("Analyzing: Feature Scale Consistency")
        target = config.target_column
        exclude = {c for c in ["id", "Id", "ID", target] if c in df.columns}
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                        if c not in exclude]

        pca_cols = _pca_cols(df)
        non_pca  = [c for c in numeric_cols if c not in pca_cols]

        # Reference stats from PCA features
        if pca_cols:
            pca_means = df[pca_cols].mean().abs()
            pca_stds  = df[pca_cols].std()
            ref_mean_abs = float(pca_means.mean())   # ≈ 0
            ref_std      = float(pca_stds.mean())    # ≈ 1
        else:
            ref_mean_abs, ref_std = 0.0, 1.0

        scale_report = {}
        for col in non_pca:
            s = df[col].dropna()
            col_mean = float(s.mean())
            col_std  = float(s.std())
            needs_scaling = abs(col_mean) > 5 * ref_std or col_std > 5 * ref_std
            scale_report[col] = {
                "mean":          round(col_mean, 4),
                "std":           round(col_std, 4),
                "pca_ref_mean":  round(ref_mean_abs, 4),
                "pca_ref_std":   round(ref_std, 4),
                "needs_scaling": needs_scaling,
                "recommendation": (
                    "Apply StandardScaler (and consider log1p first if skewed)."
                    if needs_scaling else "Scale is comparable to PCA features — OK."
                ),
            }

        return {
            "feature_scale_consistency": scale_report,
            "pca_feature_count": len(pca_cols),
            "non_pca_feature_count": len(non_pca),
        }