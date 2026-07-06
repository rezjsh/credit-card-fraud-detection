import warnings
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, fbeta_score, precision_recall_curve, precision_score, recall_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold, train_test_split, learning_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from credit_card_fraud_detection.components.data_validation.interface import ValidationStrategy
from credit_card_fraud_detection.utils.logging_setup import logger
from credit_card_fraud_detection.entity.config_entity import ValidationConfig
from credit_card_fraud_detection.components.data_validation.context import ValidationContext
from credit_card_fraud_detection.components.data_validation.interface import ValidationStrategy, ValidatorRegistry

_EXCLUDE_COLS = {"id", "Id", "ID"}

def _feature_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    feat_cols = [c for c in df.columns if c not in _EXCLUDE_COLS and c != target]
    return df[feat_cols], df[target]

def _default_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced", random_state=42)),
    ])

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "auprc": round(float(average_precision_score(y_true, y_prob)), 6),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
            "f2": round(float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)), 6),
        }
@ValidatorRegistry.register("SchemaValidator")
class SchemaValidator(ValidationStrategy):
    """Validates the schema of the input DataFrame against the expected configuration."""
    def analyze(self, context: ValidationContext) -> dict:
        config = context.config
        df = context.df
        target = config.target_column
        checks = {}
        all_pass = True

        def _fail(key: str, msg: str):
            nonlocal all_pass
            checks[key] = {"status": "FAIL", "detail": msg}
            all_pass = False

        def _pass(key: str, msg: str = "OK"):
            checks[key] = {"status": "PASS", "detail": msg}

        required_cols = [config.amount_column, config.target_column]
        missing_required = [c for c in required_cols if c not in df.columns]
        if missing_required: _fail("required_columns", f"Missing: {missing_required}")
        else: _pass("required_columns", "All required columns present")

        pca_missing = [f"{config.pca_prefix}{i}" for i in range(1, config.pca_count + 1) if f"{config.pca_prefix}{i}" not in df.columns]
        if pca_missing: _fail("pca_columns", f"Missing PCA columns: {pca_missing}")
        else: _pass("pca_columns", "All PCA columns present")

        if target in df.columns:
            actual_vals = set(df[target].dropna().unique())
            if actual_vals - set(config.allowed_classes):
                _fail("class_values", f"Unexpected values. Expected {config.allowed_classes}")
            else: _pass("class_values", "Valid classes")

        if len(df) < config.min_expected_rows:
            _fail("row_count", f"Only {len(df)} rows. Expected ≥ {config.min_expected_rows}")
        else: _pass("row_count", "Row count OK")

        return {"schema_validation": {"overall_status": "PASS" if all_pass else "FAIL", "checks": checks}}

@ValidatorRegistry.register("DuplicateCheckValidator")
class DuplicateCheckValidator(ValidationStrategy):
    """Checks for duplicate rows in the DataFrame."""
    def analyze(self, context: ValidationContext) -> dict:
        df = context.df
        # Exclude ID columns from duplication check
        feature_cols = [c for c in df.columns if c not in {"id", "Id", "ID"}]
        duplicate_count = df.duplicated(subset=feature_cols).sum()
        duplicate_ratio = duplicate_count / len(df)
        
        status = "WARN" if duplicate_ratio > 0.01 else "PASS"
        if duplicate_ratio > 0.05:
            status = "FAIL"
            
        return {
            "duplicate_check": {
                "status": status,
                "duplicate_count": int(duplicate_count),
                "duplicate_ratio": float(duplicate_ratio)
            }
        }
    
@ValidatorRegistry.register("ImbalanceValidator")
class ImbalanceValidator(ValidationStrategy):
    """Checks for class imbalance in the target column."""
    def analyze(self, context: ValidationContext) -> dict:
        df = context.df
        config = context.config
        if config.target_column not in df.columns:
            return {"imbalance_check": {"status": "SKIPPED"}}
            
        value_counts = df[config.target_column].value_counts(normalize=True)
        minority_ratio = value_counts.min()
        
        return {
            "imbalance_check": {
                "minority_class_ratio": float(minority_ratio),
                "is_highly_imbalanced": bool(minority_ratio < 0.05)
            }
        }

@ValidatorRegistry.register("DataLeakageValidator")
class DataLeakageValidator(ValidationStrategy):
    """Checks for data leakage in the input DataFrame.
    This validator looks for features that have an unusually high predictive power for the target variable, which may indicate that they are leaking information from the future or from the target itself.
    """
    def analyze(self, context: ValidationContext) -> dict:
        config = context.config
        leaks = []
        
        # Access the raw df just to check for ID columns
        id_cols = [c for c in _EXCLUDE_COLS if c in context.df.columns]
        if id_cols: leaks.append(f"ID columns found: {id_cols}")
        
        # ── Retrieve the pre-computed, cached sample ──
        sample = context.get_cv_sample()
        X, y = _feature_target(sample, config.target_column)
        
        for col in X.columns:
            if average_precision_score(y, X[col]) > 0.99:
                leaks.append(f"Feature {col} AUPRC > 0.99. Potential leak.")
                
        return {"leakage_validation": {"status": "FAIL" if leaks else "PASS", "leaks": leaks}}

@ValidatorRegistry.register("StratifiedSplitValidator")
class StratifiedSplitValidator(ValidationStrategy):
    """Validates that a stratified train/validation/test split maintains the class distribution of the target variable."""
    def analyze(self, context: ValidationContext) -> dict:
        config = context.config
        df = context.df
        X, y = _feature_target(df, config.target_column)
        
        X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=config.test_size, stratify=y, random_state=42)
        val_ratio = config.val_size / (config.train_size + config.val_size)
        _, _, y_train, y_val = train_test_split(X_tv, y_tv, test_size=val_ratio, stratify=y_tv, random_state=42)
        
        full_rate = float(y.mean())
        pass_train = abs(y_train.mean() - full_rate) <= config.split_tolerance
        
        return {"stratified_split": {"status": "PASS" if pass_train else "FAIL", "tolerance": config.split_tolerance}}

@ValidatorRegistry.register("CrossValidationValidator")
class CrossValidationValidator(ValidationStrategy):
    """Performs cross-validation on the input DataFrame and reports the mean AUPRC score across folds. This helps assess the model's generalization performance."""
    def analyze(self, context: ValidationContext) -> dict:
        config = context.config
        df = context.df
        sample = df.dropna().sample(min(config.cv_max_rows, len(df)), random_state=42)
        X, y = _feature_target(sample, config.target_column)
        
        skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=42)
        model = _default_model()
        auprc_scores = []
        
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model.fit(X_tr, y_tr)
            y_prob = model.predict_proba(X_val)[:, 1]
            auprc_scores.append(average_precision_score(y_val, y_prob))
            
        return {"cross_validation": {"n_folds": config.n_splits, "mean_auprc": float(np.mean(auprc_scores))}}

@ValidatorRegistry.register("RepeatedCVStabilityValidator")
class RepeatedCVStabilityValidator(ValidationStrategy):
    """Performs repeated cross-validation to assess the stability of the model's performance across multiple runs. It calculates the coefficient of variation (CV) of the AUPRC scores to determine if the model is stable or unstable."""
    def analyze(self, context: ValidationContext) -> dict:
        config = context.config
        df = context.df
        sample = context.get_stability_sample()
        X, y = _feature_target(sample, config.target_column)
        
        rskf = RepeatedStratifiedKFold(n_splits=config.n_splits, n_repeats=config.n_repeats, random_state=42)
        model = _default_model()
        scores = []
        
        for train_idx, val_idx in rskf.split(X, y):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            scores.append(average_precision_score(y.iloc[val_idx], model.predict_proba(X.iloc[val_idx])[:, 1]))
            
        cv_pct = float(np.std(scores) / np.mean(scores) * 100) if np.mean(scores) > 0 else 0
        return {"cv_stability": {"cv_pct": cv_pct, "status": "STABLE" if cv_pct < 5 else "UNSTABLE"}}

@ValidatorRegistry.register("ThresholdTuner")
class ThresholdTuner(ValidationStrategy):
    """Tunes the decision threshold for the model to optimize the F2 score, which is particularly useful in imbalanced classification scenarios."""
    def analyze(self, context: ValidationContext) -> dict:
        config = context.config
        df = context.df
        sample = df.dropna().sample(min(config.cv_max_rows, len(df)), random_state=42)
        X, y = _feature_target(sample, config.target_column)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config.test_size, stratify=y, random_state=42)
        
        model = _default_model().fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)
        
        best_f2 = {"threshold": None, "f2": -1}
        for thresh, prec, rec in zip(thresholds, precisions[:-1], recalls[:-1]):
            denom = 4 * prec + rec
            f2 = (5 * prec * rec) / denom if denom > 0 else 0
            if f2 > best_f2["f2"]: best_f2 = {"threshold": float(thresh), "f2": float(f2)}
            
        return {"threshold_tuning": {"optimal_f2_threshold": best_f2["threshold"]}}

@ValidatorRegistry.register("ConfusionMatrixValidator")
class ConfusionMatrixValidator(ValidationStrategy):
    """Analyzes the confusion matrix of the model's predictions to provide insights into its performance."""
    def analyze(self, context: ValidationContext) -> dict:
        config = context.config
        df = context.df
        sample = df.dropna().sample(min(config.cv_max_rows, len(df)), random_state=42)
        X, y = _feature_target(sample, config.target_column)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.test_size, stratify=y, random_state=42)
        
        model = _default_model().fit(X_train, y_train)
        y_pred = (model.predict_proba(X_test)[:, 1] >= config.decision_threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fn_cost = float(df[config.amount_column].mean()) if config.amount_column in df.columns else 100.0
        total_cost = fn * fn_cost + fp * config.fp_cost_eur
        
        return {"cost_analysis": {"model_cost_eur": total_cost, "fp_count": int(fp), "fn_count": int(fn)}}

@ValidatorRegistry.register("CalibrationValidator")
class CalibrationValidator(ValidationStrategy):
    """Evaluates the calibration of the model's predicted probabilities using the Brier score and calibration curve. A well-calibrated model's predicted probabilities should reflect the true likelihood of the positive class."""
    def analyze(self, context: ValidationContext) -> dict:
        config = context.config
        df = context.df
        sample = df.dropna().sample(min(config.stability_max_rows, len(df)), random_state=42)
        X, y = _feature_target(sample, config.target_column)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.test_size, stratify=y, random_state=42)
        
        model = _default_model().fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        brier = brier_score_loss(y_test, y_prob)
        
        return {"calibration": {"brier_score": float(brier), "bins": config.calibration_bins}}

@ValidatorRegistry.register("LearningCurveValidator")
class LearningCurveValidator(ValidationStrategy):
    """Generates a learning curve to assess the model's performance as a function of the training set size. It helps identify if the model is overfitting or underfitting."""
    def analyze(self, context: ValidationContext) -> dict:
        config = context.config
        df = context.df
        sample = df.dropna().sample(min(config.stability_max_rows, len(df)), random_state=42)
        X, y = _feature_target(sample, config.target_column)
        cv = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=42)
        
        _, train_scores, val_scores = learning_curve(
            _default_model(), X, y, cv=cv, scoring="average_precision", n_jobs=-1
        )
        gap = float(train_scores[-1].mean() - val_scores[-1].mean())
        return {"learning_curve": {"train_val_gap": gap, "diagnosis": "OVERFIT" if gap > 0.15 else "OK"}}

@ValidatorRegistry.register("FeatureImportanceValidator")
class FeatureImportanceValidator(ValidationStrategy):
    """Computes feature importance using permutation importance on a trained Random Forest model. It identifies the top features that contribute most to the model's predictions."""
    def analyze(self, context: ValidationContext) -> dict:
        config = context.config
        df = context.df
        sample = df.dropna().sample(min(config.importance_max_rows, len(df)), random_state=42)
        X, y = _feature_target(sample, config.target_column)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.test_size, stratify=y, random_state=42)
        
        rf = RandomForestClassifier(n_estimators=config.importance_estimators, n_jobs=-1, random_state=42)
        rf.fit(X_train, y_train)
        perm = permutation_importance(rf, X_test, y_test, n_repeats=config.n_repeats, scoring="average_precision", random_state=42)
        
        top_features = pd.Series(perm.importances_mean, index=X.columns).nlargest(3).to_dict()
        return {"feature_importance": {"top_features": top_features}}

@ValidatorRegistry.register("OverfitDetector")
class OverfitDetector(ValidationStrategy):
    """Detects overfitting by comparing the model's performance on the training set versus the test set using the average precision score."""
    def analyze(self, context: ValidationContext) -> dict:
        config = context.config
        df = context.df
        sample = df.dropna().sample(min(config.stability_max_rows, len(df)), random_state=42)
        X, y = _feature_target(sample, config.target_column)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.test_size, stratify=y, random_state=42)
        
        model = _default_model().fit(X_train, y_train)
        train_score = average_precision_score(y_train, model.predict_proba(X_train)[:, 1])
        test_score = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])
        return {"overfit": {"gap": float(train_score - test_score)}}

@ValidatorRegistry.register("NoisePerturbationValidator")
class NoisePerturbationValidator(ValidationStrategy):
    """Adds noise to the test set and evaluates the model's performance to assess its robustness."""
    def analyze(self, context: ValidationContext) -> dict:
        config = context.config
        df = context.df
        sample = df.dropna().sample(min(config.importance_max_rows, len(df)), random_state=42)
        X, y = _feature_target(sample, config.target_column)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.test_size, stratify=y, random_state=42)
        
        model = _default_model().fit(X_train, y_train)
        base_auprc = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])
        
        results = {}
        for sigma in config.noise_sigmas:
            noise = np.random.normal(0, sigma * X_test.std(), X_test.shape)
            noisy_auprc = average_precision_score(y_test, model.predict_proba(X_test + noise)[:, 1])
            results[f"sigma_{sigma}"] = float(base_auprc - noisy_auprc)
            
        return {"noise_perturbation": {"auprc_drops": results}}

@ValidatorRegistry.register("SubsampleStabilityValidator")
class SubsampleStabilityValidator(ValidationStrategy):
    """Evaluates the stability of the model's performance when trained on different fractions of the training data."""
    def analyze(self, context: ValidationContext) -> dict:
        config = context.config
        df = context.df
        sample = df.dropna().sample(min(config.stability_max_rows, len(df)), random_state=42)
        X, y = _feature_target(sample, config.target_column)
        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=config.test_size, stratify=y, random_state=42)
        
        results = {}
        for frac in config.subsample_fractions:
            idx = pd.Series(y_train_full).sample(frac=frac, random_state=42).index
            model = _default_model().fit(X_train_full.loc[idx], y_train_full.loc[idx])
            results[f"frac_{frac}"] = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])
            
        return {"subsample_stability": {"auprc_by_fraction": results}}