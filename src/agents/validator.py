import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import time
from datetime import datetime
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp, chi2_contingency
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel

from ..models import (
    FeatureSelectionResult,
    ValidationResult,
    ValidationCheck,
    ValidationIssue,
    CrossValidationResult,
    MLTaskType,
    PerformanceMode,
    DataType
)
from ..core import Config


class ValidatorDependencies(BaseModel):
    """Dependencies for Validator Agent"""
    df: pd.DataFrame
    selection_result: FeatureSelectionResult
    target: Optional[pd.Series] = None
    task_type: MLTaskType = MLTaskType.UNKNOWN
    temporal_column: Optional[str] = None
    performance_mode: PerformanceMode = PerformanceMode.MEDIUM
    
    class Config:
        arbitrary_types_allowed = True


validator_agent = Agent(
    Config.get_model_string(),
    deps_type=ValidatorDependencies,
    system_prompt="""You are the Validator Agent, a quality assurance expert.
    Your role is to validate feature quality and detect potential issues.
    Focus on:
    1. Detecting data leakage patterns
    2. Checking temporal consistency
    3. Validating feature stability across splits
    4. Identifying distribution shifts
    Always prioritize model reliability over performance.
    """
)


class ValidatorAgent:
    """Validator Agent for quality assurance"""
    
    def __init__(self):
        self.agent = validator_agent
    
    async def validate_features(
        self,
        df: pd.DataFrame,
        selection_result: FeatureSelectionResult,
        target: Optional[pd.Series] = None,
        task_type: MLTaskType = MLTaskType.UNKNOWN,
        temporal_column: Optional[str] = None,
        performance_mode: PerformanceMode = PerformanceMode.MEDIUM
    ) -> ValidationResult:
        """Validate feature quality and detect issues"""
        start_time = time.time()
        
        # Prepare validation data
        feature_df, selected_features = self._prepare_validation_data(df, selection_result)
        validation_state = self._initialize_validation_state()
        
        # Run all validation checks
        validation_state = self._run_validation_checks(
            df, feature_df, selected_features, target, task_type, 
            temporal_column, performance_mode, validation_state
        )
        
        # Generate final results
        return self._create_validation_result(
            validation_state, start_time, performance_mode
        )
    
    def _check_data_leakage(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        feature_names: List[str]
    ) -> List[str]:
        """Check for potential data leakage"""
        leakage_features = []
        
        if y is None:
            return leakage_features
        
        # Check for perfect correlation with target (only numeric features)
        numeric_features = X.select_dtypes(include=[np.number]).columns
        for feat in feature_names:
            if feat in numeric_features:
                correlation = X[feat].corr(y)
                if abs(correlation) > 0.99:
                    leakage_features.append(feat)
        
        # Check for features that might encode the target
        suspicious_patterns = ['target', 'label', 'y_', 'outcome', 'result']
        for feat in feature_names:
            if any(pattern in feat.lower() for pattern in suspicious_patterns):
                # Additional check: high correlation
                if feat in X.columns and abs(X[feat].corr(y)) > 0.9:
                    leakage_features.append(feat)
        
        return leakage_features
    
    def _check_temporal_consistency(
        self,
        df: pd.DataFrame,
        feature_df: pd.DataFrame,
        temporal_column: str,
        feature_names: List[str]
    ) -> List[ValidationIssue]:
        """Check if features are temporally consistent"""
        issues = []
        
        # Convert temporal column to datetime
        temporal_series = pd.to_datetime(df[temporal_column])
        
        # Split into early and late periods
        median_date = temporal_series.median()
        early_mask = temporal_series <= median_date
        late_mask = ~early_mask
        
        # Check distribution shifts between periods
        for feat in feature_names:
            if feat in feature_df.columns:
                early_data = feature_df.loc[early_mask, feat].dropna()
                late_data = feature_df.loc[late_mask, feat].dropna()
                
                if len(early_data) > 30 and len(late_data) > 30:
                    # KS test for distribution shift
                    statistic, p_value = ks_2samp(early_data, late_data)
                    
                    if p_value < 0.01:  # Significant shift
                        issues.append(ValidationIssue(
                            check_type=ValidationCheck.TEMPORAL_CONSISTENCY,
                            severity="medium",
                            feature_names=[feat],
                            description=f"Feature '{feat}' shows temporal distribution shift",
                            recommendation="Investigate if this shift is expected or indicates a problem",
                            metrics={'ks_statistic': float(statistic), 'p_value': float(p_value)}
                        ))
        
        return issues
    
    def _check_feature_stability(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        task_type: MLTaskType,
        performance_mode: PerformanceMode
    ) -> Dict[str, float]:
        """Check feature stability across data splits"""
        stability_scores = {}
        
        # Number of splits based on performance mode
        n_splits = 3 if performance_mode == PerformanceMode.FAST else 5
        
        if y is not None and task_type != MLTaskType.UNSUPERVISED:
            # Use stratified splits for classification
            if task_type in [MLTaskType.CLASSIFICATION, MLTaskType.MULTICLASS]:
                kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            else:
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            # Calculate feature statistics across splits
            numeric_features = X.select_dtypes(include=[np.number]).columns
            for feat in X.columns:
                if feat in numeric_features:
                    feat_stats = []
                    for train_idx, val_idx in kf.split(X, y):
                        train_mean = X.iloc[train_idx][feat].mean()
                        val_mean = X.iloc[val_idx][feat].mean()
                        if train_mean != 0:
                            stability = 1 - abs(val_mean - train_mean) / abs(train_mean)
                        else:
                            stability = 1.0 if val_mean == 0 else 0.0
                        feat_stats.append(stability)
                    
                    stability_scores[feat] = np.mean(feat_stats)
                else:
                    # For categorical features, check distribution stability
                    stability_scores[feat] = 0.9  # Default high stability
        else:
            # For unsupervised, use random splits
            for feat in X.columns:
                stability_scores[feat] = 0.9  # Default high stability
        
        return stability_scores
    
    def _check_multicollinearity(
        self,
        X: pd.DataFrame,
        threshold: float = 0.95
    ) -> List[List[str]]:
        """Check for multicollinearity among features"""
        # Only check numeric features
        numeric_df = X.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return []
        
        corr_matrix = numeric_df.corr().abs()
        
        # Find groups of highly correlated features
        multicollinear_groups = []
        checked = set()
        
        for i in range(len(corr_matrix.columns)):
            if corr_matrix.columns[i] in checked:
                continue
                
            group = [corr_matrix.columns[i]]
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    group.append(corr_matrix.columns[j])
                    checked.add(corr_matrix.columns[j])
            
            if len(group) > 1:
                multicollinear_groups.append(group)
        
        return multicollinear_groups
    
    def _check_class_imbalance(self, y: pd.Series) -> Optional[ValidationIssue]:
        """Check for class imbalance in target"""
        value_counts = y.value_counts()
        total = len(y)
        
        # Calculate imbalance ratio
        min_class_ratio = value_counts.min() / total
        max_class_ratio = value_counts.max() / total
        
        if min_class_ratio < 0.1:  # Less than 10% for minority class
            return ValidationIssue(
                check_type=ValidationCheck.CLASS_IMBALANCE,
                severity="high",
                feature_names=[],
                description=f"Severe class imbalance detected (minority: {min_class_ratio:.1%})",
                recommendation="Consider using SMOTE, class weights, or stratified sampling",
                metrics={
                    'minority_ratio': float(min_class_ratio),
                    'majority_ratio': float(max_class_ratio),
                    'class_distribution': value_counts.to_dict()
                }
            )
        elif min_class_ratio < 0.3:  # Less than 30% for minority class
            return ValidationIssue(
                check_type=ValidationCheck.CLASS_IMBALANCE,
                severity="medium",
                feature_names=[],
                description=f"Moderate class imbalance detected (minority: {min_class_ratio:.1%})",
                recommendation="Consider using class weights or balanced sampling",
                metrics={
                    'minority_ratio': float(min_class_ratio),
                    'majority_ratio': float(max_class_ratio)
                }
            )
        
        return None
    
    def _perform_cross_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: MLTaskType,
        performance_mode: PerformanceMode
    ) -> CrossValidationResult:
        """Perform cross-validation to estimate feature value"""
        # Only use numeric features for cross-validation
        numeric_features = X.select_dtypes(include=[np.number]).columns
        if len(numeric_features) == 0:
            # No numeric features to validate
            return CrossValidationResult(
                mean_score=0.0,
                std_score=0.0,
                fold_scores=[],
                feature_importance_stability={},
                warnings=["No numeric features available for cross-validation"]
            )
        
        X_numeric = X[numeric_features].fillna(0)
        
        # Choose model based on task type
        if task_type == MLTaskType.REGRESSION:
            model = RandomForestRegressor(
                n_estimators=30 if performance_mode == PerformanceMode.FAST else 50,
                random_state=42,
                n_jobs=-1
            )
            scoring = 'r2'
        else:
            model = RandomForestClassifier(
                n_estimators=30 if performance_mode == PerformanceMode.FAST else 50,
                random_state=42,
                n_jobs=-1
            )
            scoring = 'accuracy'
        
        # Perform cross-validation
        n_splits = 3 if performance_mode == PerformanceMode.FAST else 5
        cv_scores = cross_val_score(
            model, X_numeric, y, cv=n_splits, scoring=scoring, n_jobs=-1
        )
        
        # Get feature importance stability
        if task_type in [MLTaskType.CLASSIFICATION, MLTaskType.MULTICLASS]:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        importance_dict = {feat: [] for feat in numeric_features}
        for train_idx, _ in kf.split(X_numeric, y):
            model.fit(X_numeric.iloc[train_idx], y.iloc[train_idx])
            for feat, imp in zip(numeric_features, model.feature_importances_):
                importance_dict[feat].append(imp)
        
        # Calculate importance stability (coefficient of variation)
        importance_stability = {}
        for feat, imps in importance_dict.items():
            if np.mean(imps) > 0:
                importance_stability[feat] = 1 - (np.std(imps) / np.mean(imps))
            else:
                importance_stability[feat] = 0.0
        
        warnings = []
        if np.mean(cv_scores) < 0.5:
            warnings.append("Low cross-validation scores suggest poor feature quality")
        
        return CrossValidationResult(
            mean_score=float(np.mean(cv_scores)),
            std_score=float(np.std(cv_scores)),
            fold_scores=cv_scores.tolist(),
            feature_importance_stability=importance_stability,
            warnings=warnings
        )
    
    def _prepare_validation_data(self, df: pd.DataFrame, selection_result: FeatureSelectionResult) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare data for validation"""
        selected_features = [f.name for f in selection_result.selected_features]
        existing_features = [f for f in selected_features if f in df.columns]
        feature_df = df[existing_features].copy()
        return feature_df, selected_features
    
    def _initialize_validation_state(self) -> Dict:
        """Initialize validation state tracking"""
        return {
            'passed_checks': [],
            'failed_checks': [],
            'issues': [],
            'warnings': [],
            'recommendations': [],
            'leakage_features': [],
            'stability_scores': {},
            'cv_results': None
        }
    
    def _run_validation_checks(
        self, df: pd.DataFrame, feature_df: pd.DataFrame, selected_features: List[str],
        target: Optional[pd.Series], task_type: MLTaskType, temporal_column: Optional[str],
        performance_mode: PerformanceMode, validation_state: Dict
    ) -> Dict:
        """Run all validation checks and update state"""
        # Data leakage check
        self._run_leakage_check(feature_df, target, selected_features, validation_state)
        
        # Temporal consistency check
        if temporal_column and temporal_column in df.columns:
            self._run_temporal_check(df, feature_df, temporal_column, selected_features, validation_state)
        
        # Feature stability check
        self._run_stability_check(feature_df, target, task_type, performance_mode, validation_state)
        
        # Multicollinearity check
        self._run_multicollinearity_check(feature_df, validation_state)
        
        # Cross-validation check
        if target is not None and task_type != MLTaskType.UNSUPERVISED:
            self._run_cv_check(feature_df, target, task_type, performance_mode, validation_state)
        
        # Class imbalance check
        if target is not None and task_type in [MLTaskType.CLASSIFICATION, MLTaskType.MULTICLASS]:
            self._run_imbalance_check(target, validation_state)
        
        return validation_state
    
    def _run_leakage_check(
        self, feature_df: pd.DataFrame, target: Optional[pd.Series], 
        selected_features: List[str], validation_state: Dict
    ):
        """Run data leakage check"""
        leakage_features = self._check_data_leakage(feature_df, target, selected_features)
        validation_state['leakage_features'] = leakage_features
        
        if leakage_features:
            validation_state['failed_checks'].append(ValidationCheck.DATA_LEAKAGE)
            for feat in leakage_features:
                validation_state['issues'].append(ValidationIssue(
                    check_type=ValidationCheck.DATA_LEAKAGE,
                    severity="critical",
                    feature_names=[feat],
                    description=f"Feature '{feat}' shows signs of data leakage",
                    recommendation="Remove this feature or verify it's legitimately available at prediction time",
                    metrics={'correlation_with_target': 0.99}
                ))
        else:
            validation_state['passed_checks'].append(ValidationCheck.DATA_LEAKAGE)
    
    def _run_temporal_check(
        self, df: pd.DataFrame, feature_df: pd.DataFrame, temporal_column: str,
        selected_features: List[str], validation_state: Dict
    ):
        """Run temporal consistency check"""
        temporal_issues = self._check_temporal_consistency(
            df, feature_df, temporal_column, selected_features
        )
        if temporal_issues:
            validation_state['failed_checks'].append(ValidationCheck.TEMPORAL_CONSISTENCY)
            validation_state['issues'].extend(temporal_issues)
        else:
            validation_state['passed_checks'].append(ValidationCheck.TEMPORAL_CONSISTENCY)
    
    def _run_stability_check(
        self, feature_df: pd.DataFrame, target: Optional[pd.Series], 
        task_type: MLTaskType, performance_mode: PerformanceMode, validation_state: Dict
    ):
        """Run feature stability check"""
        stability_scores = self._check_feature_stability(feature_df, target, task_type, performance_mode)
        validation_state['stability_scores'] = stability_scores
        
        unstable_features = [feat for feat, score in stability_scores.items() if score < 0.7]
        if unstable_features:
            validation_state['failed_checks'].append(ValidationCheck.FEATURE_STABILITY)
            validation_state['issues'].append(ValidationIssue(
                check_type=ValidationCheck.FEATURE_STABILITY,
                severity="medium",
                feature_names=unstable_features,
                description=f"{len(unstable_features)} features show instability across data splits",
                recommendation="Consider removing unstable features or using regularization",
                metrics={'unstable_count': len(unstable_features)}
            ))
        else:
            validation_state['passed_checks'].append(ValidationCheck.FEATURE_STABILITY)
    
    def _run_multicollinearity_check(self, feature_df: pd.DataFrame, validation_state: Dict):
        """Run multicollinearity check"""
        multicollinear_groups = self._check_multicollinearity(feature_df)
        if multicollinear_groups:
            validation_state['failed_checks'].append(ValidationCheck.MULTICOLLINEARITY)
            for group in multicollinear_groups:
                validation_state['issues'].append(ValidationIssue(
                    check_type=ValidationCheck.MULTICOLLINEARITY,
                    severity="low",
                    feature_names=group,
                    description=f"Features are highly correlated: {', '.join(group)}",
                    recommendation="Consider removing redundant features from this group",
                    metrics={'correlation': 0.95}
                ))
        else:
            validation_state['passed_checks'].append(ValidationCheck.MULTICOLLINEARITY)
    
    def _run_cv_check(
        self, feature_df: pd.DataFrame, target: pd.Series, task_type: MLTaskType,
        performance_mode: PerformanceMode, validation_state: Dict
    ):
        """Run cross-validation check"""
        cv_results = self._perform_cross_validation(feature_df, target, task_type, performance_mode)
        validation_state['cv_results'] = cv_results
        
        if cv_results.std_score > 0.1:
            validation_state['warnings'].append("High variance in cross-validation scores suggests instability")
        validation_state['passed_checks'].append(ValidationCheck.CROSS_VALIDATION)
    
    def _run_imbalance_check(self, target: pd.Series, validation_state: Dict):
        """Run class imbalance check"""
        imbalance_issue = self._check_class_imbalance(target)
        if imbalance_issue:
            validation_state['failed_checks'].append(ValidationCheck.CLASS_IMBALANCE)
            validation_state['issues'].append(imbalance_issue)
        else:
            validation_state['passed_checks'].append(ValidationCheck.CLASS_IMBALANCE)
    
    def _create_validation_result(
        self, validation_state: Dict, start_time: float, performance_mode: PerformanceMode
    ) -> ValidationResult:
        """Create final validation result"""
        # Calculate quality score
        total_checks = len(validation_state['passed_checks']) + len(validation_state['failed_checks'])
        quality_score = len(validation_state['passed_checks']) / total_checks if total_checks > 0 else 1.0
        
        # Adjust for critical issues
        critical_issues = [i for i in validation_state['issues'] if i.severity == "critical"]
        if critical_issues:
            quality_score *= 0.5
        
        # Generate recommendations
        recommendations = self._generate_validation_recommendations(
            validation_state, quality_score, performance_mode
        )
        
        processing_time = time.time() - start_time
        
        return ValidationResult(
            passed_checks=validation_state['passed_checks'],
            failed_checks=validation_state['failed_checks'],
            issues=validation_state['issues'],
            cross_validation_results=validation_state['cv_results'],
            feature_stability_scores=validation_state['stability_scores'],
            leakage_risk_features=validation_state['leakage_features'],
            overall_quality_score=float(quality_score),
            recommendations=recommendations,
            warnings=validation_state['warnings'],
            processing_time_seconds=processing_time
        )
    
    def _generate_validation_recommendations(
        self, validation_state: Dict, quality_score: float, performance_mode: PerformanceMode
    ) -> List[str]:
        """Generate validation recommendations"""
        recommendations = []
        
        if validation_state['leakage_features']:
            recommendations.append(f"Critical: Remove {len(validation_state['leakage_features'])} features with data leakage")
        
        unstable_features = [feat for feat, score in validation_state['stability_scores'].items() if score < 0.7]
        if unstable_features:
            recommendations.append(f"Consider removing {len(unstable_features)} unstable features")
        
        if quality_score < 0.7:
            recommendations.append("Feature set needs significant improvement")
        elif quality_score < 0.9:
            recommendations.append("Address medium severity issues for better model reliability")
        else:
            recommendations.append("Feature set is ready with minor improvements")
        
        if performance_mode == PerformanceMode.FAST:
            recommendations.append("Consider running thorough validation for better confidence")
        
        return recommendations