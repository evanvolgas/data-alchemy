import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import time
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import spearmanr, chi2_contingency
from itertools import combinations
import warnings
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel

from ..models import (
    FeatureEngineeringResult,
    FeatureSelectionResult,
    SelectedFeature,
    SelectionMethod,
    MLTaskType,
    PerformanceMode
)
from ..core import Config


class CuratorDependencies(BaseModel):
    """Dependencies for Curator Agent"""
    df: pd.DataFrame
    engineering_result: FeatureEngineeringResult
    target: Optional[pd.Series] = None
    task_type: MLTaskType = MLTaskType.UNKNOWN
    performance_mode: PerformanceMode = PerformanceMode.MEDIUM
    
    class Config:
        arbitrary_types_allowed = True


curator_agent = Agent(
    Config.get_model_string(),
    deps_type=CuratorDependencies,
    system_prompt="""You are the Curator Agent, a feature selection expert.
    Your role is to select the most valuable features while removing redundancy.
    Focus on:
    1. Ranking features by importance using multiple methods
    2. Identifying and removing highly correlated features
    3. Balancing model performance with interpretability
    4. Providing clear rationale for selections
    """
)


class CuratorAgent:
    """Curator Agent for feature selection and ranking"""
    
    def __init__(self):
        self.agent = curator_agent
    
    async def select_features(
        self,
        df: pd.DataFrame,
        engineering_result: FeatureEngineeringResult,
        target: Optional[pd.Series] = None,
        task_type: MLTaskType = MLTaskType.UNKNOWN,
        performance_mode: PerformanceMode = PerformanceMode.MEDIUM
    ) -> FeatureSelectionResult:
        """Select and rank features based on importance and redundancy"""
        start_time = time.time()
        
        # Prepare data and initialize tracking
        feature_df, feature_scores, removed_features, selection_summary = self._prepare_features(
            df, engineering_result
        )
        feature_metadata = self._get_feature_metadata(engineering_result)
        
        # Remove low variance features
        feature_df, removed_features = self._apply_variance_filter(
            feature_df, removed_features, selection_summary
        )
        
        # Calculate importance scores
        if target is not None and task_type != MLTaskType.UNSUPERVISED:
            feature_scores = self._calculate_all_importance_scores(
                feature_df, target, task_type, feature_metadata, performance_mode, selection_summary
            )
        
        # Remove correlated features
        features_to_keep, correlation_matrix, correlation_clusters = self._remove_correlated_features(
            feature_df, feature_scores, performance_mode, removed_features
        )
        
        # Create selected features list
        selected_features = self._create_selected_features_list(
            features_to_keep, feature_scores, feature_metadata, correlation_clusters, engineering_result
        )
        
        # Calculate summaries and generate result
        return self._create_final_result(
            selected_features, removed_features, selection_summary, correlation_matrix,
            correlation_clusters, len([f.name for f in engineering_result.all_features]),
            start_time
        )
    
    def _variance_selection(self, df: pd.DataFrame, threshold: float = 0.01) -> List[str]:
        """Remove features with low variance"""
        # Only apply to numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_columns = [col for col in df.columns if col not in numeric_columns]
        
        if not numeric_columns:
            return non_numeric_columns
        
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(df[numeric_columns].fillna(0))
        selected_numeric = [col for col, selected in zip(numeric_columns, selector.get_support()) if selected]
        
        # Keep all non-numeric columns plus selected numeric columns
        return non_numeric_columns + selected_numeric
    
    def _mutual_information_scores(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: MLTaskType
    ) -> Dict[str, float]:
        """Calculate mutual information scores"""
        # Only calculate for numeric features
        numeric_features = X.select_dtypes(include=[np.number]).columns
        if len(numeric_features) == 0:
            return {}
        
        X_numeric = X[numeric_features].fillna(0)
        
        if task_type == MLTaskType.REGRESSION:
            scores = mutual_info_regression(X_numeric, y, random_state=42)
        else:
            scores = mutual_info_classif(X_numeric, y, random_state=42)
        
        return dict(zip(numeric_features, scores))
    
    def _random_forest_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: MLTaskType
    ) -> Dict[str, float]:
        """Calculate random forest feature importance"""
        # Only use numeric features
        numeric_features = X.select_dtypes(include=[np.number]).columns
        if len(numeric_features) == 0:
            return {}
        
        X_numeric = X[numeric_features].fillna(0)
        
        if task_type == MLTaskType.REGRESSION:
            rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        else:
            rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        
        rf.fit(X_numeric, y)
        return dict(zip(numeric_features, rf.feature_importances_))
    
    def _find_correlation_clusters(
        self,
        corr_matrix: pd.DataFrame,
        threshold: float = 0.95
    ) -> List[List[str]]:
        """Find clusters of highly correlated features"""
        # Create adjacency matrix
        adj_matrix = (corr_matrix > threshold).astype(int)
        np.fill_diagonal(adj_matrix.values, 0)
        
        # Find connected components
        visited = set()
        clusters = []
        
        for feature in corr_matrix.columns:
            if feature not in visited:
                cluster = self._dfs_correlation_cluster(
                    feature, adj_matrix, visited
                )
                if len(cluster) > 1:
                    clusters.append(cluster)
        
        return clusters
    
    def _dfs_correlation_cluster(
        self,
        feature: str,
        adj_matrix: pd.DataFrame,
        visited: set
    ) -> List[str]:
        """Depth-first search to find correlation cluster"""
        cluster = [feature]
        visited.add(feature)
        
        for other_feature in adj_matrix.columns:
            if other_feature not in visited and adj_matrix.loc[feature, other_feature] == 1:
                cluster.extend(
                    self._dfs_correlation_cluster(other_feature, adj_matrix, visited)
                )
        
        return cluster
    
    def _calculate_weighted_importance(self, scores: Dict[str, float]) -> float:
        """Calculate weighted importance score giving more weight to reliable methods"""
        if not scores:
            return 0.0
        
        # Weight different methods based on reliability
        weights = {
            'shap': 1.5,  # Most reliable
            'mutual_information': 1.2,
            'random_forest': 1.0,
            'permutation': 1.3,
            'chi_squared': 1.0,
            'target_encoding_corr': 0.9,  # Slightly lower due to leakage risk
            'target_encoding_mi': 0.9,
            'stability': 0.8,  # Modifier, not direct importance
            'interaction_strength': 1.1,
            'group_importance': 0.7,
            'default': 0.1
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for method, score in scores.items():
            weight = weights.get(method, 0.5)
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _get_feature_metadata(self, engineering_result: FeatureEngineeringResult) -> Dict[str, Dict[str, Any]]:
        """Extract metadata about features including their source and type"""
        metadata = {}
        
        # Original features
        for feat in engineering_result.original_features:
            metadata[feat.name] = {
                'type': 'original',
                'data_type': feat.dtype,
                'source': None,
                'group': feat.name
            }
        
        # Engineered features
        for feat in engineering_result.engineered_features:
            source = None
            group = feat.name
            
            # Determine source feature and group
            if '_frequency' in feat.name:
                source = feat.name.replace('_frequency', '')
                group = source
            elif '_is_' in feat.name:
                # One-hot encoded
                parts = feat.name.split('_is_')
                if parts:
                    source = parts[0]
                    group = source
            elif any(temporal in feat.name for temporal in ['_year', '_month', '_day', '_hour', '_dayofweek']):
                # Temporal features
                for temporal in ['_year', '_month', '_day', '_hour', '_dayofweek', '_sin', '_cos', '_weekend']:
                    if temporal in feat.name:
                        source = feat.name.replace(temporal, '')
                        group = source
                        break
            elif '_squared' in feat.name or '_cubed' in feat.name:
                source = feat.name.replace('_squared', '').replace('_cubed', '')
                group = source
            elif '_times_' in feat.name:
                parts = feat.name.split('_times_')
                group = 'interaction'
            elif '_over_' in feat.name:
                parts = feat.name.split('_over_')
                group = 'interaction'
            
            metadata[feat.name] = {
                'type': 'engineered',
                'data_type': feat.dtype,
                'source': source,
                'group': group,
                'engineering_method': feat.description
            }
        
        return metadata
    
    def _calculate_comprehensive_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: MLTaskType,
        feature_metadata: Dict[str, Dict[str, Any]],
        performance_mode: PerformanceMode
    ) -> Dict[str, Dict[str, float]]:
        """Calculate importance scores for all features including categorical"""
        scores = {}
        
        # Separate features by type
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Calculate numeric feature importance
        if numeric_features:
            numeric_scores = self._calculate_numeric_importance(
                X[numeric_features], y, task_type, performance_mode
            )
            scores.update(numeric_scores)
        
        # Calculate categorical feature importance
        if categorical_features and task_type != MLTaskType.UNSUPERVISED:
            categorical_scores = self._calculate_categorical_features_importance(
                X, y, task_type, categorical_features
            )
            scores.update(categorical_scores)
        
        # Add advanced importance scores
        scores = self._add_advanced_importance_scores(
            X, y, task_type, performance_mode, feature_metadata, scores
        )
        
        return scores
    
    def _calculate_categorical_importance(
        self,
        X_cat: pd.Series,
        y: pd.Series,
        task_type: MLTaskType
    ) -> Dict[str, float]:
        """Calculate importance for categorical features"""
        scores = {}
        
        # Chi-squared test for classification
        if task_type in [MLTaskType.CLASSIFICATION, MLTaskType.MULTICLASS]:
            try:
                # Create contingency table
                contingency = pd.crosstab(X_cat, y)
                chi2, p_value, _, _ = chi2_contingency(contingency)
                # Convert to score (higher is better)
                scores['chi_squared'] = 1 - p_value if p_value < 0.05 else 0.0
            except:
                pass
        
        # Mutual information with label encoding
        try:
            le = LabelEncoder()
            X_encoded = le.fit_transform(X_cat.fillna('missing'))
            
            if task_type == MLTaskType.REGRESSION:
                mi_score = mutual_info_regression(
                    X_encoded.reshape(-1, 1), y, random_state=42
                )[0]
            else:
                mi_score = mutual_info_classif(
                    X_encoded.reshape(-1, 1), y, random_state=42
                )[0]
            
            scores['mutual_information'] = float(mi_score)
        except:
            pass
        
        # Target encoding importance evaluation
        if task_type != MLTaskType.UNSUPERVISED:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    te = TargetEncoder(smooth="auto")
                    X_te = te.fit_transform(X_cat.fillna('missing').values.reshape(-1, 1), y)
                    
                    # Calculate correlation with target
                    if task_type == MLTaskType.REGRESSION:
                        corr = abs(np.corrcoef(X_te.flatten(), y)[0, 1])
                        scores['target_encoding_corr'] = float(corr) if not np.isnan(corr) else 0.0
                    else:
                        # For classification, use mutual information on target encoded values
                        mi_te = mutual_info_classif(X_te, y, random_state=42)[0]
                        scores['target_encoding_mi'] = float(mi_te)
            except Exception:
                pass
        
        return scores if scores else {'default': 0.3}
    
    def _aggregate_group_importance(
        self,
        feature_scores: Dict[str, Dict[str, float]],
        feature_metadata: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Aggregate importance scores by feature groups"""
        group_scores = {}
        group_counts = {}
        
        for feat, scores in feature_scores.items():
            if feat in feature_metadata:
                group = feature_metadata[feat]['group']
                # Calculate mean score for this feature
                feat_score = np.mean(list(scores.values()))
                
                if group not in group_scores:
                    group_scores[group] = 0
                    group_counts[group] = 0
                
                group_scores[group] += feat_score
                group_counts[group] += 1
        
        # Calculate average score per group
        for group in group_scores:
            if group_counts[group] > 0:
                group_scores[group] /= group_counts[group]
        
        return group_scores
    
    def _calculate_interaction_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: MLTaskType,
        feature_scores: Dict[str, Dict[str, float]],
        top_n: int = 10
    ) -> Dict[Tuple[str, str], float]:
        """Calculate importance of feature interactions"""
        # Get top features by importance
        if feature_scores:
            feature_importance = {
                feat: np.mean(list(scores.values())) 
                for feat, scores in feature_scores.items() 
                if feat in X.columns
            }
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
            top_feature_names = [f[0] for f in top_features]
        else:
            top_feature_names = X.columns[:top_n].tolist()
        
        interaction_scores = {}
        
        # Only check numeric features for interactions
        numeric_features = [f for f in top_feature_names if f in X.select_dtypes(include=[np.number]).columns]
        
        if len(numeric_features) < 2:
            return interaction_scores
        
        # Calculate interaction importance using a simple RF model
        for feat1, feat2 in combinations(numeric_features, 2):
            try:
                # Create interaction feature
                X_interaction = X[[feat1, feat2]].fillna(0)
                X_with_interaction = X_interaction.copy()
                X_with_interaction['interaction'] = X_interaction[feat1] * X_interaction[feat2]
                
                # Compare model performance with and without interaction
                if task_type == MLTaskType.REGRESSION:
                    model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)
                else:
                    model = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
                
                # Score without interaction
                score_without = cross_val_score(model, X_interaction, y, cv=3, n_jobs=-1).mean()
                
                # Score with interaction
                score_with = cross_val_score(model, X_with_interaction, y, cv=3, n_jobs=-1).mean()
                
                # Interaction importance is the improvement
                interaction_importance = max(0, score_with - score_without)
                
                if interaction_importance > 0.01:  # Threshold for meaningful interaction
                    interaction_scores[(feat1, feat2)] = float(interaction_importance)
            
            except Exception:
                continue
        
        return interaction_scores
    
    def _calculate_feature_stability(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: MLTaskType,
        feature_metadata: Dict[str, Dict[str, Any]],
        n_splits: int = 5
    ) -> Dict[str, float]:
        """Calculate feature importance stability across data subsets"""
        stability_scores = {}
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_features or len(X) < n_splits * 10:
            return stability_scores
        
        # Store importance scores across folds
        fold_importances = {feat: [] for feat in numeric_features}
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for train_idx, _ in kf.split(X):
            X_fold = X.iloc[train_idx][numeric_features].fillna(0)
            y_fold = y.iloc[train_idx]
            
            # Calculate importance for this fold
            if task_type == MLTaskType.REGRESSION:
                mi_scores = mutual_info_regression(X_fold, y_fold, random_state=42)
            else:
                mi_scores = mutual_info_classif(X_fold, y_fold, random_state=42)
            
            for feat, score in zip(numeric_features, mi_scores):
                fold_importances[feat].append(score)
        
        # Calculate stability as 1 - coefficient of variation
        for feat, scores in fold_importances.items():
            if len(scores) > 1 and np.mean(scores) > 0:
                cv = np.std(scores) / np.mean(scores)
                stability = 1 / (1 + cv)  # Maps CV to [0, 1]
                stability_scores[feat] = float(stability)
            else:
                stability_scores[feat] = 0.5  # Default medium stability
        
        return stability_scores
    
    def _evaluate_encoding_options(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        categorical_features: List[str],
        task_type: MLTaskType
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate different encoding options for categorical features"""
        encoding_scores = {}
        
        for cat_feat in categorical_features:
            if cat_feat not in X.columns:
                continue
                
            encoding_scores[cat_feat] = {}
            X_cat = X[cat_feat].fillna('missing')
            
            # 1. Frequency encoding
            try:
                freq_map = X_cat.value_counts(normalize=True).to_dict()
                X_freq = X_cat.map(freq_map).values.reshape(-1, 1)
                
                if task_type == MLTaskType.REGRESSION:
                    mi_score = mutual_info_regression(X_freq, y, random_state=42)[0]
                else:
                    mi_score = mutual_info_classif(X_freq, y, random_state=42)[0]
                
                encoding_scores[cat_feat]['frequency'] = float(mi_score)
            except Exception:
                pass
            
            # 2. Target encoding (with cross-validation to avoid leakage)
            if task_type != MLTaskType.UNSUPERVISED:
                try:
                    scores = []
                    kf = KFold(n_splits=3, shuffle=True, random_state=42)
                    
                    for train_idx, val_idx in kf.split(X):
                        # Fit encoder on train, evaluate on validation
                        te = TargetEncoder(smooth="auto")
                        X_train_cat = X_cat.iloc[train_idx].values.reshape(-1, 1)
                        y_train = y.iloc[train_idx]
                        X_val_cat = X_cat.iloc[val_idx].values.reshape(-1, 1)
                        y_val = y.iloc[val_idx]
                        
                        te.fit(X_train_cat, y_train)
                        X_val_encoded = te.transform(X_val_cat)
                        
                        if task_type == MLTaskType.REGRESSION:
                            corr = abs(np.corrcoef(X_val_encoded.flatten(), y_val)[0, 1])
                            scores.append(corr if not np.isnan(corr) else 0)
                        else:
                            mi = mutual_info_classif(X_val_encoded, y_val, random_state=42)[0]
                            scores.append(mi)
                    
                    encoding_scores[cat_feat]['target'] = float(np.mean(scores))
                except Exception:
                    pass
            
            # 3. One-hot encoding (only for low cardinality)
            if X_cat.nunique() <= 10:
                try:
                    # Create dummy variables
                    dummies = pd.get_dummies(X_cat, prefix=cat_feat)
                    
                    if task_type == MLTaskType.REGRESSION:
                        mi_scores = mutual_info_regression(dummies, y, random_state=42)
                    else:
                        mi_scores = mutual_info_classif(dummies, y, random_state=42)
                    
                    encoding_scores[cat_feat]['onehot'] = float(np.mean(mi_scores))
                except Exception:
                    pass
        
        return encoding_scores
    
    def _prepare_features(
        self, df: pd.DataFrame, engineering_result: FeatureEngineeringResult
    ) -> Tuple[pd.DataFrame, Dict, List, Dict]:
        """Prepare features and initialize tracking structures"""
        all_features = [f.name for f in engineering_result.all_features]
        existing_features = [f for f in all_features if f in df.columns]
        feature_df = df[existing_features].copy()
        
        feature_scores = {}
        removed_features = []
        selection_summary = {}
        
        return feature_df, feature_scores, removed_features, selection_summary
    
    def _apply_variance_filter(
        self, feature_df: pd.DataFrame, removed_features: List, selection_summary: Dict
    ) -> Tuple[pd.DataFrame, List]:
        """Apply variance threshold to remove constant features"""
        variance_features = self._variance_selection(feature_df)
        removed_by_variance = set(feature_df.columns) - set(variance_features)
        
        if removed_by_variance:
            for feat in removed_by_variance:
                removed_features.append({
                    'name': feat,
                    'reason': 'zero_variance',
                    'score': 0.0
                })
            selection_summary['removed_zero_variance'] = len(removed_by_variance)
            feature_df = feature_df[variance_features]
        
        return feature_df, removed_features
    
    def _calculate_all_importance_scores(
        self, feature_df: pd.DataFrame, target: pd.Series, task_type: MLTaskType,
        feature_metadata: Dict, performance_mode: PerformanceMode, selection_summary: Dict
    ) -> Dict:
        """Calculate importance scores for all features"""
        # Evaluate encoding options for categorical features
        categorical_features = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
        if categorical_features and performance_mode != PerformanceMode.FAST:
            encoding_evaluation = self._evaluate_encoding_options(
                feature_df, target, categorical_features, task_type
            )
            for cat_feat, encodings in encoding_evaluation.items():
                if cat_feat in feature_metadata:
                    feature_metadata[cat_feat]['encoding_scores'] = encodings
            selection_summary['features_with_encoding_eval'] = len(encoding_evaluation)
        
        # Calculate comprehensive importance
        return self._calculate_comprehensive_importance(
            feature_df, target, task_type, feature_metadata, performance_mode
        )
    
    def _remove_correlated_features(
        self, feature_df: pd.DataFrame, feature_scores: Dict, performance_mode: PerformanceMode,
        removed_features: List
    ) -> Tuple[set, pd.DataFrame, List]:
        """Remove highly correlated features"""
        # Correlation analysis (only for numeric features)
        numeric_df = feature_df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            correlation_matrix = numeric_df.corr().abs()
            correlation_clusters = self._find_correlation_clusters(
                correlation_matrix,
                threshold=0.9 if performance_mode == PerformanceMode.FAST else 0.95
            )
        else:
            correlation_matrix = pd.DataFrame()
            correlation_clusters = []
        
        # Remove redundant features from correlation clusters
        features_to_keep = set(feature_df.columns)
        for cluster_features in correlation_clusters:
            if len(cluster_features) > 1:
                best_feature = self._select_best_feature_from_cluster(
                    cluster_features, feature_scores
                )
                
                # Remove others
                for feat in cluster_features:
                    if feat != best_feature:
                        features_to_keep.discard(feat)
                        removed_features.append({
                            'name': feat,
                            'reason': 'high_correlation',
                            'correlated_with': best_feature,
                            'correlation': float(correlation_matrix.loc[feat, best_feature])
                        })
        
        return features_to_keep, correlation_matrix, correlation_clusters
    
    def _select_best_feature_from_cluster(self, cluster_features: List[str], feature_scores: Dict) -> str:
        """Select the best feature from a correlation cluster"""
        if feature_scores:
            cluster_scores = {
                feat: np.mean(list(feature_scores.get(feat, {}).values()))
                for feat in cluster_features
                if feat in feature_scores
            }
            if cluster_scores:
                return max(cluster_scores, key=cluster_scores.get)
        return cluster_features[0]
    
    def _create_selected_features_list(
        self, features_to_keep: set, feature_scores: Dict, feature_metadata: Dict,
        correlation_clusters: List, engineering_result: FeatureEngineeringResult
    ) -> List[SelectedFeature]:
        """Create the final list of selected features"""
        selected_features = []
        
        for rank, feat in enumerate(features_to_keep):
            importance, methods = self._calculate_feature_importance(feat, feature_scores, feature_metadata)
            cluster_id = self._find_feature_cluster(feat, correlation_clusters)
            interpretability = self._calculate_interpretability_score(feat, engineering_result)
            
            selected_features.append(SelectedFeature(
                name=feat,
                importance_score=float(importance),
                selection_methods=[SelectionMethod(m) for m in methods if m in [e.value for e in SelectionMethod]],
                rank=rank + 1,
                correlation_cluster=cluster_id,
                redundancy_score=0.0,
                interpretability_score=float(interpretability)
            ))
        
        # Sort by importance and update ranks
        selected_features.sort(key=lambda x: x.importance_score, reverse=True)
        for i, feat in enumerate(selected_features):
            feat.rank = i + 1
        
        return selected_features
    
    def _calculate_feature_importance(
        self, feat: str, feature_scores: Dict, feature_metadata: Dict
    ) -> Tuple[float, List[str]]:
        """Calculate importance score and methods for a feature"""
        if feat in feature_scores:
            importance = self._calculate_weighted_importance(feature_scores[feat])
            methods = list(feature_scores[feat].keys())
        else:
            # Try to get group-based importance
            if feat in feature_metadata:
                group = feature_metadata[feat]['group']
                group_scores = self._aggregate_group_importance(feature_scores, feature_metadata)
                if group in group_scores:
                    importance = group_scores[group]
                    methods = ['group_importance']
                else:
                    importance = 0.1
                    methods = ['default']
            else:
                importance = 0.1
                methods = ['default']
        
        return importance, methods
    
    def _find_feature_cluster(self, feat: str, correlation_clusters: List) -> Optional[int]:
        """Find which correlation cluster a feature belongs to"""
        for idx, cluster in enumerate(correlation_clusters):
            if feat in cluster:
                return idx
        return None
    
    def _calculate_interpretability_score(self, feat: str, engineering_result: FeatureEngineeringResult) -> float:
        """Calculate interpretability score for a feature"""
        original_features = [f for f in engineering_result.original_features if f.name == feat]
        if original_features:
            return 1.0  # Original features are most interpretable
        elif 'squared' in feat or 'cubed' in feat:
            return 0.7  # Polynomial features
        elif 'times' in feat or 'over' in feat:
            return 0.6  # Interactions
        elif 'log' in feat or 'sqrt' in feat:
            return 0.8  # Simple transforms
        else:
            return 0.5  # Complex features
    
    def _create_final_result(
        self, selected_features: List[SelectedFeature], removed_features: List,
        selection_summary: Dict, correlation_matrix: pd.DataFrame, correlation_clusters: List,
        original_count: int, start_time: float
    ) -> FeatureSelectionResult:
        """Create the final feature selection result"""
        # Calculate correlation summary
        if not correlation_matrix.empty:
            correlation_summary = {
                'mean_correlation': float(correlation_matrix.mean().mean()),
                'max_correlation': float(correlation_matrix[correlation_matrix < 1].max().max()) 
                    if (correlation_matrix < 1).any().any() else 0.0,
                'correlation_clusters': len(correlation_clusters)
            }
        else:
            correlation_summary = {
                'mean_correlation': 0.0,
                'max_correlation': 0.0,
                'correlation_clusters': 0
            }
        
        # Update selection summary
        selection_summary['removed_correlated'] = len([
            f for f in removed_features if f.get('reason') == 'high_correlation'
        ])
        
        # Calculate dimensionality reduction
        dimensionality_reduction = {
            'original_features': original_count,
            'selected_features': len(selected_features),
            'reduction_percentage': (1 - len(selected_features) / original_count) * 100 if original_count else 0
        }
        
        # Estimate performance impact
        performance_impact = {
            'expected_accuracy_retention': min(0.95, 0.85 + len(selected_features) / max(1, original_count) * 0.1),
            'training_speedup': original_count / max(1, len(selected_features)),
            'interpretability_gain': np.mean([f.interpretability_score for f in selected_features])
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(selected_features, dimensionality_reduction, correlation_summary)
        
        processing_time = time.time() - start_time
        
        return FeatureSelectionResult(
            selected_features=selected_features,
            removed_features=removed_features,
            selection_summary=selection_summary,
            correlation_matrix_summary=correlation_summary,
            dimensionality_reduction=dimensionality_reduction,
            performance_impact_estimate=performance_impact,
            recommendations=recommendations,
            processing_time_seconds=processing_time
        )
    
    def _generate_recommendations(
        self, selected_features: List, dimensionality_reduction: Dict, correlation_summary: Dict
    ) -> List[str]:
        """Generate recommendations based on feature selection results"""
        recommendations = []
        
        if len(selected_features) > 50:
            recommendations.append(f"Consider further reduction - {len(selected_features)} features may still be too many")
        if dimensionality_reduction['reduction_percentage'] > 80:
            recommendations.append("Aggressive reduction applied - consider validating with cross-validation")
        if correlation_summary['max_correlation'] > 0.95:
            recommendations.append("Some features still highly correlated - consider lower threshold")
        
        return recommendations
    
    def _calculate_numeric_importance(
        self, X_numeric: pd.DataFrame, y: pd.Series, task_type: MLTaskType, 
        performance_mode: PerformanceMode
    ) -> Dict[str, Dict[str, float]]:
        """Calculate importance scores for numeric features"""
        scores = {}
        X_filled = X_numeric.fillna(0)
        
        # Mutual information
        if task_type == MLTaskType.REGRESSION:
            mi_scores = mutual_info_regression(X_filled, y, random_state=42)
        else:
            mi_scores = mutual_info_classif(X_filled, y, random_state=42)
        
        for feat, score in zip(X_numeric.columns, mi_scores):
            scores[feat] = {'mutual_information': float(score)}
        
        # Random forest importance
        if performance_mode != PerformanceMode.FAST:
            rf_scores = self._calculate_rf_importance(X_filled, y, task_type, performance_mode)
            for feat, rf_score_dict in rf_scores.items():
                scores[feat].update(rf_score_dict)
        
        return scores
    
    def _calculate_rf_importance(
        self, X_numeric: pd.DataFrame, y: pd.Series, task_type: MLTaskType,
        performance_mode: PerformanceMode
    ) -> Dict[str, Dict[str, float]]:
        """Calculate random forest and related importance scores"""
        rf_scores = {}
        
        if task_type == MLTaskType.REGRESSION:
            rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        else:
            rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        
        rf.fit(X_numeric, y)
        
        for feat, importance in zip(X_numeric.columns, rf.feature_importances_):
            rf_scores[feat] = {'random_forest': float(importance)}
        
        # Add permutation and SHAP importance for thorough mode
        if performance_mode == PerformanceMode.THOROUGH:
            perm_scores = self._calculate_permutation_importance(rf, X_numeric, y)
            shap_scores = self._calculate_shap_importance(rf, X_numeric)
            
            for feat in X_numeric.columns:
                if feat in perm_scores:
                    rf_scores[feat]['permutation'] = perm_scores[feat]
                if feat in shap_scores:
                    rf_scores[feat]['shap'] = shap_scores[feat]
        
        return rf_scores
    
    def _calculate_permutation_importance(
        self, model, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """Calculate permutation importance scores"""
        try:
            perm_importance = permutation_importance(
                model, X, y, n_repeats=5, random_state=42, n_jobs=-1
            )
            return {feat: float(importance) for feat, importance 
                   in zip(X.columns, perm_importance.importances_mean)}
        except Exception:
            return {}
    
    def _calculate_shap_importance(self, model, X: pd.DataFrame) -> Dict[str, float]:
        """Calculate SHAP importance scores"""
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # For multiclass, average across classes
            if isinstance(shap_values, list):
                shap_importance = np.abs(shap_values).mean(axis=0).mean(axis=0)
            else:
                shap_importance = np.abs(shap_values).mean(axis=0)
            
            return {feat: float(importance) for feat, importance 
                   in zip(X.columns, shap_importance)}
        except (ImportError, Exception):
            return {}
    
    def _calculate_categorical_features_importance(
        self, X: pd.DataFrame, y: pd.Series, task_type: MLTaskType, 
        categorical_features: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate importance for all categorical features"""
        scores = {}
        for feat in categorical_features:
            cat_scores = self._calculate_categorical_importance(X[feat], y, task_type)
            scores[feat] = cat_scores
        return scores
    
    def _add_advanced_importance_scores(
        self, X: pd.DataFrame, y: pd.Series, task_type: MLTaskType,
        performance_mode: PerformanceMode, feature_metadata: Dict,
        scores: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Add interaction, stability, and group importance scores"""
        # Calculate interaction importance for top features
        if performance_mode != PerformanceMode.FAST and len(X.columns) < 50:
            interaction_scores = self._calculate_interaction_importance(
                X, y, task_type, scores, top_n=10
            )
            # Add interaction scores to main scores
            for feat_pair, score in interaction_scores.items():
                feat1, feat2 = feat_pair
                if feat1 in scores:
                    scores[feat1]['interaction_strength'] = max(
                        scores[feat1].get('interaction_strength', 0), score
                    )
                if feat2 in scores:
                    scores[feat2]['interaction_strength'] = max(
                        scores[feat2].get('interaction_strength', 0), score
                    )
        
        # Calculate feature stability across subsets
        if performance_mode == PerformanceMode.THOROUGH:
            stability_scores = self._calculate_feature_stability(
                X, y, task_type, feature_metadata
            )
            for feat, stability in stability_scores.items():
                if feat in scores:
                    scores[feat]['stability'] = stability
        
        # Group-based importance aggregation
        group_scores = self._aggregate_group_importance(scores, feature_metadata)
        
        # Apply group scores to encoded features without direct scores
        for feat in X.columns:
            if feat not in scores and feat in feature_metadata:
                group = feature_metadata[feat]['group']
                if group in group_scores:
                    scores[feat] = {'group_importance': group_scores[group]}
        
        return scores