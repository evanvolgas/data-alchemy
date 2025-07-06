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
    "openai:gpt-4o-mini",
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
        
        # Get all feature names that actually exist in the dataframe
        all_features = [f.name for f in engineering_result.all_features]
        existing_features = [f for f in all_features if f in df.columns]
        feature_df = df[existing_features].copy()
        
        # Initialize tracking
        feature_scores = {}
        removed_features = []
        selection_summary = {}
        feature_metadata = self._get_feature_metadata(engineering_result)
        
        # 1. Variance threshold (remove constant features)
        variance_features = self._variance_selection(feature_df)
        removed_by_variance = set(all_features) - set(variance_features)
        if removed_by_variance:
            for feat in removed_by_variance:
                removed_features.append({
                    'name': feat,
                    'reason': 'zero_variance',
                    'score': 0.0
                })
            selection_summary['removed_zero_variance'] = len(removed_by_variance)
            feature_df = feature_df[variance_features]
        
        # 2. Calculate importance scores if target is provided
        if target is not None and task_type != MLTaskType.UNSUPERVISED:
            # Evaluate encoding options for categorical features
            categorical_features = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
            if categorical_features and performance_mode != PerformanceMode.FAST:
                encoding_evaluation = self._evaluate_encoding_options(
                    feature_df, target, categorical_features, task_type
                )
                # Store encoding evaluation results in feature metadata
                for cat_feat, encodings in encoding_evaluation.items():
                    if cat_feat in feature_metadata:
                        feature_metadata[cat_feat]['encoding_scores'] = encodings
                selection_summary['features_with_encoding_eval'] = len(encoding_evaluation)
            
            # Calculate importance for all features (including categorical)
            all_importance_scores = self._calculate_comprehensive_importance(
                feature_df, target, task_type, feature_metadata, performance_mode
            )
            
            # Merge scores
            for feat, scores in all_importance_scores.items():
                feature_scores[feat] = scores
        
        # 3. Correlation analysis (only for numeric features)
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
        
        # 4. Remove redundant features from correlation clusters
        features_to_keep = set(feature_df.columns)
        for cluster_id, cluster_features in enumerate(correlation_clusters):
            if len(cluster_features) > 1:
                # Keep the feature with highest importance score
                if feature_scores:
                    cluster_scores = {
                        feat: np.mean(list(feature_scores.get(feat, {}).values()))
                        for feat in cluster_features
                        if feat in feature_scores
                    }
                    if cluster_scores:
                        best_feature = max(cluster_scores, key=cluster_scores.get)
                    else:
                        best_feature = cluster_features[0]
                else:
                    # If no scores, keep the first one
                    best_feature = cluster_features[0]
                
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
        
        selection_summary['removed_correlated'] = len([
            f for f in removed_features if f.get('reason') == 'high_correlation'
        ])
        
        # 5. Create selected features list
        selected_features = []
        for rank, feat in enumerate(features_to_keep):
            # Calculate combined importance score
            if feat in feature_scores:
                # Calculate weighted importance based on score reliability
                importance = self._calculate_weighted_importance(feature_scores[feat])
                methods = list(feature_scores[feat].keys())
            else:
                # Try to get group-based importance first
                if feat in feature_metadata:
                    group = feature_metadata[feat]['group']
                    group_scores = self._aggregate_group_importance(feature_scores, feature_metadata)
                    if group in group_scores:
                        importance = group_scores[group]  # Inherit full group importance
                        methods = ['group_importance']
                    else:
                        # No group score available, use minimal default
                        importance = 0.1
                        methods = ['default']
                else:
                    # Feature not in metadata, use minimal default
                    importance = 0.1
                    methods = ['default']
            
            # Find correlation cluster
            cluster_id = None
            for idx, cluster in enumerate(correlation_clusters):
                if feat in cluster:
                    cluster_id = idx
                    break
            
            # Calculate interpretability score
            original_features = [f for f in engineering_result.original_features if f.name == feat]
            if original_features:
                interpretability = 1.0  # Original features are most interpretable
            elif 'squared' in feat or 'cubed' in feat:
                interpretability = 0.7  # Polynomial features
            elif 'times' in feat or 'over' in feat:
                interpretability = 0.6  # Interactions
            elif 'log' in feat or 'sqrt' in feat:
                interpretability = 0.8  # Simple transforms
            else:
                interpretability = 0.5  # Complex features
            
            selected_features.append(SelectedFeature(
                name=feat,
                importance_score=float(importance),
                selection_methods=[SelectionMethod(m) for m in methods if m in [e.value for e in SelectionMethod]],
                rank=rank + 1,
                correlation_cluster=cluster_id,
                redundancy_score=0.0,  # Could calculate based on correlation
                interpretability_score=float(interpretability)
            ))
        
        # Sort by importance
        selected_features.sort(key=lambda x: x.importance_score, reverse=True)
        for i, feat in enumerate(selected_features):
            feat.rank = i + 1
        
        # 6. Calculate summary statistics
        if not correlation_matrix.empty:
            correlation_summary = {
                'mean_correlation': float(correlation_matrix.mean().mean()),
                'max_correlation': float(correlation_matrix[correlation_matrix < 1].max().max()) if (correlation_matrix < 1).any().any() else 0.0,
                'correlation_clusters': len(correlation_clusters)
            }
        else:
            correlation_summary = {
                'mean_correlation': 0.0,
                'max_correlation': 0.0,
                'correlation_clusters': 0
            }
        
        dimensionality_reduction = {
            'original_features': len(existing_features),
            'selected_features': len(selected_features),
            'reduction_percentage': (1 - len(selected_features) / len(existing_features)) * 100 if existing_features else 0
        }
        
        # 7. Estimate performance impact
        performance_impact = {
            'expected_accuracy_retention': min(0.95, 0.85 + len(selected_features) / max(1, len(existing_features)) * 0.1),
            'training_speedup': len(existing_features) / max(1, len(selected_features)),
            'interpretability_gain': np.mean([f.interpretability_score for f in selected_features])
        }
        
        # 8. Generate recommendations
        recommendations = []
        if len(selected_features) > 50:
            recommendations.append(f"Consider further reduction - {len(selected_features)} features may still be too many")
        if dimensionality_reduction['reduction_percentage'] > 80:
            recommendations.append("Aggressive reduction applied - consider validating with cross-validation")
        if correlation_summary['max_correlation'] > 0.95:
            recommendations.append("Some features still highly correlated - consider lower threshold")
        
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
        
        # 1. Numeric features - standard methods
        if numeric_features:
            X_numeric = X[numeric_features].fillna(0)
            
            # Mutual information
            if task_type == MLTaskType.REGRESSION:
                mi_scores = mutual_info_regression(X_numeric, y, random_state=42)
            else:
                mi_scores = mutual_info_classif(X_numeric, y, random_state=42)
            
            for feat, score in zip(numeric_features, mi_scores):
                scores[feat] = {'mutual_information': float(score)}
            
            # Random forest importance
            if performance_mode != PerformanceMode.FAST:
                if task_type == MLTaskType.REGRESSION:
                    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                else:
                    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                
                rf.fit(X_numeric, y)
                
                for feat, importance in zip(numeric_features, rf.feature_importances_):
                    if feat in scores:
                        scores[feat]['random_forest'] = float(importance)
                    else:
                        scores[feat] = {'random_forest': float(importance)}
                
                # Permutation importance for top features
                if performance_mode == PerformanceMode.THOROUGH:
                    perm_importance = permutation_importance(
                        rf, X_numeric, y, n_repeats=5, random_state=42, n_jobs=-1
                    )
                    for feat, importance in zip(numeric_features, perm_importance.importances_mean):
                        if feat in scores:
                            scores[feat]['permutation'] = float(importance)
                    
                    # SHAP-based importance (using TreeExplainer for efficiency)
                    try:
                        import shap
                        explainer = shap.TreeExplainer(rf)
                        shap_values = explainer.shap_values(X_numeric)
                        
                        # For multiclass, average across classes
                        if isinstance(shap_values, list):
                            shap_importance = np.abs(shap_values).mean(axis=0).mean(axis=0)
                        else:
                            shap_importance = np.abs(shap_values).mean(axis=0)
                        
                        for feat, importance in zip(numeric_features, shap_importance):
                            if feat in scores:
                                scores[feat]['shap'] = float(importance)
                    except ImportError:
                        pass  # SHAP not installed
                    except Exception:
                        pass  # SHAP calculation failed
        
        # 2. Categorical features - specialized methods
        if categorical_features and task_type != MLTaskType.UNSUPERVISED:
            for feat in categorical_features:
                cat_scores = self._calculate_categorical_importance(
                    X[feat], y, task_type
                )
                scores[feat] = cat_scores
        
        # 3. Calculate interaction importance for top features
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
        
        # 4. Calculate feature stability across subsets
        if performance_mode == PerformanceMode.THOROUGH:
            stability_scores = self._calculate_feature_stability(
                X, y, task_type, feature_metadata
            )
            for feat, stability in stability_scores.items():
                if feat in scores:
                    scores[feat]['stability'] = stability
        
        # 5. Group-based importance aggregation
        group_scores = self._aggregate_group_importance(scores, feature_metadata)
        
        # 6. Apply group scores to encoded features without direct scores
        for feat in X.columns:
            if feat not in scores and feat in feature_metadata:
                group = feature_metadata[feat]['group']
                if group in group_scores:
                    # Inherit full group score
                    scores[feat] = {
                        'group_importance': group_scores[group]
                    }
        
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