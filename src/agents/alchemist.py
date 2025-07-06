import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import time
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel

from ..models import (
    DataProfileResult,
    FeatureEngineeringResult,
    Feature,
    FeatureType,
    DataType,
    PerformanceMode
)
from ..transformers import transformer_registry
from ..core import Config


class AlchemistDependencies(BaseModel):
    """Dependencies for Alchemist Agent"""
    df: pd.DataFrame
    profile_result: DataProfileResult
    
    class Config:
        arbitrary_types_allowed = True


alchemist_agent = Agent(
    Config.get_model_string(),
    deps_type=AlchemistDependencies,
    system_prompt="""You are the Alchemist Agent, a feature engineering expert.
    Your role is to create meaningful features from raw data.
    Focus on:
    1. Creating polynomial and interaction features for numeric data
    2. Engineering temporal features from datetime columns
    3. Encoding categorical variables effectively
    4. Extracting features from text data
    Always provide mathematical explanations for transformations.
    """
)


class AlchemistAgent:
    """Alchemist Agent for feature engineering using transformer system"""
    
    def __init__(self):
        self.agent = alchemist_agent
        self._setup_transformers()
    
    def _setup_transformers(self):
        """Configure transformers based on common use cases"""
        # Transformer configurations from centralized config
        self.transformer_configs = Config.get_transformer_configs()
    
    async def engineer_features(
        self,
        df: pd.DataFrame,
        profile_result: DataProfileResult,
        target: Optional[pd.Series] = None
    ) -> FeatureEngineeringResult:
        """Engineer features using the transformer system"""
        start_time = time.time()
        
        # Initialize feature tracking
        original_features = []
        engineered_features = []
        warnings = []
        strategies_used = []
        
        # Work directly on the dataframe (it's already a copy in the pipeline)
        feature_df = df
        
        # Build feature type mapping from profile
        feature_types = self._build_feature_type_mapping(profile_result, target)
        
        # Track original features (excluding target)
        for col_profile in profile_result.column_profiles:
            # Skip target column if specified
            if target is not None and col_profile.name == target.name:
                continue
            
            original_features.append(Feature(
                name=col_profile.name,
                type=FeatureType.ORIGINAL,
                source_columns=[col_profile.name],
                description=f"Original column: {col_profile.dtype}",
                dtype=col_profile.dtype,
                computational_complexity="O(1)"
            ))
        
        # Configure transformers based on performance mode
        excluded_transformers = self._get_excluded_transformers(profile_result.context.performance_mode)
        
        # Configure categorical transformer with target if available
        if target is not None and 'categorical' not in excluded_transformers:
            cat_config = self.transformer_configs['categorical'].copy()
            cat_config['target_column'] = target.name
            transformer_registry.create('categorical', cat_config)
        
        # Apply all transformers
        transformation_results = transformer_registry.apply_all(
            feature_df,
            feature_types,
            exclude=excluded_transformers
        )
        
        # Convert transformation results to features and add to dataframe
        for result in transformation_results:
            # Add the feature to the dataframe
            feature_df[result.feature_name] = result.values
            
            # Determine feature type based on transformation type
            feature_type = self._map_transformation_to_feature_type(result.transformation_type)
            
            # Create Feature object with transformation metadata
            feature = Feature(
                name=result.feature_name,
                type=feature_type,
                source_columns=result.source_columns,
                description=result.description,
                dtype=str(result.values.dtype),
                mathematical_explanation=result.formula,
                computational_complexity="O(n)",
                metadata={
                    'python_code': result.python_code,
                    'transformation_type': result.transformation_type,
                    **result.metadata  # Include any additional metadata from the transformer
                }
            )
            engineered_features.append(feature)
            
            # Track strategy
            if result.transformation_type not in strategies_used:
                strategies_used.append(result.transformation_type)
        
        # Calculate memory estimate
        total_features = len(original_features) + len(engineered_features)
        memory_estimate_mb = (
            total_features * len(df) * 8  # 8 bytes per float64
        ) / (1024 * 1024)
        
        # Add warnings for dimensionality
        if total_features > 100:
            warnings.append(f"High dimensionality: {total_features} features")
        if memory_estimate_mb > 500:
            warnings.append(f"Large memory footprint: {memory_estimate_mb:.1f} MB")
        
        processing_time = time.time() - start_time
        
        # Map strategies to user-friendly names
        strategy_mapping = {
            'polynomial': 'Polynomial transformations',
            'interaction': 'Feature interactions',
            'temporal': 'Temporal features',
            'categorical': 'Categorical encoding',
            'mathematical': 'Mathematical transformations'
        }
        
        strategies_used = [strategy_mapping.get(s, s.title()) for s in strategies_used]
        
        return FeatureEngineeringResult(
            original_features=original_features,
            engineered_features=engineered_features,
            total_features_created=len(engineered_features),
            feature_matrix_shape=(len(df), total_features),
            memory_estimate_mb=memory_estimate_mb,
            engineering_strategies_used=list(set(strategies_used)),
            warnings=warnings,
            processing_time_seconds=processing_time
        )
    
    def _build_feature_type_mapping(
        self, 
        profile_result: DataProfileResult,
        target: Optional[pd.Series]
    ) -> Dict[str, str]:
        """Build a mapping of column names to feature types for transformers"""
        feature_types = {}
        
        for col_profile in profile_result.column_profiles:
            # Skip target column
            if target is not None and col_profile.name == target.name:
                continue
                
            if col_profile.dtype == DataType.NUMERIC:
                feature_types[col_profile.name] = 'numeric'
            elif col_profile.dtype == DataType.CATEGORICAL:
                feature_types[col_profile.name] = 'categorical'
            elif col_profile.dtype == DataType.DATETIME:
                feature_types[col_profile.name] = 'datetime'
            elif col_profile.dtype == DataType.TEXT:
                feature_types[col_profile.name] = 'text'
            elif col_profile.dtype == DataType.BOOLEAN:
                feature_types[col_profile.name] = 'boolean'
            else:
                feature_types[col_profile.name] = 'unknown'
        
        return feature_types
    
    def _get_excluded_transformers(self, performance_mode: PerformanceMode) -> List[str]:
        """Get list of transformers to exclude based on performance mode"""
        if performance_mode == PerformanceMode.FAST:
            # Only basic transformations
            return ['polynomial', 'interaction', 'mathematical']
        elif performance_mode == PerformanceMode.MEDIUM:
            # Exclude some expensive transformations
            return ['mathematical']
        else:  # THOROUGH
            # Use all transformers
            return []
    
    def _map_transformation_to_feature_type(self, transformation_type: str) -> FeatureType:
        """Map transformer types to simplified feature types"""
        mapping = {
            'polynomial': FeatureType.MATHEMATICAL,
            'interaction': FeatureType.INTERACTION,
            'ratio': FeatureType.INTERACTION,
            'temporal': FeatureType.TEMPORAL,
            'date_component': FeatureType.TEMPORAL,
            'temporal_diff': FeatureType.TEMPORAL,
            'temporal_cyclical': FeatureType.TEMPORAL,
            'one_hot': FeatureType.CATEGORICAL,
            'frequency_encoded': FeatureType.CATEGORICAL,
            'target_encoded': FeatureType.CATEGORICAL,
            'mathematical': FeatureType.MATHEMATICAL,
            'binned': FeatureType.MATHEMATICAL,
            'log': FeatureType.MATHEMATICAL,
            'sqrt': FeatureType.MATHEMATICAL,
            'reciprocal': FeatureType.MATHEMATICAL,
            'exponential': FeatureType.MATHEMATICAL,
            'trigonometric': FeatureType.MATHEMATICAL,
            'text': FeatureType.TEXT
        }
        return mapping.get(transformation_type, FeatureType.ENGINEERED)