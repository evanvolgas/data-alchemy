import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import time
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
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


class AlchemistDependencies(BaseModel):
    """Dependencies for Alchemist Agent"""
    df: pd.DataFrame
    profile_result: DataProfileResult
    
    class Config:
        arbitrary_types_allowed = True


alchemist_agent = Agent(
    "openai:gpt-4o-mini",
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
    """Alchemist Agent for feature engineering"""
    
    def __init__(self):
        self.agent = alchemist_agent
    
    async def engineer_features(
        self,
        df: pd.DataFrame,
        profile_result: DataProfileResult,
        target: Optional[pd.Series] = None
    ) -> FeatureEngineeringResult:
        """Engineer features based on data profile"""
        start_time = time.time()
        
        # Initialize feature tracking
        original_features = []
        engineered_features = []
        warnings = []
        strategies_used = []
        
        # Work directly on the dataframe (it's already a copy in the pipeline)
        feature_df = df
        
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
        
        # Engineer features based on data types
        for col_profile in profile_result.column_profiles:
            col_name = col_profile.name
            
            # Skip target column
            if target is not None and col_name == target.name:
                continue
            
            if col_profile.dtype == DataType.NUMERIC:
                new_features = self._engineer_numeric_features(
                    feature_df, col_name, col_profile,
                    profile_result.context.performance_mode
                )
                engineered_features.extend(new_features)
                if new_features:
                    strategies_used.append("Numeric transformations")
                    
            elif col_profile.dtype == DataType.CATEGORICAL:
                new_features = self._engineer_categorical_features(
                    feature_df, col_name, col_profile,
                    profile_result.context.performance_mode
                )
                engineered_features.extend(new_features)
                if new_features:
                    strategies_used.append("Categorical encoding")
                    
            elif col_profile.dtype == DataType.DATETIME:
                new_features = self._engineer_datetime_features(
                    feature_df, col_name, col_profile
                )
                engineered_features.extend(new_features)
                if new_features:
                    strategies_used.append("Temporal features")
                    
            elif col_profile.dtype == DataType.TEXT:
                new_features = self._engineer_text_features(
                    feature_df, col_name, col_profile
                )
                engineered_features.extend(new_features)
                if new_features:
                    strategies_used.append("Text features")
        
        # Create interaction features for top numeric columns
        if profile_result.context.performance_mode != PerformanceMode.FAST:
            numeric_cols = [p.name for p in profile_result.column_profiles 
                           if p.dtype == DataType.NUMERIC and p.unique_count > 2
                           and (target is None or p.name != target.name)]
            
            if len(numeric_cols) >= 2:
                interaction_features = self._engineer_interaction_features(
                    feature_df, numeric_cols[:5],  # Limit to top 5 to avoid explosion
                    profile_result.context.performance_mode
                )
                engineered_features.extend(interaction_features)
                if interaction_features:
                    strategies_used.append("Interaction features")
        
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
    
    def _engineer_numeric_features(
        self,
        df: pd.DataFrame,
        col_name: str,
        col_profile: Any,
        performance_mode: PerformanceMode
    ) -> List[Feature]:
        """Engineer features for numeric columns"""
        features = []
        series = df[col_name]
        
        # Skip if constant or mostly missing
        if col_profile.unique_count <= 1 or col_profile.missing_percentage > 0.8:
            return features
        
        # Log transform (for positive skewed data)
        if col_profile.statistics.get('min', 0) > 0 and abs(col_profile.statistics.get('skewness', 0)) > 1:
            features.append(Feature(
                name=f"{col_name}_log",
                type=FeatureType.LOG_TRANSFORM,
                source_columns=[col_name],
                description=f"Natural log transform of {col_name}",
                dtype="float64",
                mathematical_explanation="f(x) = ln(x) - reduces right skewness",
                computational_complexity="O(n)"
            ))
            df[f"{col_name}_log"] = np.log1p(series)
        
        # Square root transform (for moderate skewness)
        if col_profile.statistics.get('min', 0) >= 0:
            features.append(Feature(
                name=f"{col_name}_sqrt",
                type=FeatureType.SQRT_TRANSFORM,
                source_columns=[col_name],
                description=f"Square root transform of {col_name}",
                dtype="float64",
                mathematical_explanation="f(x) = √x - mild variance stabilization",
                computational_complexity="O(n)"
            ))
            df[f"{col_name}_sqrt"] = np.sqrt(series.fillna(0))
        
        # Polynomial features (only in medium/thorough mode)
        if performance_mode != PerformanceMode.FAST:
            features.append(Feature(
                name=f"{col_name}_squared",
                type=FeatureType.POLYNOMIAL,
                source_columns=[col_name],
                description=f"Square of {col_name}",
                dtype="float64",
                mathematical_explanation="f(x) = x² - captures quadratic relationships",
                computational_complexity="O(n)"
            ))
            df[f"{col_name}_squared"] = series ** 2
            
            if performance_mode == PerformanceMode.THOROUGH:
                features.append(Feature(
                    name=f"{col_name}_cubed",
                    type=FeatureType.POLYNOMIAL,
                    source_columns=[col_name],
                    description=f"Cube of {col_name}",
                    dtype="float64",
                    mathematical_explanation="f(x) = x³ - captures cubic relationships",
                    computational_complexity="O(n)"
                ))
                df[f"{col_name}_cubed"] = series ** 3
        
        # Binning for continuous variables
        if col_profile.unique_count > 10:
            n_bins = 5 if performance_mode == PerformanceMode.FAST else 10
            features.append(Feature(
                name=f"{col_name}_binned",
                type=FeatureType.BINNED,
                source_columns=[col_name],
                description=f"{col_name} discretized into {n_bins} bins",
                dtype="int64",
                mathematical_explanation=f"Equal-width binning into {n_bins} categories",
                computational_complexity="O(n)"
            ))
            df[f"{col_name}_binned"] = pd.qcut(series, n_bins, labels=False, duplicates='drop')
        
        return features
    
    def _engineer_categorical_features(
        self,
        df: pd.DataFrame,
        col_name: str,
        col_profile: Any,
        performance_mode: PerformanceMode
    ) -> List[Feature]:
        """Engineer features for categorical columns"""
        features = []
        series = df[col_name]
        
        # Skip if too many unique values or constant
        if col_profile.unique_count <= 1 or col_profile.unique_count > 100:
            return features
        
        # Frequency encoding
        features.append(Feature(
            name=f"{col_name}_frequency",
            type=FeatureType.FREQUENCY_ENCODED,
            source_columns=[col_name],
            description=f"Frequency encoding of {col_name}",
            dtype="float64",
            mathematical_explanation="Maps each category to its occurrence frequency",
            computational_complexity="O(n)"
        ))
        freq_map = series.value_counts().to_dict()
        df[f"{col_name}_frequency"] = series.map(freq_map)
        
        # One-hot encoding for low cardinality
        if col_profile.unique_count <= 10 and performance_mode != PerformanceMode.FAST:
            # Get top categories
            top_categories = series.value_counts().head(5).index.tolist()
            
            for category in top_categories:
                safe_cat_name = str(category).replace(' ', '_').replace('-', '_')
                features.append(Feature(
                    name=f"{col_name}_is_{safe_cat_name}",
                    type=FeatureType.ONE_HOT,
                    source_columns=[col_name],
                    description=f"Binary indicator for {col_name}=={category}",
                    dtype="int64",
                    mathematical_explanation="1 if category matches, 0 otherwise",
                    computational_complexity="O(n)"
                ))
                df[f"{col_name}_is_{safe_cat_name}"] = (series == category).astype(int)
        
        return features
    
    def _engineer_datetime_features(
        self,
        df: pd.DataFrame,
        col_name: str,
        col_profile: Any
    ) -> List[Feature]:
        """Engineer features for datetime columns"""
        features = []
        
        # Convert to datetime if needed
        if col_profile.dtype == DataType.DATETIME:
            series = pd.to_datetime(df[col_name], errors='coerce')
        else:
            return features
        
        # Basic components
        components = [
            ('year', 'Year component', 'Extracts year as integer'),
            ('month', 'Month component (1-12)', 'Extracts month as integer'),
            ('day', 'Day of month (1-31)', 'Extracts day as integer'),
            ('dayofweek', 'Day of week (0=Monday)', 'Extracts day of week'),
            ('hour', 'Hour component (0-23)', 'Extracts hour for intraday patterns'),
        ]
        
        for attr, desc, math_exp in components:
            if hasattr(series.dt, attr):
                features.append(Feature(
                    name=f"{col_name}_{attr}",
                    type=FeatureType.DATE_COMPONENT,
                    source_columns=[col_name],
                    description=desc,
                    dtype="int64",
                    mathematical_explanation=math_exp,
                    computational_complexity="O(n)"
                ))
                df[f"{col_name}_{attr}"] = getattr(series.dt, attr)
        
        # Cyclical encoding for month and day of week
        features.append(Feature(
            name=f"{col_name}_month_sin",
            type=FeatureType.DATE_COMPONENT,
            source_columns=[col_name],
            description="Sine encoding of month",
            dtype="float64",
            mathematical_explanation="sin(2π * month / 12) - captures cyclical nature",
            computational_complexity="O(n)"
        ))
        features.append(Feature(
            name=f"{col_name}_month_cos",
            type=FeatureType.DATE_COMPONENT,
            source_columns=[col_name],
            description="Cosine encoding of month",
            dtype="float64",
            mathematical_explanation="cos(2π * month / 12) - captures cyclical nature",
            computational_complexity="O(n)"
        ))
        df[f"{col_name}_month_sin"] = np.sin(2 * np.pi * series.dt.month / 12)
        df[f"{col_name}_month_cos"] = np.cos(2 * np.pi * series.dt.month / 12)
        
        # Is weekend
        features.append(Feature(
            name=f"{col_name}_is_weekend",
            type=FeatureType.DATE_COMPONENT,
            source_columns=[col_name],
            description="Binary indicator for weekend",
            dtype="int64",
            mathematical_explanation="1 if Saturday/Sunday, 0 otherwise",
            computational_complexity="O(n)"
        ))
        df[f"{col_name}_is_weekend"] = (series.dt.dayofweek >= 5).astype(int)
        
        return features
    
    def _engineer_text_features(
        self,
        df: pd.DataFrame,
        col_name: str,
        col_profile: Any
    ) -> List[Feature]:
        """Engineer features for text columns"""
        features = []
        series = df[col_name].fillna('')
        
        # Text length
        features.append(Feature(
            name=f"{col_name}_length",
            type=FeatureType.TEXT_LENGTH,
            source_columns=[col_name],
            description=f"Character length of {col_name}",
            dtype="int64",
            mathematical_explanation="Number of characters in string",
            computational_complexity="O(n)"
        ))
        df[f"{col_name}_length"] = series.str.len()
        
        # Word count
        features.append(Feature(
            name=f"{col_name}_word_count",
            type=FeatureType.TEXT_WORD_COUNT,
            source_columns=[col_name],
            description=f"Word count in {col_name}",
            dtype="int64",
            mathematical_explanation="Number of space-separated tokens",
            computational_complexity="O(n)"
        ))
        df[f"{col_name}_word_count"] = series.str.split().str.len()
        
        # Contains patterns
        patterns = [
            ('has_numbers', r'\d', 'Contains numeric characters'),
            ('has_uppercase', r'[A-Z]', 'Contains uppercase letters'),
            ('has_special', r'[^a-zA-Z0-9\s]', 'Contains special characters')
        ]
        
        for pattern_name, regex, desc in patterns:
            features.append(Feature(
                name=f"{col_name}_{pattern_name}",
                type=FeatureType.TEXT_LENGTH,
                source_columns=[col_name],
                description=desc,
                dtype="int64",
                mathematical_explanation=f"Binary indicator for regex pattern: {regex}",
                computational_complexity="O(n)"
            ))
            df[f"{col_name}_{pattern_name}"] = series.str.contains(regex, na=False).astype(int)
        
        return features
    
    def _engineer_interaction_features(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        performance_mode: PerformanceMode
    ) -> List[Feature]:
        """Create interaction features between numeric columns"""
        features = []
        
        # Limit combinations based on performance mode
        max_pairs = 3 if performance_mode == PerformanceMode.MEDIUM else 6
        
        for i, (col1, col2) in enumerate(combinations(numeric_cols, 2)):
            if i >= max_pairs:
                break
                
            # Multiplication interaction
            features.append(Feature(
                name=f"{col1}_times_{col2}",
                type=FeatureType.INTERACTION,
                source_columns=[col1, col2],
                description=f"Multiplication interaction between {col1} and {col2}",
                dtype="float64",
                mathematical_explanation="f(x,y) = x * y - captures multiplicative effects",
                computational_complexity="O(n)"
            ))
            df[f"{col1}_times_{col2}"] = df[col1] * df[col2]
            
            # Ratio (if denominator is not zero)
            if not (df[col2] == 0).any():
                features.append(Feature(
                    name=f"{col1}_over_{col2}",
                    type=FeatureType.RATIO,
                    source_columns=[col1, col2],
                    description=f"Ratio of {col1} to {col2}",
                    dtype="float64",
                    mathematical_explanation="f(x,y) = x / y - captures proportional relationships",
                    computational_complexity="O(n)"
                ))
                df[f"{col1}_over_{col2}"] = df[col1] / (df[col2] + 1e-8)
        
        return features