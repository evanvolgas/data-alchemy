"""
Categorical feature transformations
"""
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from .base import BaseTransformer, TransformationResult, register_transformer


@register_transformer('categorical')
class CategoricalTransformer(BaseTransformer):
    """Creates features from categorical columns"""
    
    def _validate_config(self) -> None:
        """Validate transformer configuration"""
        # Set defaults
        self.encoding_methods = self.config.get('encoding_methods', [
            'one_hot', 'frequency', 'target'
        ])
        self.max_cardinality = self.config.get('max_cardinality', 20)
        self.min_frequency = self.config.get('min_frequency', 10)
        self.target_column = self.config.get('target_column', None)
        
        # Validate
        valid_methods = {'one_hot', 'frequency', 'target', 'ordinal'}
        invalid = set(self.encoding_methods) - valid_methods
        if invalid:
            raise ValueError(f"Invalid encoding methods: {invalid}")
    
    def get_applicable_columns(self, df: pd.DataFrame, feature_types: Dict[str, str]) -> List[str]:
        """Get categorical columns suitable for encoding"""
        categorical_columns = []
        
        for col, dtype in feature_types.items():
            if dtype in ['categorical', 'object', 'string'] and col in df.columns:
                # Check cardinality
                n_unique = df[col].nunique()
                if 2 <= n_unique <= self.max_cardinality:
                    categorical_columns.append(col)
        
        return categorical_columns
    
    def transform(self, df: pd.DataFrame, columns: List[str]) -> List[TransformationResult]:
        """Apply categorical transformations"""
        results = []
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # Apply each encoding method
            for method in self.encoding_methods:
                if method == 'one_hot':
                    results.extend(self._one_hot_encode(df, col))
                elif method == 'frequency':
                    result = self._frequency_encode(df, col)
                    if result:
                        results.append(result)
                elif method == 'target' and self.target_column:
                    result = self._target_encode(df, col)
                    if result:
                        results.append(result)
        
        return results
    
    def _one_hot_encode(self, df: pd.DataFrame, col: str) -> List[TransformationResult]:
        """Create one-hot encoded features"""
        results = []
        
        try:
            # Get unique values
            unique_values = df[col].dropna().unique()
            
            # Skip if too many unique values
            if len(unique_values) > self.max_cardinality:
                return results
            
            # Create binary feature for each value
            for value in unique_values:
                # Create safe feature name
                value_str = str(value).replace(' ', '_').replace('-', '_')
                feature_name = f"{col}_is_{value_str}"
                feature_name = self._safe_column_name(feature_name)
                
                # Skip if would create duplicate
                if feature_name in df.columns:
                    continue
                
                values = (df[col] == value).astype(int)
                
                # Skip if feature has very low frequency
                if values.sum() < self.min_frequency:
                    continue
                
                result = TransformationResult(
                    feature_name=feature_name,
                    values=values,
                    source_columns=[col],
                    transformation_type='one_hot',
                    formula="1 if value matches, 0 otherwise",
                    python_code=f"df['{feature_name}'] = (df['{col}'] == '{value}').astype(int)",
                    description=f"Binary indicator for {col}=={value}",
                    metadata={'value': str(value)}
                )
                results.append(result)
                
        except Exception:
            pass
        
        return results
    
    def _frequency_encode(self, df: pd.DataFrame, col: str) -> Optional[TransformationResult]:
        """Create frequency encoded feature"""
        try:
            feature_name = f"{col}_frequency"
            feature_name = self._safe_column_name(feature_name)
            
            # Skip if would create duplicate
            if feature_name in df.columns:
                return None
            
            # Calculate frequency mapping
            freq_map = df[col].value_counts().to_dict()
            values = df[col].map(freq_map)
            
            return TransformationResult(
                feature_name=feature_name,
                values=values,
                source_columns=[col],
                transformation_type='frequency_encoded',
                formula="Count of each category value",
                python_code=(
                    f"freq_map = df['{col}'].value_counts().to_dict()\n"
                    f"df['{feature_name}'] = df['{col}'].map(freq_map)"
                ),
                description=f"Frequency encoding of {col}",
                metadata={'encoding': 'frequency'}
            )
            
        except Exception:
            return None
    
    def _target_encode(self, df: pd.DataFrame, col: str) -> Optional[TransformationResult]:
        """Create target encoded feature (mean encoding)"""
        if self.target_column not in df.columns:
            return None
            
        try:
            feature_name = f"{col}_target_encoded"
            feature_name = self._safe_column_name(feature_name)
            
            # Skip if would create duplicate
            if feature_name in df.columns:
                return None
            
            # Simple mean encoding (in practice, should use cross-validation)
            target_map = df.groupby(col)[self.target_column].mean().to_dict()
            values = df[col].map(target_map)
            
            # Fill NaN with global mean
            global_mean = df[self.target_column].mean()
            values = values.fillna(global_mean)
            
            return TransformationResult(
                feature_name=feature_name,
                values=values,
                source_columns=[col],
                transformation_type='target_encoded',
                formula=f"Mean of {self.target_column} for each category",
                python_code=(
                    f"target_map = df.groupby('{col}')['{self.target_column}'].mean().to_dict()\n"
                    f"df['{feature_name}'] = df['{col}'].map(target_map).fillna(df['{self.target_column}'].mean())"
                ),
                description=f"Target encoding of {col}",
                metadata={'encoding': 'target', 'target': self.target_column}
            )
            
        except Exception:
            return None
    
    def get_recreation_code(self, feature_name: str, source_columns: List[str], **kwargs) -> str:
        """Get Python code to recreate categorical transformation"""
        encoding_type = kwargs.get('encoding', 'one_hot')
        
        if encoding_type == 'one_hot':
            value = kwargs.get('value', 'VALUE')
            return f"df['{feature_name}'] = (df['{source_columns[0]}'] == '{value}').astype(int)"
        elif encoding_type == 'frequency':
            return (
                f"freq_map = df['{source_columns[0]}'].value_counts().to_dict()\n"
                f"df['{feature_name}'] = df['{source_columns[0]}'].map(freq_map)"
            )
        else:
            return f"# Categorical transformation: {encoding_type}"