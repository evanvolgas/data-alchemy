"""
Polynomial feature transformations
"""
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from .base import BaseTransformer, TransformationResult, register_transformer


@register_transformer('polynomial')
class PolynomialTransformer(BaseTransformer):
    """Creates polynomial features from numeric columns"""
    
    def _validate_config(self) -> None:
        """Validate transformer configuration"""
        # Set defaults
        self.degrees = self.config.get('degrees', [2, 3])
        self.max_features = self.config.get('max_features', 50)
        
        # Validate
        if not isinstance(self.degrees, list) or not all(isinstance(d, int) for d in self.degrees):
            raise ValueError("degrees must be a list of integers")
        if not all(d > 1 for d in self.degrees):
            raise ValueError("All degrees must be greater than 1")
        if not isinstance(self.max_features, int) or self.max_features < 1:
            raise ValueError("max_features must be a positive integer")
    
    def get_applicable_columns(self, df: pd.DataFrame, feature_types: Dict[str, str]) -> List[str]:
        """Get numeric columns suitable for polynomial transformation"""
        numeric_columns = []
        
        for col, dtype in feature_types.items():
            if dtype in ['numeric', 'int', 'float'] and col in df.columns:
                # Check if column has enough unique values
                if df[col].nunique() > 5:
                    numeric_columns.append(col)
        
        return numeric_columns[:self.max_features]  # Limit number of columns
    
    def transform(self, df: pd.DataFrame, columns: List[str]) -> List[TransformationResult]:
        """Apply polynomial transformations"""
        results = []
        
        for col in columns:
            if col not in df.columns:
                continue
                
            for degree in self.degrees:
                feature_name = f"{col}_power_{degree}"
                feature_name = self._safe_column_name(feature_name)
                
                # Skip if would create duplicate
                if feature_name in df.columns:
                    continue
                
                try:
                    values = df[col] ** degree
                    
                    # Check for infinities or all NaN
                    if np.isinf(values).any() or values.isna().all():
                        continue
                    
                    result = TransformationResult(
                        feature_name=feature_name,
                        values=values,
                        source_columns=[col],
                        transformation_type='polynomial',
                        formula=f"f(x) = x^{degree}",
                        python_code=self.get_recreation_code(feature_name, [col], degree=degree),
                        description=f"Polynomial degree {degree} of {col}",
                        metadata={'degree': degree}
                    )
                    results.append(result)
                    
                except Exception:
                    # Skip on any calculation error
                    continue
        
        return results
    
    def get_recreation_code(self, feature_name: str, source_columns: List[str], **kwargs) -> str:
        """Get Python code to recreate polynomial transformation"""
        col = source_columns[0]
        degree = kwargs.get('degree', 2)
        
        if degree == 2:
            return f"df['{feature_name}'] = df['{col}'] ** 2"
        elif degree == 3:
            return f"df['{feature_name}'] = df['{col}'] ** 3"
        else:
            return f"df['{feature_name}'] = df['{col}'] ** {degree}"