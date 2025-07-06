"""
Mathematical feature transformations
"""
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from .base import BaseTransformer, TransformationResult, register_transformer
from ..core import Config


@register_transformer('mathematical')
class MathematicalTransformer(BaseTransformer):
    """Creates mathematical transformations of numeric features"""
    
    def _validate_config(self) -> None:
        """Validate transformer configuration"""
        # Set defaults
        self.transforms = self.config.get('transforms', [
            'log', 'sqrt', 'reciprocal', 'exp', 'sin', 'cos'
        ])
        self.create_binned = self.config.get('create_binned', True)
        self.n_bins = self.config.get('n_bins', 10)
        
        # Validate
        valid_transforms = {
            'log', 'log1p', 'sqrt', 'reciprocal', 'square', 'exp',
            'sin', 'cos', 'tan', 'abs', 'sign'
        }
        invalid = set(self.transforms) - valid_transforms
        if invalid:
            raise ValueError(f"Invalid transforms: {invalid}")
    
    def get_applicable_columns(self, df: pd.DataFrame, feature_types: Dict[str, str]) -> List[str]:
        """Get numeric columns suitable for mathematical transformations"""
        numeric_columns = []
        
        for col, dtype in feature_types.items():
            if dtype in ['numeric', 'int', 'float'] and col in df.columns:
                # Check if column has reasonable values for transformations
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    # For log/sqrt, need positive values
                    if 'log' in self.transforms or 'sqrt' in self.transforms:
                        if (col_data > 0).any():
                            numeric_columns.append(col)
                    else:
                        numeric_columns.append(col)
        
        return numeric_columns
    
    def transform(self, df: pd.DataFrame, columns: List[str]) -> List[TransformationResult]:
        """Apply mathematical transformations"""
        results = []
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # Apply each transformation
            for transform in self.transforms:
                result = self._apply_transform(df, col, transform)
                if result:
                    results.append(result)
            
            # Create binned version if requested
            if self.create_binned:
                result = self._create_binned(df, col)
                if result:
                    results.append(result)
        
        return results
    
    def _apply_transform(self, df: pd.DataFrame, col: str, 
                        transform: str) -> Optional[TransformationResult]:
        """Apply a single mathematical transformation"""
        try:
            feature_name = f"{col}_{transform}"
            feature_name = self._safe_column_name(feature_name)
            
            # Skip if would create duplicate
            if feature_name in df.columns:
                return None
            
            col_data = df[col]
            
            if transform == 'log':
                # Only apply to positive values
                mask = col_data > 0
                if not mask.any():
                    return None
                values = pd.Series(index=df.index, dtype=float)
                values[mask] = np.log(col_data[mask])
                formula = "log(x) for x > 0"
                code = f"df['{feature_name}'] = np.log(df['{col}'].clip(lower=1e-8))"
                
            elif transform == 'log1p':
                # Safe for zero and positive values
                mask = col_data >= 0
                if not mask.any():
                    return None
                values = pd.Series(index=df.index, dtype=float)
                values[mask] = np.log1p(col_data[mask])
                formula = "log(1 + x) for x >= 0"
                code = f"df['{feature_name}'] = np.log1p(df['{col}'].clip(lower=0))"
                
            elif transform == 'sqrt':
                # Only apply to non-negative values
                mask = col_data >= 0
                if not mask.any():
                    return None
                values = pd.Series(index=df.index, dtype=float)
                values[mask] = np.sqrt(col_data[mask])
                formula = "sqrt(x) for x >= 0"
                code = f"df['{feature_name}'] = np.sqrt(df['{col}'].clip(lower=0))"
                
            elif transform == 'reciprocal':
                # Avoid division by zero
                values = 1.0 / (col_data + Config.EPSILON)
                formula = "1 / (x + ε)"
                code = f"df['{feature_name}'] = 1.0 / (df['{col}'] + {Config.EPSILON})"
                
            elif transform == 'square':
                values = col_data ** 2
                formula = "x²"
                code = f"df['{feature_name}'] = df['{col}'] ** 2"
                
            elif transform == 'exp':
                # Clip to avoid overflow
                values = np.exp(col_data.clip(upper=Config.MAX_EXP_INPUT))
                formula = "exp(x)"
                code = f"df['{feature_name}'] = np.exp(df['{col}'].clip(upper={Config.MAX_EXP_INPUT}))"
                
            elif transform == 'sin':
                values = np.sin(col_data)
                formula = "sin(x)"
                code = f"df['{feature_name}'] = np.sin(df['{col}'])"
                
            elif transform == 'cos':
                values = np.cos(col_data)
                formula = "cos(x)"
                code = f"df['{feature_name}'] = np.cos(df['{col}'])"
                
            elif transform == 'abs':
                values = np.abs(col_data)
                formula = "|x|"
                code = f"df['{feature_name}'] = np.abs(df['{col}'])"
                
            else:
                return None
            
            # Check for all NaN or infinite values
            if values.isna().all() or np.isinf(values).any():
                return None
            
            # Map specific transforms to their types
            transform_type_mapping = {
                'log': 'log',
                'log1p': 'log',
                'sqrt': 'sqrt',
                'reciprocal': 'reciprocal',
                'square': 'polynomial',
                'exp': 'exponential',
                'sin': 'trigonometric',
                'cos': 'trigonometric',
                'tan': 'trigonometric',
                'abs': 'mathematical'
            }
            
            return TransformationResult(
                feature_name=feature_name,
                values=values,
                source_columns=[col],
                transformation_type=transform_type_mapping.get(transform, 'mathematical'),
                formula=formula,
                python_code=code,
                description=f"{transform} transformation of {col}",
                metadata={'transform': transform}
            )
            
        except Exception:
            return None
    
    def _create_binned(self, df: pd.DataFrame, col: str) -> Optional[TransformationResult]:
        """Create binned version of numeric feature"""
        try:
            feature_name = f"{col}_binned"
            feature_name = self._safe_column_name(feature_name)
            
            # Skip if would create duplicate
            if feature_name in df.columns:
                return None
            
            # Use quantile-based binning
            values = pd.qcut(df[col], self.n_bins, labels=False, duplicates='drop')
            
            return TransformationResult(
                feature_name=feature_name,
                values=values,
                source_columns=[col],
                transformation_type='binned',
                formula=f"Quantile-based binning into {self.n_bins} bins",
                python_code=f"df['{feature_name}'] = pd.qcut(df['{col}'], {self.n_bins}, labels=False, duplicates='drop')",
                description=f"{col} discretized into {self.n_bins} bins",
                metadata={'n_bins': self.n_bins}
            )
            
        except Exception:
            return None
    
    def get_recreation_code(self, feature_name: str, source_columns: List[str], **kwargs) -> str:
        """Get Python code to recreate mathematical transformation"""
        transform = kwargs.get('transform')
        
        if transform:
            return f"# See _apply_transform for {transform} implementation"
        else:
            return f"# Mathematical transformation of {source_columns[0]}"