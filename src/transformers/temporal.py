"""
Temporal feature transformations for datetime columns
"""
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from .base import BaseTransformer, TransformationResult, register_transformer


@register_transformer('temporal')
class TemporalTransformer(BaseTransformer):
    """Creates temporal features from datetime columns"""
    
    def _validate_config(self) -> None:
        """Validate transformer configuration"""
        # Set defaults
        self.components = self.config.get('components', [
            'year', 'month', 'day', 'dayofweek', 'quarter',
            'is_weekend', 'month_sin', 'month_cos', 'day_sin', 'day_cos'
        ])
        self.create_diffs = self.config.get('create_diffs', True)
        
        # Validate
        valid_components = {
            'year', 'month', 'day', 'hour', 'minute', 'dayofweek', 
            'dayofyear', 'weekofyear', 'quarter', 'is_weekend',
            'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end',
            'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos'
        }
        
        invalid = set(self.components) - valid_components
        if invalid:
            raise ValueError(f"Invalid components: {invalid}")
    
    def get_applicable_columns(self, df: pd.DataFrame, feature_types: Dict[str, str]) -> List[str]:
        """Get datetime columns"""
        datetime_columns = []
        
        for col, dtype in feature_types.items():
            if dtype in ['datetime', 'date'] and col in df.columns:
                datetime_columns.append(col)
        
        return datetime_columns
    
    def transform(self, df: pd.DataFrame, columns: List[str]) -> List[TransformationResult]:
        """Apply temporal transformations"""
        results = []
        
        for col in columns:
            if col not in df.columns:
                continue
            
            try:
                # Ensure datetime type
                dt_series = pd.to_datetime(df[col])
                
                # Extract components
                for component in self.components:
                    result = self._extract_component(dt_series, col, component)
                    if result:
                        results.append(result)
                
                # Create time differences if requested
                if self.create_diffs and len(columns) > 1:
                    results.extend(self._create_time_diffs(df, col, columns))
                    
            except Exception:
                continue
        
        return results
    
    def _extract_component(self, dt_series: pd.Series, col_name: str, 
                         component: str) -> Optional[TransformationResult]:
        """Extract a single temporal component"""
        try:
            feature_name = f"{col_name}_{component}"
            feature_name = self._safe_column_name(feature_name)
            
            if component == 'year':
                values = dt_series.dt.year
                formula = "Extract year"
                code = f"df['{feature_name}'] = pd.to_datetime(df['{col_name}']).dt.year"
            elif component == 'month':
                values = dt_series.dt.month
                formula = "Extract month (1-12)"
                code = f"df['{feature_name}'] = pd.to_datetime(df['{col_name}']).dt.month"
            elif component == 'day':
                values = dt_series.dt.day
                formula = "Extract day of month (1-31)"
                code = f"df['{feature_name}'] = pd.to_datetime(df['{col_name}']).dt.day"
            elif component == 'hour':
                values = dt_series.dt.hour
                formula = "Extract hour (0-23)"
                code = f"df['{feature_name}'] = pd.to_datetime(df['{col_name}']).dt.hour"
            elif component == 'dayofweek':
                values = dt_series.dt.dayofweek
                formula = "Extract day of week (0=Monday)"
                code = f"df['{feature_name}'] = pd.to_datetime(df['{col_name}']).dt.dayofweek"
            elif component == 'dayofyear':
                values = dt_series.dt.dayofyear
                formula = "Extract day of year (1-366)"
                code = f"df['{feature_name}'] = pd.to_datetime(df['{col_name}']).dt.dayofyear"
            elif component == 'quarter':
                values = dt_series.dt.quarter
                formula = "Extract quarter (1-4)"
                code = f"df['{feature_name}'] = pd.to_datetime(df['{col_name}']).dt.quarter"
            elif component == 'is_weekend':
                values = dt_series.dt.dayofweek.isin([5, 6]).astype(int)
                formula = "1 if Saturday/Sunday, 0 otherwise"
                code = f"df['{feature_name}'] = pd.to_datetime(df['{col_name}']).dt.dayofweek.isin([5, 6]).astype(int)"
            elif component == 'month_sin':
                values = np.sin(2 * np.pi * dt_series.dt.month / 12)
                formula = "sin(2π * month / 12)"
                code = f"df['{feature_name}'] = np.sin(2 * np.pi * pd.to_datetime(df['{col_name}']).dt.month / 12)"
            elif component == 'month_cos':
                values = np.cos(2 * np.pi * dt_series.dt.month / 12)
                formula = "cos(2π * month / 12)"
                code = f"df['{feature_name}'] = np.cos(2 * np.pi * pd.to_datetime(df['{col_name}']).dt.month / 12)"
            elif component == 'day_sin':
                values = np.sin(2 * np.pi * dt_series.dt.day / 31)
                formula = "sin(2π * day / 31)"
                code = f"df['{feature_name}'] = np.sin(2 * np.pi * pd.to_datetime(df['{col_name}']).dt.day / 31)"
            elif component == 'day_cos':
                values = np.cos(2 * np.pi * dt_series.dt.day / 31)
                formula = "cos(2π * day / 31)"
                code = f"df['{feature_name}'] = np.cos(2 * np.pi * pd.to_datetime(df['{col_name}']).dt.day / 31)"
            else:
                return None
            
            # Map components to specific types
            component_type_mapping = {
                'year': 'temporal_component',
                'month': 'temporal_component', 
                'day': 'temporal_component',
                'hour': 'temporal_component',
                'dayofweek': 'temporal_component',
                'dayofyear': 'temporal_component',
                'quarter': 'temporal_component',
                'is_weekend': 'temporal_component',
                'month_sin': 'temporal_cyclical',
                'month_cos': 'temporal_cyclical',
                'day_sin': 'temporal_cyclical',
                'day_cos': 'temporal_cyclical',
                'hour_sin': 'temporal_cyclical',
                'hour_cos': 'temporal_cyclical'
            }
            
            return TransformationResult(
                feature_name=feature_name,
                values=values,
                source_columns=[col_name],
                transformation_type=component_type_mapping.get(component, 'temporal_component'),
                formula=formula,
                python_code=code,
                description=f"{component} component of {col_name}",
                metadata={'component': component}
            )
            
        except Exception:
            return None
    
    def _create_time_diffs(self, df: pd.DataFrame, col: str, 
                          all_columns: List[str]) -> List[TransformationResult]:
        """Create time difference features between datetime columns"""
        results = []
        
        for other_col in all_columns:
            if other_col == col or other_col not in df.columns:
                continue
            
            try:
                dt1 = pd.to_datetime(df[col])
                dt2 = pd.to_datetime(df[other_col])
                
                # Days difference
                feature_name = f"{col}_minus_{other_col}_days"
                feature_name = self._safe_column_name(feature_name)
                
                values = (dt1 - dt2).dt.total_seconds() / 86400  # Convert to days
                
                if not values.isna().all():
                    result = TransformationResult(
                        feature_name=feature_name,
                        values=values,
                        source_columns=[col, other_col],
                        transformation_type='temporal_diff',
                        formula="(datetime1 - datetime2) in days",
                        python_code=f"df['{feature_name}'] = (pd.to_datetime(df['{col}']) - pd.to_datetime(df['{other_col}'])).dt.total_seconds() / 86400",
                        description=f"Days between {col} and {other_col}",
                        metadata={'unit': 'days'}
                    )
                    results.append(result)
                    
            except Exception:
                continue
        
        return results
    
    def get_recreation_code(self, feature_name: str, source_columns: List[str], **kwargs) -> str:
        """Get Python code to recreate temporal transformation"""
        # This is handled in _extract_component for most cases
        # This method is mainly for consistency with the interface
        component = kwargs.get('component')
        if component:
            return f"# See _extract_component for {component} transformation"
        else:
            return f"# Temporal transformation for {source_columns}"