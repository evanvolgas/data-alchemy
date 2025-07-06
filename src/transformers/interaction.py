"""
Feature interaction transformations
"""
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from itertools import combinations
from .base import BaseTransformer, TransformationResult, register_transformer
from ..core import Config


@register_transformer('interaction')
class InteractionTransformer(BaseTransformer):
    """Creates interaction features between numeric columns"""
    
    def _validate_config(self) -> None:
        """Validate transformer configuration"""
        # Set defaults
        self.max_interactions = self.config.get('max_interactions', 100)
        self.operations = self.config.get('operations', ['multiply', 'divide'])
        self.min_correlation_diff = self.config.get('min_correlation_diff', 0.05)
        
        # Validate
        valid_operations = ['multiply', 'divide', 'add', 'subtract']
        if not all(op in valid_operations for op in self.operations):
            raise ValueError(f"operations must be from {valid_operations}")
        if not isinstance(self.max_interactions, int) or self.max_interactions < 1:
            raise ValueError("max_interactions must be a positive integer")
    
    def get_applicable_columns(self, df: pd.DataFrame, feature_types: Dict[str, str]) -> List[str]:
        """Get numeric columns suitable for interactions"""
        numeric_columns = []
        
        for col, dtype in feature_types.items():
            if dtype in ['numeric', 'int', 'float'] and col in df.columns:
                # Check basic properties
                if df[col].nunique() > 2 and df[col].std() > 0:
                    numeric_columns.append(col)
        
        return numeric_columns
    
    def transform(self, df: pd.DataFrame, columns: List[str]) -> List[TransformationResult]:
        """Apply interaction transformations"""
        results = []
        interaction_count = 0
        
        # Get all pairs of columns
        for col1, col2 in combinations(columns, 2):
            if col1 not in df.columns or col2 not in df.columns:
                continue
                
            # Check if we've reached the limit
            if interaction_count >= self.max_interactions:
                break
            
            # Apply each operation
            for operation in self.operations:
                if interaction_count >= self.max_interactions:
                    break
                    
                result = self._create_interaction(df, col1, col2, operation)
                if result:
                    results.append(result)
                    interaction_count += 1
        
        return results
    
    def _create_interaction(self, df: pd.DataFrame, col1: str, col2: str, 
                          operation: str) -> Optional[TransformationResult]:
        """Create a single interaction feature"""
        try:
            # Generate feature name
            if operation == 'multiply':
                feature_name = f"{col1}_times_{col2}"
                values = df[col1] * df[col2]
                formula = "f(x,y) = x * y"
                symbol = "*"
            elif operation == 'divide':
                feature_name = f"{col1}_over_{col2}"
                # Safe division: only divide where denominator is significant
                denominator = df[col2]
                mask = np.abs(denominator) > Config.EPSILON
                values = pd.Series(index=df.index, dtype=float)
                if mask.any():
                    values[mask] = df[col1][mask] / denominator[mask]
                # Fill remaining with 0 or could use median of valid ratios
                values = values.fillna(0)
                formula = "f(x,y) = x / y (safe division)"
                symbol = "/"
            elif operation == 'add':
                feature_name = f"{col1}_plus_{col2}"
                values = df[col1] + df[col2]
                formula = "f(x,y) = x + y"
                symbol = "+"
            elif operation == 'subtract':
                feature_name = f"{col1}_minus_{col2}"
                values = df[col1] - df[col2]
                formula = "f(x,y) = x - y"
                symbol = "-"
            else:
                return None
            
            feature_name = self._safe_column_name(feature_name)
            
            # Skip if would create duplicate
            if feature_name in df.columns:
                return None
            
            # Check for infinities or all NaN
            if np.isinf(values).any() or values.isna().all():
                return None
            
            # Check if interaction provides value over individual features
            # (This is a simple heuristic - could be made more sophisticated)
            if operation in ['multiply', 'divide']:
                # Check if variance is reasonable
                if values.std() == 0:
                    return None
            
            return TransformationResult(
                feature_name=feature_name,
                values=values,
                source_columns=[col1, col2],
                transformation_type='interaction',
                formula=formula,
                python_code=self.get_recreation_code(feature_name, [col1, col2], 
                                                   operation=operation),
                description=f"{operation.capitalize()} interaction between {col1} and {col2}",
                metadata={'operation': operation}
            )
            
        except Exception:
            return None
    
    def get_recreation_code(self, feature_name: str, source_columns: List[str], **kwargs) -> str:
        """Get Python code to recreate interaction"""
        col1, col2 = source_columns[0], source_columns[1]
        operation = kwargs.get('operation', 'multiply')
        
        if operation == 'multiply':
            return f"df['{feature_name}'] = df['{col1}'] * df['{col2}']"
        elif operation == 'divide':
            return f"""# Safe division to avoid explosion from small denominators
mask = np.abs(df['{col2}']) > {Config.EPSILON}
df['{feature_name}'] = 0.0  # Initialize with zeros
df.loc[mask, '{feature_name}'] = df.loc[mask, '{col1}'] / df.loc[mask, '{col2}']"""
        elif operation == 'add':
            return f"df['{feature_name}'] = df['{col1}'] + df['{col2}']"
        elif operation == 'subtract':
            return f"df['{feature_name}'] = df['{col1}'] - df['{col2}']"
        else:
            return f"# Unknown operation: {operation}"