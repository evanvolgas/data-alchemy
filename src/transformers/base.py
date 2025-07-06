"""
Base transformer interface and registry for feature transformations
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Type
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import json


@dataclass
class TransformationResult:
    """Result of a feature transformation"""
    feature_name: str
    values: pd.Series
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_columns: List[str] = field(default_factory=list)
    transformation_type: str = ""
    formula: str = ""
    python_code: str = ""
    description: str = ""


class BaseTransformer(ABC):
    """Abstract base class for all feature transformers"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate transformer configuration"""
        pass
    
    @abstractmethod
    def get_applicable_columns(self, df: pd.DataFrame, feature_types: Dict[str, str]) -> List[str]:
        """
        Get columns this transformer can be applied to
        
        Args:
            df: Input dataframe
            feature_types: Dictionary mapping column names to their types
            
        Returns:
            List of column names this transformer can process
        """
        pass
    
    @abstractmethod
    def transform(self, df: pd.DataFrame, columns: List[str]) -> List[TransformationResult]:
        """
        Apply transformation to specified columns
        
        Args:
            df: Input dataframe
            columns: Columns to transform
            
        Returns:
            List of transformation results
        """
        pass
    
    @abstractmethod
    def get_recreation_code(self, feature_name: str, source_columns: List[str], **kwargs) -> str:
        """
        Get Python code to recreate a specific transformation
        
        Args:
            feature_name: Name of the transformed feature
            source_columns: Source columns used
            **kwargs: Additional parameters specific to the transformation
            
        Returns:
            Python code as a string
        """
        pass
    
    def _safe_column_name(self, name: str) -> str:
        """Create a safe column name"""
        # Replace spaces and special characters
        safe_name = name.replace(' ', '_').replace('-', '_').replace('.', '_')
        # Remove any remaining non-alphanumeric characters except underscore
        safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
        # Ensure it doesn't start with a number
        if safe_name and safe_name[0].isdigit():
            safe_name = f'feature_{safe_name}'
        return safe_name


class TransformerRegistry:
    """Registry for managing feature transformers"""
    
    def __init__(self):
        self._transformers: Dict[str, Type[BaseTransformer]] = {}
        self._instances: Dict[str, BaseTransformer] = {}
    
    def register(self, name: str, transformer_class: Type[BaseTransformer]) -> None:
        """Register a transformer class"""
        if not issubclass(transformer_class, BaseTransformer):
            raise ValueError(f"{transformer_class} must be a subclass of BaseTransformer")
        self._transformers[name] = transformer_class
    
    def create(self, name: str, config: Optional[Dict[str, Any]] = None) -> BaseTransformer:
        """Create a transformer instance"""
        if name not in self._transformers:
            raise ValueError(f"Unknown transformer: {name}")
        
        # Create instance if not already created
        if name not in self._instances:
            self._instances[name] = self._transformers[name](name, config)
        
        return self._instances[name]
    
    def get_all_transformers(self) -> List[str]:
        """Get list of all registered transformer names"""
        return list(self._transformers.keys())
    
    def apply_all(self, df: pd.DataFrame, feature_types: Dict[str, str], 
                  exclude: Optional[List[str]] = None) -> List[TransformationResult]:
        """
        Apply all registered transformers to a dataframe
        
        Args:
            df: Input dataframe
            feature_types: Dictionary mapping column names to their types
            exclude: List of transformer names to exclude
            
        Returns:
            List of all transformation results
        """
        exclude = exclude or []
        all_results = []
        
        for name in self._transformers:
            if name in exclude:
                continue
                
            transformer = self.create(name)
            applicable_columns = transformer.get_applicable_columns(df, feature_types)
            
            if applicable_columns:
                results = transformer.transform(df, applicable_columns)
                all_results.extend(results)
        
        return all_results


# Global registry instance
transformer_registry = TransformerRegistry()


def register_transformer(name: str):
    """Decorator to register a transformer class"""
    def decorator(cls: Type[BaseTransformer]):
        transformer_registry.register(name, cls)
        return cls
    return decorator