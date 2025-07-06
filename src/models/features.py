from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from enum import Enum
import numpy as np


class FeatureType(str, Enum):
    """Simplified feature types for the system"""
    # Original features
    ORIGINAL = "original"
    
    # Mathematical transformations (consolidated)
    MATHEMATICAL = "mathematical"
    
    # Feature combinations
    INTERACTION = "interaction"
    
    # Categorical encodings (consolidated)
    CATEGORICAL = "categorical"
    
    # Temporal features (consolidated)
    TEMPORAL = "temporal"
    
    # Text features (consolidated)
    TEXT = "text"
    
    # Generic engineered feature
    ENGINEERED = "engineered"


class Feature(BaseModel):
    name: str
    type: FeatureType
    source_columns: List[str]
    description: str
    dtype: Optional[str] = None  # Data type of the feature (e.g., 'float64', 'int64', 'object')
    mathematical_explanation: Optional[str] = None
    computational_complexity: Literal["O(1)", "O(n)", "O(n²)", "O(n³)"] = "O(n)"
    importance_score: Optional[float] = None
    correlation_with_target: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FeatureEngineeringResult(BaseModel):
    """Alchemist Agent output"""
    original_features: List[Feature]
    engineered_features: List[Feature]
    total_features_created: int
    feature_matrix_shape: tuple[int, int]
    memory_estimate_mb: float
    engineering_strategies_used: List[str]
    warnings: List[str] = Field(default_factory=list)
    processing_time_seconds: float
    
    @property
    def all_features(self) -> List[Feature]:
        return self.original_features + self.engineered_features