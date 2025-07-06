from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
import numpy as np


class SelectionMethod(str, Enum):
    MUTUAL_INFORMATION = "mutual_information"
    RANDOM_FOREST = "random_forest"
    CORRELATION = "correlation"
    VARIANCE_THRESHOLD = "variance_threshold"
    STATISTICAL_TEST = "statistical_test"
    L1_REGULARIZATION = "l1_regularization"
    SHAP = "shap"
    PERMUTATION = "permutation"
    CHI_SQUARED = "chi_squared"
    TARGET_ENCODING = "target_encoding"
    GROUP_IMPORTANCE = "group_importance"
    INTERACTION = "interaction"
    STABILITY = "stability"
    DEFAULT = "default"
    HEURISTIC = "heuristic"


class SelectedFeature(BaseModel):
    name: str
    importance_score: float
    selection_methods: List[SelectionMethod]
    rank: int
    correlation_cluster: Optional[int] = None
    redundancy_score: float = Field(
        ge=0.0, le=1.0,
        description="Score indicating redundancy with other features (0=unique, 1=redundant)"
    )
    interpretability_score: float = Field(
        ge=0.0, le=1.0,
        description="How easy the feature is to interpret (0=complex, 1=simple)"
    )


class FeatureSelectionResult(BaseModel):
    """Curator Agent output"""
    selected_features: List[SelectedFeature]
    removed_features: List[Dict[str, Any]]
    selection_summary: Dict[str, int] = Field(
        description="Summary of features by selection reason"
    )
    correlation_matrix_summary: Dict[str, Any] = Field(
        description="Summary statistics of the correlation matrix"
    )
    dimensionality_reduction: Dict[str, Any] = Field(
        description="Before/after dimensionality comparison"
    )
    performance_impact_estimate: Dict[str, float] = Field(
        description="Estimated impact on model performance metrics"
    )
    recommendations: List[str] = Field(default_factory=list)
    processing_time_seconds: float