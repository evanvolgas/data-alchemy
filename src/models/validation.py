from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from enum import Enum


class ValidationCheck(str, Enum):
    DATA_LEAKAGE = "data_leakage"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    FEATURE_STABILITY = "feature_stability"
    CROSS_VALIDATION = "cross_validation"
    DISTRIBUTION_SHIFT = "distribution_shift"
    MULTICOLLINEARITY = "multicollinearity"
    CLASS_IMBALANCE = "class_imbalance"


class ValidationIssue(BaseModel):
    check_type: ValidationCheck
    severity: Literal["low", "medium", "high", "critical"]
    feature_names: List[str]
    description: str
    recommendation: str
    metrics: Dict[str, Any] = Field(default_factory=dict)


class CrossValidationResult(BaseModel):
    mean_score: float
    std_score: float
    fold_scores: List[float]
    feature_importance_stability: Dict[str, float]
    warnings: List[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """Validator Agent output"""
    passed_checks: List[ValidationCheck]
    failed_checks: List[ValidationCheck]
    issues: List[ValidationIssue]
    cross_validation_results: Optional[CrossValidationResult] = None
    feature_stability_scores: Dict[str, float] = Field(
        description="Stability score for each feature across data splits"
    )
    leakage_risk_features: List[str] = Field(
        description="Features with potential data leakage"
    )
    overall_quality_score: float = Field(
        ge=0.0, le=1.0,
        description="Overall feature set quality (0-1)"
    )
    recommendations: List[str]
    warnings: List[str]
    processing_time_seconds: float