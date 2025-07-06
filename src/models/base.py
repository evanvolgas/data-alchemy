from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class MLTaskType(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    MULTICLASS = "multiclass"
    UNSUPERVISED = "unsupervised"
    UNKNOWN = "unknown"


class PerformanceMode(str, Enum):
    FAST = "fast"
    MEDIUM = "medium"
    THOROUGH = "thorough"


class DataType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BINARY = "binary"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


class ColumnProfile(BaseModel):
    name: str
    dtype: DataType
    missing_count: int
    missing_percentage: float
    unique_count: int
    unique_percentage: float
    sample_values: List[Any] = Field(default_factory=list)
    statistics: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


class DataContext(BaseModel):
    """Shared state across all agents"""
    file_path: str
    target_column: Optional[str] = None
    performance_mode: PerformanceMode = PerformanceMode.MEDIUM
    n_rows: int
    n_columns: int
    columns: List[str]
    dtypes: Dict[str, str]
    memory_usage_mb: float
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DataProfileResult(BaseModel):
    """Scout Agent output"""
    context: DataContext
    column_profiles: List[ColumnProfile]
    suggested_task_type: MLTaskType
    domain_hints: List[str] = Field(
        description="Domain-specific insights based on column names and content"
    )
    quality_score: float = Field(
        ge=0.0, le=1.0,
        description="Overall data quality score (0-1)"
    )
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    processing_time_seconds: float