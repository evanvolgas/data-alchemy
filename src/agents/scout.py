import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
from pathlib import Path
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel

from ..models import (
    DataContext,
    DataProfileResult,
    ColumnProfile,
    DataType,
    MLTaskType,
    PerformanceMode
)


class ScoutDependencies(BaseModel):
    """Dependencies for Scout Agent"""
    df: pd.DataFrame
    file_path: str
    target_column: Optional[str] = None
    performance_mode: PerformanceMode = PerformanceMode.MEDIUM
    
    class Config:
        arbitrary_types_allowed = True


scout_agent = Agent(
    "openai:gpt-4o-mini",
    deps_type=ScoutDependencies,
    system_prompt="""You are the Scout Agent, a data profiling expert.
    Your role is to analyze datasets and provide comprehensive profiling insights.
    Focus on:
    1. Understanding data types and distributions
    2. Detecting patterns that suggest feature engineering opportunities
    3. Recommending appropriate ML task types
    4. Providing domain-specific insights based on column names
    """
)


def infer_data_type(series: pd.Series, col_name: str) -> DataType:
    """Infer the semantic data type of a pandas Series"""
    if series.dtype == 'bool':
        return DataType.BINARY
    elif pd.api.types.is_numeric_dtype(series):
        if series.nunique() == 2:
            return DataType.BINARY
        return DataType.NUMERIC
    elif pd.api.types.is_datetime64_any_dtype(series):
        return DataType.DATETIME
    elif pd.api.types.is_categorical_dtype(series) or series.dtype == 'object':
        # Check if it might be datetime stored as string
        if col_name.lower() in ['date', 'datetime', 'timestamp', 'created_at', 'updated_at']:
            try:
                pd.to_datetime(series.dropna().head(100))
                return DataType.DATETIME
            except:
                pass
        
        # Check if it's text (long strings) vs categorical
        avg_length = series.dropna().astype(str).str.len().mean()
        if avg_length > 50:
            return DataType.TEXT
        return DataType.CATEGORICAL
    
    return DataType.UNKNOWN


def profile_column(series: pd.Series, col_name: str, sample_size: int = 10) -> ColumnProfile:
    """Generate profile for a single column"""
    missing_count = series.isna().sum()
    total_count = len(series)
    unique_count = series.nunique()
    
    # Get sample values
    non_null = series.dropna()
    if len(non_null) > 0:
        sample_values = non_null.sample(min(sample_size, len(non_null))).tolist()
    else:
        sample_values = []
    
    # Calculate statistics based on data type
    dtype = infer_data_type(series, col_name)
    statistics = {}
    warnings = []
    
    if dtype == DataType.NUMERIC:
        statistics = {
            "mean": float(series.mean()) if not series.empty else None,
            "std": float(series.std()) if not series.empty else None,
            "min": float(series.min()) if not series.empty else None,
            "max": float(series.max()) if not series.empty else None,
            "q25": float(series.quantile(0.25)) if not series.empty else None,
            "q50": float(series.quantile(0.50)) if not series.empty else None,
            "q75": float(series.quantile(0.75)) if not series.empty else None,
            "skewness": float(series.skew()) if len(series) > 2 else None,
            "kurtosis": float(series.kurtosis()) if len(series) > 3 else None,
        }
        
        # Check for potential issues
        if statistics.get("std", 0) == 0:
            warnings.append("Zero variance - constant column")
        if abs(statistics.get("skewness", 0)) > 2:
            warnings.append("Highly skewed distribution")
            
    elif dtype == DataType.CATEGORICAL:
        value_counts = series.value_counts()
        statistics = {
            "mode": value_counts.index[0] if len(value_counts) > 0 else None,
            "mode_frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            "cardinality": unique_count,
            "cardinality_ratio": unique_count / total_count if total_count > 0 else 0
        }
        
        # Check for high cardinality
        if statistics["cardinality_ratio"] > 0.5:
            warnings.append("High cardinality - might not be truly categorical")
            
    elif dtype == DataType.DATETIME:
        non_null_dates = pd.to_datetime(series.dropna(), errors='coerce')
        if len(non_null_dates) > 0:
            statistics = {
                "min_date": str(non_null_dates.min()),
                "max_date": str(non_null_dates.max()),
                "date_range_days": (non_null_dates.max() - non_null_dates.min()).days
            }
    
    # Check missing data
    missing_pct = missing_count / total_count if total_count > 0 else 0
    if missing_pct > 0.3:
        warnings.append(f"High missing rate: {missing_pct:.1%}")
    
    return ColumnProfile(
        name=col_name,
        dtype=dtype,
        missing_count=int(missing_count),
        missing_percentage=float(missing_pct),
        unique_count=int(unique_count),
        unique_percentage=float(unique_count / total_count if total_count > 0 else 0),
        sample_values=sample_values,
        statistics=statistics,
        warnings=warnings
    )


def suggest_task_type(df: pd.DataFrame, target_column: Optional[str], column_profiles: List[ColumnProfile]) -> MLTaskType:
    """Suggest the ML task type based on data characteristics"""
    if target_column is None:
        return MLTaskType.UNSUPERVISED
    
    if target_column not in df.columns:
        return MLTaskType.UNKNOWN
    
    target_profile = next((p for p in column_profiles if p.name == target_column), None)
    if not target_profile:
        return MLTaskType.UNKNOWN
    
    if target_profile.dtype == DataType.NUMERIC:
        return MLTaskType.REGRESSION
    elif target_profile.dtype in [DataType.CATEGORICAL, DataType.BINARY]:
        if target_profile.unique_count == 2:
            return MLTaskType.CLASSIFICATION
        else:
            return MLTaskType.MULTICLASS
    
    return MLTaskType.UNKNOWN


def get_domain_hints(column_names: List[str], column_profiles: List[ColumnProfile]) -> List[str]:
    """Detect domain-specific patterns from column names and content"""
    hints = []
    
    # Common domain patterns
    financial_keywords = ['price', 'cost', 'revenue', 'profit', 'amount', 'balance', 'transaction']
    temporal_keywords = ['date', 'time', 'year', 'month', 'day', 'timestamp', 'created', 'updated']
    customer_keywords = ['customer', 'user', 'client', 'person', 'name', 'email', 'phone']
    product_keywords = ['product', 'item', 'sku', 'category', 'brand', 'model']
    geo_keywords = ['location', 'city', 'country', 'state', 'zip', 'latitude', 'longitude', 'address']
    
    lower_columns = [col.lower() for col in column_names]
    
    # Check for domain patterns
    if any(keyword in ' '.join(lower_columns) for keyword in financial_keywords):
        hints.append("Financial/transactional data detected")
    
    if any(keyword in ' '.join(lower_columns) for keyword in temporal_keywords):
        hints.append("Temporal data detected - consider time-series features")
    
    if any(keyword in ' '.join(lower_columns) for keyword in customer_keywords):
        hints.append("Customer data detected - consider aggregation features")
    
    if any(keyword in ' '.join(lower_columns) for keyword in product_keywords):
        hints.append("Product/inventory data detected")
    
    if any(keyword in ' '.join(lower_columns) for keyword in geo_keywords):
        hints.append("Geographic data detected - consider spatial features")
    
    # Check for ID columns
    id_columns = [col for col in column_names if 'id' in col.lower() and col.lower() != 'id']
    if id_columns:
        hints.append(f"Multiple ID columns found ({len(id_columns)}) - possible relational structure")
    
    # Check for high cardinality text
    text_columns = [p.name for p in column_profiles if p.dtype == DataType.TEXT]
    if text_columns:
        hints.append(f"Text columns found ({len(text_columns)}) - consider NLP features")
    
    return hints


def calculate_quality_score(column_profiles: List[ColumnProfile]) -> float:
    """Calculate overall data quality score"""
    if not column_profiles:
        return 0.0
    
    scores = []
    for profile in column_profiles:
        col_score = 1.0
        
        # Penalize missing data
        col_score -= profile.missing_percentage * 0.5
        
        # Penalize constant columns
        if profile.unique_count <= 1:
            col_score *= 0.1
        
        # Penalize high cardinality for categorical
        if profile.dtype == DataType.CATEGORICAL and profile.unique_percentage > 0.5:
            col_score *= 0.7
        
        # Bonus for clean numeric columns
        if profile.dtype == DataType.NUMERIC and profile.missing_percentage == 0:
            col_score *= 1.1
        
        scores.append(max(0, min(1, col_score)))
    
    return sum(scores) / len(scores)


class ScoutAgent:
    """Scout Agent for data profiling and analysis"""
    
    def __init__(self):
        self.agent = scout_agent
    
    async def profile_data(
        self,
        df: pd.DataFrame,
        file_path: str,
        target_column: Optional[str] = None,
        performance_mode: PerformanceMode = PerformanceMode.MEDIUM
    ) -> DataProfileResult:
        """Profile the dataset and generate insights"""
        start_time = time.time()
        
        # Create data context
        context = DataContext(
            file_path=file_path,
            target_column=target_column,
            performance_mode=performance_mode,
            n_rows=len(df),
            n_columns=len(df.columns),
            columns=df.columns.tolist(),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            memory_usage_mb=df.memory_usage(deep=True).sum() / 1024 / 1024
        )
        
        # Profile each column
        column_profiles = []
        sample_size = {
            PerformanceMode.FAST: 5,
            PerformanceMode.MEDIUM: 10,
            PerformanceMode.THOROUGH: 20
        }[performance_mode]
        
        for col in df.columns:
            profile = profile_column(df[col], col, sample_size)
            column_profiles.append(profile)
        
        # Generate insights
        task_type = suggest_task_type(df, target_column, column_profiles)
        domain_hints = get_domain_hints(df.columns.tolist(), column_profiles)
        quality_score = calculate_quality_score(column_profiles)
        
        # Generate recommendations
        recommendations = []
        warnings = []
        
        # Check for issues
        constant_cols = [p.name for p in column_profiles if p.unique_count <= 1]
        if constant_cols:
            warnings.append(f"Constant columns detected: {', '.join(constant_cols)}")
            recommendations.append("Consider removing constant columns")
        
        high_missing = [p.name for p in column_profiles if p.missing_percentage > 0.3]
        if high_missing:
            warnings.append(f"High missing data in: {', '.join(high_missing)}")
            recommendations.append("Consider imputation strategies for missing data")
        
        # Check for datetime columns stored as strings
        potential_dates = [p for p in column_profiles 
                          if p.dtype == DataType.CATEGORICAL 
                          and any(kw in p.name.lower() for kw in ['date', 'time'])]
        if potential_dates:
            recommendations.append("Consider parsing datetime columns for temporal features")
        
        # Memory warnings
        if context.memory_usage_mb > 1000:
            warnings.append(f"Large dataset: {context.memory_usage_mb:.1f} MB")
            recommendations.append("Consider using chunking or sampling for large operations")
        
        processing_time = time.time() - start_time
        
        return DataProfileResult(
            context=context,
            column_profiles=column_profiles,
            suggested_task_type=task_type,
            domain_hints=domain_hints,
            quality_score=quality_score,
            recommendations=recommendations,
            warnings=warnings,
            processing_time_seconds=processing_time
        )