# DataAlchemy API Reference

## Overview

DataAlchemy provides a comprehensive API for automated feature engineering using a multi-agent system. The API is designed to be both powerful for advanced users and simple for common use cases.

## Quick Start

### Simple Usage

```python
from data_alchemy import run_data_alchemy

# Basic feature engineering
results = run_data_alchemy(
    file_path="data.csv",
    target_column="target",
    output_path="features.parquet"
)

print(f"Created {len(results['selection'].selected_features)} features")
```

### Advanced Usage

```python
from data_alchemy import DataAlchemy, PerformanceMode
import asyncio

async def advanced_example():
    # Create DataAlchemy instance with custom settings
    alchemy = DataAlchemy(performance_mode=PerformanceMode.THOROUGH)
    
    # Run the pipeline
    results = await alchemy.transform_data(
        file_path="large_dataset.parquet",
        target_column="sales_amount",
        output_path="engineered_features.parquet",
        sample_size=10000,  # For testing on subset
        evaluation_output="evaluation_report.json"
    )
    
    # Get pipeline summary
    summary = alchemy.get_pipeline_summary(results)
    print(f"Pipeline Summary: {summary}")

# Run async function
asyncio.run(advanced_example())
```

## Core Classes

### DataAlchemy

The main orchestrator class for the feature engineering pipeline.

#### Constructor

```python
DataAlchemy(performance_mode: PerformanceMode = PerformanceMode.MEDIUM)
```

**Parameters:**
- `performance_mode` (PerformanceMode): Processing mode that controls speed vs thoroughness
  - `FAST`: Quick processing with basic transformations
  - `MEDIUM`: Balanced approach with most transformations (default)
  - `THOROUGH`: Comprehensive processing with all transformations and validation

#### Methods

##### transform_data

```python
async def transform_data(
    file_path: Union[str, Path],
    target_column: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    sample_size: Optional[int] = None,
    evaluation_output: Optional[Union[str, Path]] = None
) -> Dict[str, Any]
```

Execute the complete feature engineering pipeline.

**Parameters:**
- `file_path` (str or Path): Path to input CSV or Parquet file
- `target_column` (str, optional): Target column for supervised learning
- `output_path` (str or Path, optional): Path to save engineered features
- `sample_size` (int, optional): Limit rows for testing (useful for large datasets)
- `evaluation_output` (str or Path, optional): Path to save detailed evaluation report

**Returns:**
- `dict`: Results dictionary containing:
  - `file_info`: Input file metadata
  - `profile`: Data profiling results (DataProfileResult)
  - `engineering`: Feature engineering results (FeatureEngineeringResult)
  - `selection`: Feature selection results (FeatureSelectionResult)
  - `validation`: Feature validation results (ValidationResult)

**Example:**
```python
results = await alchemy.transform_data(
    file_path="sales_data.csv",
    target_column="revenue",
    output_path="features.parquet"
)

# Access specific results
profile = results['profile']
print(f"Data quality score: {profile.quality_score}")

engineering = results['engineering']
print(f"Created {engineering.total_features_created} features")

selection = results['selection']
print(f"Selected {len(selection.selected_features)} features")
```

##### get_pipeline_summary

```python
def get_pipeline_summary(results: Dict[str, Any]) -> Dict[str, Any]
```

Generate a concise summary of pipeline execution.

**Parameters:**
- `results` (dict): Results dictionary from transform_data

**Returns:**
- `dict`: Summary with key metrics including data quality, features created/selected, validation scores, and processing times

## Convenience Functions

### run_data_alchemy

```python
def run_data_alchemy(
    file_path: Union[str, Path],
    target_column: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    sample_size: Optional[int] = None,
    evaluation_output: Optional[Union[str, Path]] = None,
    performance_mode: str = "medium"
) -> Dict[str, Any]
```

Synchronous wrapper for the complete feature engineering pipeline.

**Parameters:** (Same as DataAlchemy.transform_data, plus:)
- `performance_mode` (str): Performance mode as string ("fast", "medium", "thorough")

**Returns:**
- `dict`: Complete pipeline results

**Example:**
```python
# Process a dataset with classification target
results = run_data_alchemy(
    file_path="customer_data.csv",
    target_column="churn",
    output_path="customer_features.csv",
    performance_mode="thorough",
    evaluation_output="model_evaluation.json"
)

# Check validation quality
quality_score = results['validation'].overall_quality_score
print(f"Feature quality: {quality_score:.2f}")
```

## Data Models

### PerformanceMode

Enumeration controlling the thoroughness of processing.

```python
class PerformanceMode(str, Enum):
    FAST = "fast"        # Quick processing, basic features
    MEDIUM = "medium"    # Balanced approach (default)
    THOROUGH = "thorough" # Comprehensive processing
```

### DataProfileResult

Results from the data profiling stage (Scout Agent).

**Key Attributes:**
- `quality_score` (float): Overall data quality score (0-1)
- `suggested_task_type` (MLTaskType): Detected task type (classification/regression)
- `domain_hints` (List[str]): Insights about the data domain
- `warnings` (List[str]): Data quality warnings
- `processing_time_seconds` (float): Time taken for profiling

### FeatureEngineeringResult

Results from the feature engineering stage (Alchemist Agent).

**Key Attributes:**
- `original_features` (List[Feature]): Original input features
- `engineered_features` (List[Feature]): Newly created features
- `total_features_created` (int): Count of engineered features
- `engineering_strategies_used` (List[str]): Applied transformation strategies
- `memory_estimate_mb` (float): Estimated memory usage
- `warnings` (List[str]): Engineering warnings

### FeatureSelectionResult

Results from the feature selection stage (Curator Agent).

**Key Attributes:**
- `selected_features` (List[SelectedFeature]): Features chosen for the final set
- `removed_features` (List[dict]): Features removed and reasons
- `dimensionality_reduction` (dict): Reduction statistics
- `performance_impact_estimate` (dict): Expected performance effects
- `recommendations` (List[str]): Selection recommendations

### ValidationResult

Results from the feature validation stage (Validator Agent).

**Key Attributes:**
- `overall_quality_score` (float): Final quality score (0-1)
- `passed_checks` (List[ValidationCheck]): Successful validation checks
- `failed_checks` (List[ValidationCheck]): Failed validation checks
- `issues` (List[ValidationIssue]): Detailed issues found
- `leakage_risk_features` (List[str]): Features with potential data leakage
- `recommendations` (List[str]): Validation recommendations

## Configuration

### Environment Variables

DataAlchemy can be configured using environment variables:

```bash
# Model configuration
MODEL_PROVIDER=anthropic  # or openai, gemini, grok
MODEL_NAME=claude-3-5-sonnet-20241022

# Processing configuration
MIN_ROWS_REQUIRED=100
EPSILON=1e-8
MAX_EXP_INPUT=700

# Transformer configuration (optional)
POLYNOMIAL_MAX_DEGREE=3
INTERACTION_MAX_FEATURES=100
```

### .env File

Create a `.env` file in your project root:

```env
MODEL_PROVIDER=anthropic
MODEL_NAME=claude-3-5-sonnet-20241022
MIN_ROWS_REQUIRED=100
```

## Error Handling

DataAlchemy provides specific exceptions for different error conditions:

```python
from data_alchemy import (
    DataAlchemyError,
    TargetNotFoundError,
    InsufficientDataError,
    AgentError
)

try:
    results = run_data_alchemy(
        file_path="data.csv",
        target_column="nonexistent_column"
    )
except TargetNotFoundError as e:
    print(f"Target column not found: {e}")
    print(f"Available columns: {e.details['available_columns']}")
    
except InsufficientDataError as e:
    print(f"Need more data: {e}")
    print(f"Required: {e.details['required_rows']}, Got: {e.details['actual_rows']}")
    
except DataAlchemyError as e:
    print(f"General error: {e}")
```

## Best Practices

### Performance Optimization

```python
# For large datasets, use sampling during development
results = run_data_alchemy(
    file_path="large_dataset.parquet",
    target_column="target",
    sample_size=50000,  # Work with subset first
    performance_mode="fast"  # Quick iteration
)

# Once satisfied, run full processing
final_results = run_data_alchemy(
    file_path="large_dataset.parquet",
    target_column="target",
    performance_mode="thorough",  # Full processing
    output_path="final_features.parquet"
)
```

### Feature Quality Assessment

```python
results = run_data_alchemy(
    file_path="data.csv",
    target_column="target",
    evaluation_output="quality_report.json"
)

# Check data quality
profile = results['profile']
if profile.quality_score < 0.7:
    print("Warning: Low data quality detected")
    for warning in profile.warnings:
        print(f"  - {warning}")

# Check feature validation
validation = results['validation']
if validation.overall_quality_score < 0.8:
    print("Warning: Feature quality issues detected")
    for issue in validation.issues:
        if issue.severity in ['critical', 'high']:
            print(f"  - {issue.severity.upper()}: {issue.description}")
```

### Iterative Development

```python
# Start with fast mode for rapid iteration
quick_results = run_data_alchemy(
    file_path="data.csv",
    target_column="target",
    performance_mode="fast"
)

# Review results and adjust if needed
print(f"Quick run created {len(quick_results['selection'].selected_features)} features")

# Run thorough analysis when ready
final_results = run_data_alchemy(
    file_path="data.csv",
    target_column="target",
    performance_mode="thorough",
    output_path="production_features.parquet",
    evaluation_output="production_evaluation.json"
)
```

## Integration Examples

### Scikit-learn Integration

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Engineer features
results = run_data_alchemy(
    file_path="data.csv",
    target_column="target",
    output_path="features.csv"
)

# Load engineered features
df = pd.read_csv("features.csv")
X = df.drop('target', axis=1)
y = df['target']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
```

### Pandas Integration

```python
import pandas as pd

# Load and process data
df = pd.read_csv("raw_data.csv")

# Apply feature engineering
results = run_data_alchemy(
    file_path="raw_data.csv",
    target_column="target",
    output_path="engineered.parquet"
)

# Work with engineered features
engineered_df = pd.read_parquet("engineered.parquet")

# Combine with additional manual features
engineered_df['custom_feature'] = engineered_df['feature1'] / engineered_df['feature2']

# Continue with modeling pipeline
```

## Troubleshooting

### Common Issues

1. **Memory Issues with Large Datasets**
   ```python
   # Use sampling for large datasets
   results = run_data_alchemy(
       file_path="large_data.csv",
       sample_size=100000,  # Limit rows
       performance_mode="fast"  # Reduce memory usage
   )
   ```

2. **Target Column Not Found**
   ```python
   # Check column names first
   df = pd.read_csv("data.csv")
   print("Available columns:", df.columns.tolist())
   
   results = run_data_alchemy(
       file_path="data.csv",
       target_column="correct_target_name"
   )
   ```

3. **Low Feature Quality Scores**
   ```python
   # Review data quality first
   results = run_data_alchemy(
       file_path="data.csv",
       evaluation_output="diagnosis.json"
   )
   
   # Check for data quality issues
   with open("diagnosis.json") as f:
       report = json.load(f)
       print("Data quality score:", report['summary']['data_quality_score'])
   ```

### Performance Tuning

- Use `PerformanceMode.FAST` for rapid prototyping
- Use `sample_size` parameter for large datasets during development
- Monitor memory usage with memory estimates in results
- Save intermediate results to avoid recomputation

### Logging and Debugging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Run with detailed logging
results = run_data_alchemy(
    file_path="data.csv",
    target_column="target",
    performance_mode="medium"
)
```