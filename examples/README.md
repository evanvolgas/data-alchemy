# DataAlchemy Examples

This directory contains example usage of DataAlchemy with sample datasets designed to demonstrate feature engineering capabilities.

## Files

- `comprehensive_example.py` - Main example that runs DataAlchemy on all sample datasets
- `generate_sample_data.py` - Generates sample datasets with patterns that benefit from feature engineering
- `data/` - Directory containing sample datasets in parquet format
- `output/` - Directory where engineered features are saved

## Sample Datasets

1. **simple_demo.parquet** (500 rows)
   - Simple dataset with clear non-linear patterns
   - Benefits from polynomial and interaction features
   - Shows ~4% R² improvement with engineered features

2. **customer_churn.parquet** (2,000 rows)
   - Customer churn prediction dataset
   - Benefits from ratios, time-based features, and interactions
   - Engineered features capture usage patterns and customer behavior

3. **sales_forecast.parquet** (1,500 rows)
   - Sales forecasting dataset with temporal patterns
   - Benefits from seasonal decomposition and category interactions
   - Shows importance of price ratios and store-product interactions

## Running the Example

```bash
python comprehensive_example.py
```

This will:
1. Generate fresh sample datasets
2. Run DataAlchemy on each dataset
3. Compare model performance with and without engineered features
4. Display detailed results and feature importance scores

## Key Features Demonstrated

- **Automatic Feature Engineering**: Creates polynomial, interaction, ratio, and temporal features
- **Intelligent Feature Selection**: Uses multiple importance methods to select the most valuable features
- **Target Exclusion**: Properly excludes target column from feature engineering to prevent leakage
- **Feature Type Handling**: Handles numeric, categorical, and datetime features appropriately
- **Performance Comparison**: Shows actual model improvement from engineered features

## Results Summary

The example typically shows:
- **Simple Demo**: 20/30 engineered features selected, ~4% R² improvement
- **Customer Churn**: 37/60 engineered features selected, capturing complex patterns
- **Sales Forecast**: 28/47 engineered features selected, strong importance for ratios

## Output

For each dataset, DataAlchemy generates two files in the `output/` directory:

### 1. Feature Data Files (`.parquet`)
- `simple_demo_features.parquet`
- `customer_churn_features.parquet`
- `sales_forecast_features.parquet`

These contain the selected features plus the target column for easy model training.

### 2. Feature Recipe Files (`.json`)
- `simple_demo_features_recipe.json`
- `customer_churn_features_recipe.json`
- `sales_forecast_features_recipe.json`

These JSON files contain:
- List of selected features with importance scores
- Python code to recreate each engineered feature
- Source columns and transformation formulas
- Usage instructions

## Using Feature Recipes

The recipe files make it easy to understand and recreate the engineered features:

```python
# Example from a recipe file
{
  "name": "feature_1_cubed",
  "importance": 0.355,
  "type": "polynomial",
  "source_columns": ["feature_1"],
  "python_code": "df['feature_1_cubed'] = df['feature_1'] ** 3",
  "formula": "f(x) = x³ - captures cubic relationships"
}
```

To apply a recipe to new data, use the `use_recipe.py` example:

```bash
python use_recipe.py
```

This demonstrates how to:
1. Load a recipe file
2. Apply all transformations to create engineered features
3. Verify the features match DataAlchemy's output