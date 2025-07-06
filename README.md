# DataAlchemy: Multi-Agent Feature Engineering System

![DataAlchemy](data-alchemy.png)

A multi-agent system that automatically engineers features from any CSV/Parquet file using specialized AI agents powered by PydanticAI.

## 🚀 Quick Start

```python
from src.data_alchemy import run_data_alchemy

# Transform your data with one function call
results = run_data_alchemy(
    file_path="data.csv",
    target_column="target",  # Optional for supervised learning
    output_path="features.parquet",
    performance_mode="medium"  # fast, medium, or thorough
)
```

## 🏗️ System Architecture

DataAlchemy uses four specialized AI agents that work together in a pipeline:

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Raw Data  │     │Scout Agent  │     │Alchemist     │     │Curator      │
│  CSV/Parquet│────▶│   Profile   │────▶│  Engineer    │────▶│  Select     │
└─────────────┘     └─────────────┘     └──────────────┘     └─────────────┘
                           │                     │                     │
                    ┌──────▼──────┐      ┌──────▼──────┐      ┌──────▼──────┐
                    │DataProfile  │      │Feature      │      │Feature      │
                    │Result       │      │Engineering  │      │Selection    │
                    │             │      │Result       │      │Result       │
                    └─────────────┘      └─────────────┘      └─────────────┘
                                                                       │
                                                                       ▼
┌─────────────┐     ┌─────────────┐                          ┌─────────────┐
│Final Output │     │Validator    │                          │  Selected   │
│  Features   │◀────│  Validate   │◀─────────────────────────│  Features   │
└─────────────┘     └─────────────┘                          └─────────────┘
                           │
                    ┌──────▼──────┐
                    │Validation   │
                    │Result       │
                    └─────────────┘
```

## 🤖 The Four Agents

### 1. Scout Agent (Data Profiler)
- **Role**: Analyzes your data to understand its structure and quality
- **Outputs**: 
  - Data type detection (numeric, categorical, datetime, text)
  - Quality metrics and missing data analysis
  - ML task type recommendation (classification/regression/unsupervised)
  - Domain insights (e.g., "financial data detected")

### 2. Alchemist Agent (Feature Creator)
- **Role**: Creates new features based on data characteristics
- **Transformations**:
  - **Numeric**: log, sqrt, polynomial, binning
  - **Categorical**: frequency encoding, one-hot encoding
  - **Datetime**: year, month, day, hour, cyclical features
  - **Text**: length, word count, pattern detection
  - **Interactions**: multiplication, ratios between numeric features

### 3. Curator Agent (Feature Selector)
- **Role**: Selects the most valuable features
- **Methods**:
  - Mutual information scoring
  - Random forest importance
  - Correlation analysis and redundancy removal
  - Variance threshold filtering
- **Balances**: Performance vs interpretability

### 4. Validator Agent (Quality Assurance)
- **Role**: Ensures feature quality and reliability
- **Checks**:
  - Data leakage detection
  - Feature stability across data splits
  - Cross-validation performance
  - Class imbalance detection
  - Multicollinearity analysis

## 🎯 Key Features

### Automatic Everything
- File format detection (CSV/Parquet)
- Data type inference
- Missing value handling
- Feature engineering strategy selection

### Performance Modes
- **Fast**: Quick profiling, basic features (~1-5 seconds)
- **Medium**: Balanced approach with interaction features (~5-30 seconds)
- **Thorough**: Comprehensive analysis with all features (~30+ seconds)

### Mathematical Transparency
Every feature includes:
- Clear description of the transformation
- Mathematical formula (e.g., `f(x) = ln(x)`)
- Computational complexity (e.g., `O(n)`)

### Production Ready
- Type-safe with Pydantic models
- Comprehensive error handling
- Progress tracking with rich terminal output
- Memory-efficient processing

## 📦 Installation

```bash
# Create virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install pydantic pydantic-ai pandas numpy scikit-learn scipy pyarrow rich
```

## 💡 Usage Examples

### Basic Usage
```python
from src.data_alchemy import run_data_alchemy

# Unsupervised feature engineering
results = run_data_alchemy("sales_data.csv")

# Supervised with target column
results = run_data_alchemy(
    file_path="customer_data.csv",
    target_column="churn",
    output_path="features.parquet"
)
```

### Advanced Usage
```python
# Use fast mode for quick exploration
results = run_data_alchemy(
    file_path="large_dataset.parquet",
    target_column="revenue",
    performance_mode="fast",
    sample_size=10000  # Process only first 10k rows
)

# Access individual agent results
profile = results['profile']
print(f"Data quality score: {profile.quality_score}")
print(f"Suggested ML task: {profile.suggested_task_type}")

features = results['engineering']
print(f"Created {features.total_features_created} new features")

validation = results['validation']
print(f"Quality score: {validation.overall_quality_score}")
```

## 📊 Output Structure

The system returns a dictionary with results from each agent:

```python
{
    'file_info': {...},           # File metadata
    'profile': DataProfileResult,  # Scout agent output
    'engineering': FeatureEngineeringResult,  # Alchemist output
    'selection': FeatureSelectionResult,      # Curator output
    'validation': ValidationResult            # Validator output
}
```

## 🧪 Running Examples

```bash
# Generate sample datasets
uv run python examples/generate_sample_data.py

# Run comprehensive analysis on all datasets
uv run python examples/comprehensive_example.py

# This will:
# - Analyze multiple datasets (CSV and Parquet)
# - Show top original and engineered features
# - Save evaluation JSON files
# - Display performance comparisons
```

## 🔧 Development

### Project Structure
```
data-alchemy/
├── src/
│   ├── agents/          # Agent implementations
│   ├── models/          # Pydantic data models
│   ├── utils/           # File handling utilities
│   └── data_alchemy.py  # Main orchestrator
├── examples/            # Example scripts and data
└── tests/              # Unit tests
```

### Adding New Features

1. **New Transformations**: Add to `AlchemistAgent._engineer_*_features()`
2. **New Selection Methods**: Add to `CuratorAgent`
3. **New Validation Checks**: Add to `ValidatorAgent`

## ⚠️ Limitations

- Currently handles tabular data only (CSV/Parquet)
- Text features are basic (no embeddings or NLP)
- No support for image or time series specific features
- Memory constraints for very large datasets (>1GB)

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional feature engineering strategies
- Support for more file formats
- Advanced text processing
- GPU acceleration for large datasets
- Real-time streaming data support

## 📄 License

MIT License - See LICENSE file for details