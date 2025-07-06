# DataAlchemy: Multi-Agent Feature Engineering System

![DataAlchemy](data-alchemy.png)

A powerful multi-agent system that automatically engineers features from any CSV/Parquet file using specialized AI agents. Built with a modern service-oriented architecture for maximum maintainability and extensibility.

## 🚀 Quick Start

### Simple Usage (Recommended)

```python
from data_alchemy import run_data_alchemy

# Transform your data with one function call
results = run_data_alchemy(
    file_path="data.csv",
    target_column="target",  # Optional for supervised learning
    output_path="features.parquet",
    performance_mode="medium"  # fast, medium, or thorough
)

print(f"Created {len(results['selection'].selected_features)} features")
```

### Advanced Usage (Async)

```python
from data_alchemy import DataAlchemy, PerformanceMode
import asyncio

async def advanced_example():
    # Create DataAlchemy instance with custom settings
    alchemy = DataAlchemy(performance_mode=PerformanceMode.THOROUGH)
    
    # Run the complete pipeline
    results = await alchemy.transform_data(
        file_path="large_dataset.parquet",
        target_column="target",
        output_path="engineered_features.parquet",
        sample_size=50000,  # For testing on subset
        evaluation_output="evaluation_report.json"
    )
    
    # Get detailed summary
    summary = alchemy.get_pipeline_summary(results)
    return results, summary

# Run the async function
results, summary = asyncio.run(advanced_example())
```

## 🏗️ System Architecture

DataAlchemy features a **service-oriented architecture** that separates concerns for better maintainability:

### Service Layer
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   DataService   │    │ OrchestrationSvc│    │  OutputService  │    │ DisplayService  │
│                 │    │                 │    │                 │    │                 │
│ • Data loading  │    │ • Agent coord.  │    │ • File saving   │    │ • Console UI    │
│ • Validation    │    │ • Pipeline mgmt │    │ • Report gen.   │    │ • Progress bars │
│ • Preparation   │    │ • Error handling│    │ • Recipe export │    │ • Rich tables   │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Agent Pipeline
The four specialized AI agents work together in sequence:

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

### Key Architectural Benefits

- **Maintainability**: Each service has a single responsibility
- **Extensibility**: Easy to add new services or modify existing ones
- **Testability**: Services can be tested independently
- **Reusability**: Services can be used in different contexts

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

## 🔧 Configuration

DataAlchemy supports customization through environment variables and a `.env` file:

```bash
# Copy the example configuration
cp .env.example .env

# Edit .env to customize settings
```

### Available Settings

```bash
# Model Provider (openai, anthropic, gemini, grok)
MODEL_PROVIDER=anthropic

# Model Selection
# OpenAI: gpt-4o, gpt-4o-mini, gpt-4-turbo-preview, gpt-3.5-turbo
# Anthropic: claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307
# Gemini: gemini-pro, gemini-1.5-pro-latest
# Grok: grok-beta
MODEL_NAME=claude-3-sonnet-20240229

# API Keys (set as environment variables)
export OPENAI_API_KEY=your-key-here
export ANTHROPIC_API_KEY=your-key-here
export GOOGLE_API_KEY=your-key-here
export XAI_API_KEY=your-key-here

# Feature Engineering Settings
MAX_POLYNOMIAL_DEGREE=3
MAX_INTERACTIONS=100
MAX_CARDINALITY=20
```

## 📦 Installation

```bash
# Create virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install pydantic pydantic-ai pandas numpy scikit-learn scipy pyarrow rich python-dotenv structlog
```

## 💡 Usage Examples

### Basic Usage
```python
from data_alchemy import run_data_alchemy

# Unsupervised feature engineering
results = run_data_alchemy("sales_data.csv")

# Supervised with target column
results = run_data_alchemy(
    file_path="customer_data.csv",
    target_column="churn",
    output_path="features.parquet"
)
```

### Advanced Usage with Services
```python
from data_alchemy import DataAlchemy, PerformanceMode

# Create instance with custom configuration
alchemy = DataAlchemy(performance_mode=PerformanceMode.THOROUGH)

# Use fast mode for quick exploration
results = await alchemy.transform_data(
    file_path="large_dataset.parquet",
    target_column="revenue",
    sample_size=10000,  # Process only first 10k rows
    evaluation_output="detailed_report.json"
)

# Access individual agent results
profile = results['profile']
print(f"Data quality score: {profile.quality_score}")
print(f"Suggested ML task: {profile.suggested_task_type}")

features = results['engineering']
print(f"Created {features.total_features_created} new features")

validation = results['validation']
print(f"Quality score: {validation.overall_quality_score}")

# Get pipeline summary
summary = alchemy.get_pipeline_summary(results)
print(f"Processing time: {sum(summary['processing_times'].values()):.2f}s")
```

### Error Handling
```python
from data_alchemy import run_data_alchemy, TargetNotFoundError, InsufficientDataError

try:
    results = run_data_alchemy(
        file_path="data.csv",
        target_column="target",
        output_path="features.parquet"
    )
except TargetNotFoundError as e:
    print(f"Target column not found: {e}")
    print(f"Available columns: {e.details['available_columns']}")
except InsufficientDataError as e:
    print(f"Need more data: required {e.details['required_rows']}, got {e.details['actual_rows']}")
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
│   ├── services/        # Service layer
│   │   ├── data_service.py        # Data loading & validation
│   │   ├── orchestration_service.py  # Pipeline coordination
│   │   ├── output_service.py      # File output & reports
│   │   └── display_service.py     # Console UI & progress
│   ├── models/          # Pydantic data models
│   ├── transformers/    # Feature transformation system
│   ├── core/           # Configuration & utilities
│   ├── utils/          # File handling utilities
│   └── data_alchemy.py # Main orchestrator (refactored)
├── docs/               # API documentation
├── examples/           # Example scripts and data
└── tests/             # Unit tests
```

### Adding New Features

1. **New Transformations**: Add to transformer registry in `transformers/`
2. **New Services**: Create new service in `services/` directory
3. **New Selection Methods**: Extend `CuratorAgent` or create new service
4. **New Validation Checks**: Extend `ValidatorAgent` validation methods

### API Documentation

For detailed API reference, examples, and best practices, see:
- **[Complete API Reference](docs/api_reference.md)** - Comprehensive documentation
- **[Service Architecture Guide](docs/architecture.md)** - Technical details (coming soon)
- **[Developer Guide](docs/contributing.md)** - Contribution guidelines (coming soon)

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