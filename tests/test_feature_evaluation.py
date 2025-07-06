"""Test the feature evaluation and top features analysis"""
from src.data_alchemy import run_data_alchemy
from pathlib import Path
import json


def test_with_evaluation():
    """Test DataAlchemy with feature evaluation output"""
    print("Testing DataAlchemy with Feature Evaluation")
    print("=" * 60)
    
    # Test with financial data (classification)
    results = run_data_alchemy(
        file_path="examples/data/financial_loans.csv",
        target_column="default",
        output_path="test_features.csv",
        evaluation_output="feature_evaluation.json",
        performance_mode="medium",
        sample_size=500
    )
    
    print("\n" + "=" * 60)
    print("Feature Evaluation Saved!")
    print("=" * 60)
    
    # Load and display the evaluation JSON
    if Path("feature_evaluation.json").exists():
        with open("feature_evaluation.json", "r") as f:
            evaluation = json.load(f)
        
        print("\n📄 Evaluation Summary from JSON:")
        print(f"  Timestamp: {evaluation['timestamp']}")
        print(f"  Task Type: {evaluation['profile']['task_type']}")
        print(f"  Data Quality: {evaluation['profile']['quality_score']:.2%}")
        print(f"  Original Features: {evaluation['engineering']['original_features']}")
        print(f"  Engineered Features: {evaluation['engineering']['engineered_features']}")
        print(f"  Selected Features: {evaluation['selection']['selected_features']}")
        print(f"  Validation Quality: {evaluation['validation']['quality_score']:.2%}")
        
        print("\n📊 Top 3 Features from JSON:")
        for i, feat in enumerate(evaluation['features'][:3]):
            print(f"  {i+1}. {feat['name']} (importance: {feat['importance']:.3f})")
    
    # Clean up
    Path("test_features.csv").unlink(missing_ok=True)
    Path("feature_evaluation.json").unlink(missing_ok=True)


def test_portfolio_analysis():
    """Test with portfolio data to see top features"""
    print("\n\nTesting Top Features Analysis with Portfolio Data")
    print("=" * 60)
    
    # Test with portfolio data
    results = run_data_alchemy(
        file_path="examples/portfolios_10k.csv",
        performance_mode="medium",
        sample_size=1000,
        evaluation_output="portfolio_evaluation.json"
    )
    
    # The top features analysis is now displayed automatically!
    
    # Clean up
    Path("portfolio_evaluation.json").unlink(missing_ok=True)


if __name__ == "__main__":
    test_with_evaluation()
    test_portfolio_analysis()