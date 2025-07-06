"""Test the complete DataAlchemy system"""
from src.data_alchemy import run_data_alchemy
from pathlib import Path


def test_financial_classification():
    """Test with financial data and classification target"""
    print("Testing DataAlchemy with financial classification data...")
    
    results = run_data_alchemy(
        file_path="examples/data/financial_loans.csv",
        target_column="default",
        output_path="test_output_financial.csv",
        performance_mode="fast",
        sample_size=100
    )
    
    print(f"\n✅ Profile Quality Score: {results['profile'].quality_score:.2f}")
    print(f"✅ Features Engineered: {results['engineering'].total_features_created}")
    print(f"✅ Features Selected: {len(results['selection'].selected_features)}")
    print(f"✅ Validation Quality Score: {results['validation'].overall_quality_score:.2f}")
    
    # Check output file was created
    if Path("test_output_financial.csv").exists():
        print("✅ Output file created successfully")
        Path("test_output_financial.csv").unlink()  # Clean up
    
    return results


def test_ecommerce_unsupervised():
    """Test with e-commerce data without target"""
    print("\nTesting DataAlchemy with e-commerce unsupervised data...")
    
    results = run_data_alchemy(
        file_path="examples/data/ecommerce_transactions.csv",
        target_column=None,
        output_path="test_output_ecommerce.parquet",
        performance_mode="medium",
        sample_size=200
    )
    
    print(f"\n✅ Profile Quality Score: {results['profile'].quality_score:.2f}")
    print(f"✅ Features Engineered: {results['engineering'].total_features_created}")
    print(f"✅ Features Selected: {len(results['selection'].selected_features)}")
    print(f"✅ Validation Quality Score: {results['validation'].overall_quality_score:.2f}")
    
    # Check output file was created
    if Path("test_output_ecommerce.parquet").exists():
        print("✅ Output file created successfully")
        Path("test_output_ecommerce.parquet").unlink()  # Clean up
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("DataAlchemy Complete System Test")
    print("=" * 60)
    
    try:
        # Test 1: Financial classification
        financial_results = test_financial_classification()
        
        # Test 2: E-commerce unsupervised
        ecommerce_results = test_ecommerce_unsupervised()
        
        print("\n" + "=" * 60)
        print("✨ All tests passed successfully! ✨")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise