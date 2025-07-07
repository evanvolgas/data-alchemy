#!/usr/bin/env python3
"""Test script to verify the performance comparison issue in comprehensive_example.py"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from pathlib import Path
import os
import sys

# Add parent directory to path to import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import run_data_alchemy


def evaluate_dataset(file_path, target_column, task_type='regression'):
    """Evaluate a dataset with and without feature engineering"""
    
    print(f"\n=== Evaluating {Path(file_path).stem} ===")
    
    # Load original data
    df_original = pd.read_parquet(file_path)
    
    # Prepare original features
    X_original = df_original.drop(columns=[target_column])
    y = df_original[target_column]
    
    # Handle categorical columns in original dataset
    cat_cols = X_original.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        X_original = pd.get_dummies(X_original, columns=cat_cols, drop_first=True)
    
    # Drop datetime columns
    datetime_cols = X_original.select_dtypes(include=['datetime64']).columns
    if len(datetime_cols) > 0:
        X_original = X_original.drop(columns=datetime_cols)
    
    # Run DataAlchemy
    output_path = f'/tmp/{Path(file_path).stem}_test_features.parquet'
    results = run_data_alchemy(
        file_path=file_path,
        target_column=target_column,
        output_path=output_path,
        performance_mode='thorough'
    )
    
    if not results:
        print("DataAlchemy failed!")
        return
    
    # Load engineered features
    df_engineered = pd.read_parquet(output_path)
    
    # Remove target if present
    if target_column in df_engineered.columns:
        df_engineered = df_engineered.drop(columns=[target_column])
    
    # Split data with same random state
    X_orig_train, X_orig_test, y_train, y_test = train_test_split(
        X_original, y, test_size=0.2, random_state=42
    )
    X_eng_train, X_eng_test, _, _ = train_test_split(
        df_engineered, y, test_size=0.2, random_state=42
    )
    
    # Train models
    if task_type == 'regression':
        model_orig = RandomForestRegressor(n_estimators=100, random_state=42)
        model_eng = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model_orig = RandomForestClassifier(n_estimators=100, random_state=42)
        model_eng = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Fit models
    model_orig.fit(X_orig_train, y_train)
    model_eng.fit(X_eng_train, y_train)
    
    # Predictions
    pred_orig = model_orig.predict(X_orig_test)
    pred_eng = model_eng.predict(X_eng_test)
    
    # Calculate and display metrics
    print(f"\nOriginal features: {X_original.shape[1]}")
    print(f"Engineered features: {df_engineered.shape[1]}")
    print(f"Feature engineering summary: {results['summary']}")
    
    if task_type == 'regression':
        mse_orig = mean_squared_error(y_test, pred_orig)
        mse_eng = mean_squared_error(y_test, pred_eng)
        r2_orig = r2_score(y_test, pred_orig)
        r2_eng = r2_score(y_test, pred_eng)
        
        print(f"\nOriginal R²: {r2_orig:.4f}")
        print(f"Engineered R²: {r2_eng:.4f}")
        print(f"R² difference: {r2_eng - r2_orig:.4f}")
        print(f"Relative R² improvement: {(r2_eng - r2_orig) / (1 - r2_orig) * 100:.1f}%")
        
        print(f"\nOriginal MSE: {mse_orig:.2f}")
        print(f"Engineered MSE: {mse_eng:.2f}")
        print(f"MSE reduction: {(mse_orig - mse_eng) / mse_orig * 100:.1f}%")
    else:
        acc_orig = accuracy_score(y_test, pred_orig)
        acc_eng = accuracy_score(y_test, pred_eng)
        
        print(f"\nOriginal accuracy: {acc_orig:.4f}")
        print(f"Engineered accuracy: {acc_eng:.4f}")
        print(f"Accuracy difference: {acc_eng - acc_orig:.4f}")
        print(f"Accuracy difference (percentage points): {(acc_eng - acc_orig) * 100:.1f}%")
        
        # Show what the comprehensive example displays
        print(f"\nWhat comprehensive_example.py shows:")
        print(f"  Displayed as: +{(acc_eng - acc_orig) * 100:.1f}% (BUG: always shows + sign)")
        print(f"  Actual change: {'+' if acc_eng > acc_orig else ''}{(acc_eng - acc_orig) * 100:.1f}%")
        
        # Feature importance analysis
        if hasattr(model_eng, 'feature_importances_'):
            importances = model_eng.feature_importances_
            indices = np.argsort(importances)[::-1]
            print(f"\nTop 10 most important features:")
            for i in range(min(10, len(indices))):
                print(f"  {i+1}. {df_engineered.columns[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # Clean up
    os.remove(output_path)
    recipe_path = output_path.replace('.parquet', '_recipe.json')
    if os.path.exists(recipe_path):
        os.remove(recipe_path)


def main():
    """Test the performance issue with all three datasets"""
    
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    # Test each dataset
    datasets = [
        ("simple_demo.parquet", "target", "regression"),
        ("customer_churn.parquet", "churn", "classification"),
        ("sales_forecast.parquet", "sales", "regression")
    ]
    
    for filename, target, task_type in datasets:
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            evaluate_dataset(file_path, target, task_type)
        else:
            print(f"File not found: {file_path}")


if __name__ == "__main__":
    main()