#!/usr/bin/env python3
"""
Comprehensive example demonstrating DataAlchemy's capabilities.
Shows how engineered features improve model performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_alchemy import run_data_alchemy

console = Console()


def evaluate_features(df_original, df_engineered, target_col, task_type='regression'):
    """Compare model performance with and without engineered features"""
    
    # Prepare datasets
    X_original = df_original.drop(columns=[target_col])
    # df_engineered should already have target removed, but check just in case
    if target_col in df_engineered.columns:
        X_engineered = df_engineered.drop(columns=[target_col])
    else:
        X_engineered = df_engineered
    y = df_original[target_col]
    
    # Handle categorical and datetime columns for both datasets
    # Original dataset
    cat_cols_orig = X_original.select_dtypes(include=['object']).columns
    if len(cat_cols_orig) > 0:
        X_original = pd.get_dummies(X_original, columns=cat_cols_orig, drop_first=True)
    
    # Drop datetime columns (they should have been engineered into numeric features)
    datetime_cols_orig = X_original.select_dtypes(include=['datetime64']).columns
    if len(datetime_cols_orig) > 0:
        X_original = X_original.drop(columns=datetime_cols_orig)
    
    # Engineered dataset
    cat_cols_eng = X_engineered.select_dtypes(include=['object']).columns
    if len(cat_cols_eng) > 0:
        X_engineered = pd.get_dummies(X_engineered, columns=cat_cols_eng, drop_first=True)
    
    # Drop datetime columns
    datetime_cols_eng = X_engineered.select_dtypes(include=['datetime64']).columns
    if len(datetime_cols_eng) > 0:
        X_engineered = X_engineered.drop(columns=datetime_cols_eng)
    
    # Split data
    X_orig_train, X_orig_test, y_train, y_test = train_test_split(
        X_original, y, test_size=0.2, random_state=42
    )
    X_eng_train, X_eng_test, _, _ = train_test_split(
        X_engineered, y, test_size=0.2, random_state=42
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
    
    # Calculate metrics
    if task_type == 'regression':
        mse_orig = mean_squared_error(y_test, pred_orig)
        mse_eng = mean_squared_error(y_test, pred_eng)
        r2_orig = r2_score(y_test, pred_orig)
        r2_eng = r2_score(y_test, pred_eng)
        
        improvement = {
            'mse_reduction': (mse_orig - mse_eng) / mse_orig * 100,
            'r2_improvement': (r2_eng - r2_orig) / (1 - r2_orig) * 100,
            'original_r2': r2_orig,
            'engineered_r2': r2_eng,
            'original_mse': mse_orig,
            'engineered_mse': mse_eng
        }
    else:
        acc_orig = accuracy_score(y_test, pred_orig)
        acc_eng = accuracy_score(y_test, pred_eng)
        
        improvement = {
            'accuracy_improvement': (acc_eng - acc_orig) * 100,
            'original_accuracy': acc_orig,
            'engineered_accuracy': acc_eng
        }
    
    return improvement, model_eng


def display_results(results, improvement, dataset_name):
    """Display analysis results in a formatted way"""
    
    # Header
    console.print(f"\n[bold cyan]═══ {dataset_name} Analysis Complete ═══[/bold cyan]\n")
    
    # Feature engineering summary
    eng_table = Table(title="Feature Engineering Summary")
    eng_table.add_column("Metric", style="cyan")
    eng_table.add_column("Value", style="green")
    
    eng_result = results['engineering']
    eng_table.add_row("Original Features", str(len(eng_result.original_features)))
    eng_table.add_row("Engineered Features", str(len(eng_result.engineered_features)))
    eng_table.add_row("Total Features", str(len(eng_result.all_features)))
    eng_table.add_row("Processing Time", f"{eng_result.processing_time_seconds:.2f}s")
    
    console.print(eng_table)
    
    # Feature selection summary
    sel_table = Table(title="Feature Selection Summary")
    sel_table.add_column("Metric", style="cyan")
    sel_table.add_column("Value", style="green")
    
    sel_result = results['selection']
    if sel_result:
        original_selected = [f for f in sel_result.selected_features 
                           if f.name in [of.name for of in eng_result.original_features]]
        engineered_selected = [f for f in sel_result.selected_features 
                             if f.name not in [of.name for of in eng_result.original_features]]
        
        sel_table.add_row("Features Selected", str(len(sel_result.selected_features)))
        sel_table.add_row("Original Selected", str(len(original_selected)))
        sel_table.add_row("Engineered Selected", str(len(engineered_selected)))
        sel_table.add_row("Reduction %", f"{sel_result.dimensionality_reduction['reduction_percentage']:.1f}%")
        
        console.print(sel_table)
        
        # Top engineered features
        if engineered_selected:
            console.print("\n[bold]Top Engineered Features Selected:[/bold]")
            eng_sorted = sorted(engineered_selected, key=lambda x: x.importance_score, reverse=True)
            for i, feat in enumerate(eng_sorted[:5], 1):
                console.print(f"  {i}. [yellow]{feat.name}[/yellow] (score: {feat.importance_score:.3f})")
    
    # Model performance improvement
    perf_table = Table(title="Model Performance Comparison")
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Original", style="red")
    perf_table.add_column("With Engineering", style="green")
    perf_table.add_column("Improvement", style="bold green")
    
    if 'r2_improvement' in improvement:
        r2_diff = improvement['r2_improvement']
        r2_sign = "+" if r2_diff >= 0 else ""
        mse_diff = improvement['mse_reduction']
        mse_sign = "-" if mse_diff >= 0 else "+"
        
        perf_table.add_row(
            "R² Score",
            f"{improvement['original_r2']:.4f}",
            f"{improvement['engineered_r2']:.4f}",
            f"{r2_sign}{r2_diff:.1f}%"
        )
        perf_table.add_row(
            "MSE",
            f"{improvement['original_mse']:.2f}",
            f"{improvement['engineered_mse']:.2f}",
            f"{mse_sign}{abs(mse_diff):.1f}%"
        )
    else:
        acc_diff = improvement['accuracy_improvement']
        sign = "+" if acc_diff >= 0 else ""
        perf_table.add_row(
            "Accuracy",
            f"{improvement['original_accuracy']:.4f}",
            f"{improvement['engineered_accuracy']:.4f}",
            f"{sign}{acc_diff:.1f}%"
        )
    
    console.print(perf_table)


def run_example(file_path, target_column, dataset_name, task_type='regression'):
    """Run DataAlchemy on a dataset and evaluate results"""
    
    console.print(f"\n[bold blue]Analyzing {dataset_name}...[/bold blue]")
    
    # Load original data
    if file_path.endswith('.parquet'):
        df_original = pd.read_parquet(file_path)
    else:
        df_original = pd.read_csv(file_path)
    
    # Run DataAlchemy
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    output_path = os.path.join(output_dir, f'{Path(file_path).stem}_features.parquet')
    results = run_data_alchemy(
        file_path=file_path,
        target_column=target_column,
        output_path=output_path,
        performance_mode='thorough'  # Use thorough mode to get best results
    )
    
    if results:
        # Load engineered features
        df_engineered = pd.read_parquet(output_path)
        
        # Ensure target column is not in engineered features (should already be excluded)
        if target_column in df_engineered.columns:
            console.print(f"[yellow]Warning: Target column '{target_column}' found in engineered features, removing it[/yellow]")
            df_engineered = df_engineered.drop(columns=[target_column])
        
        # Evaluate model performance
        improvement, model = evaluate_features(
            df_original, 
            df_engineered, 
            target_column,
            task_type
        )
        
        # Display results
        display_results(results, improvement, dataset_name)
        
        return results, improvement
    else:
        console.print(f"[red]Failed to analyze {dataset_name}[/red]")
        return None, None


def main():
    """Run comprehensive examples"""
    
    console.print(Panel.fit(
        "[bold]DataAlchemy Comprehensive Example[/bold]\n"
        "Demonstrating feature engineering on datasets designed to benefit from engineered features",
        style="cyan"
    ))
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    Path(output_dir).mkdir(exist_ok=True)
    
    # First, generate the sample data
    console.print("\n[yellow]Generating sample datasets...[/yellow]")
    # Run from the examples directory
    os.system(f"cd {os.path.dirname(__file__)} && python generate_sample_data.py")
    
    # Example 1: Simple Demo (shows clear improvement)
    console.print("\n" + "="*60)
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    results1, improvement1 = run_example(
        os.path.join(data_dir, 'simple_demo.parquet'),
        'target',
        'Simple Demo Dataset',
        'regression'
    )
    
    # Example 2: Customer Churn (complex patterns)
    console.print("\n" + "="*60)
    results2, improvement2 = run_example(
        os.path.join(data_dir, 'customer_churn.parquet'),
        'churn',
        'Customer Churn Dataset',
        'classification'
    )
    
    # Example 3: Sales Forecasting (time-based patterns)
    console.print("\n" + "="*60)
    results3, improvement3 = run_example(
        os.path.join(data_dir, 'sales_forecast.parquet'),
        'units_sold',
        'Sales Forecasting Dataset',
        'regression'
    )
    
    # Summary
    console.print("\n" + "="*60)
    console.print(Panel.fit(
        "[bold green]Analysis Complete![/bold green]\n\n"
        "DataAlchemy successfully engineered features for all datasets.\n"
        "The system automatically:\n"
        "• Detected non-linear relationships and created polynomial features\n"
        "• Identified important ratios and interactions\n"
        "• Extracted temporal patterns from date columns\n"
        "• Encoded categorical variables effectively\n"
        "• Selected the most impactful features while removing redundancy\n\n"
        "Note: Feature engineering may not always improve performance.\n"
        "Results depend on the dataset characteristics and existing feature quality.",
        style="green"
    ))
    
    # Clean up (optional)
    console.print("\n[dim]Output files saved in examples/output/[/dim]")
    console.print("[dim]To keep output files, comment out the cleanup section[/dim]")
    
    # Uncomment to clean up
    # import shutil
    # shutil.rmtree(output_dir, ignore_errors=True)
    # Note: Don't delete data directory as it contains other sample files


if __name__ == "__main__":
    main()