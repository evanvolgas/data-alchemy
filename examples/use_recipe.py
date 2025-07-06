#!/usr/bin/env python3
"""
Example of how to use the feature recipe to recreate engineered features
"""

import pandas as pd
import numpy as np
import json

def apply_feature_recipe(df, recipe_path):
    """Apply a feature recipe to create engineered features"""
    
    # Load the recipe
    with open(recipe_path, 'r') as f:
        recipe = json.load(f)
    
    print(f"\nLoaded recipe with {recipe['dataset_info']['total_features_selected']} features")
    print(f"- Original features: {recipe['dataset_info']['original_features_selected']}")
    print(f"- Engineered features: {recipe['dataset_info']['engineered_features_selected']}")
    
    # Create a copy to avoid modifying original
    df_features = df.copy()
    
    # Apply each engineered feature transformation
    print("\nApplying engineered features:")
    for feature in recipe['engineered_features']:
        print(f"  Creating {feature['name']} (importance: {feature['importance']:.3f})")
        
        # Execute the Python code to create the feature
        try:
            # Create local context with df pointing to df_features
            local_context = {'df': df_features, 'pd': pd, 'np': np}
            exec(feature['python_code'], local_context)
            print(f"    ✓ Created successfully")
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    # Select only the features mentioned in the recipe
    all_selected_features = []
    all_selected_features.extend([f['name'] for f in recipe['original_features']])
    all_selected_features.extend([f['name'] for f in recipe['engineered_features']])
    
    # Filter to only include columns that exist
    existing_features = [col for col in all_selected_features if col in df_features.columns]
    
    print(f"\nFinal feature set: {len(existing_features)} features")
    print(f"Available columns in df_features: {len(df_features.columns)}")
    
    # Debug: show what's missing
    missing = set(all_selected_features) - set(df_features.columns)
    if missing:
        print(f"Missing features: {missing}")
    
    # Return all columns for now to debug
    return df_features


if __name__ == "__main__":
    # Example usage
    print("DataAlchemy Feature Recipe Example")
    print("==================================")
    
    # Load the original data
    df = pd.read_parquet('data/simple_demo.parquet')
    print(f"\nOriginal data shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Apply the recipe
    df_with_features = apply_feature_recipe(df, 'output/simple_demo_features_recipe.json')
    
    print(f"\nTransformed data shape: {df_with_features.shape}")
    print(f"\nTop 10 features by name:")
    for col in sorted(df_with_features.columns)[:10]:
        print(f"  - {col}")
    
    # Show that the features match what DataAlchemy created
    print("\nComparing with DataAlchemy output:")
    df_alchemy = pd.read_parquet('output/simple_demo_features.parquet')
    
    # Remove target column for comparison
    if 'target' in df_alchemy.columns:
        df_alchemy = df_alchemy.drop(columns=['target'])
    if 'target' in df_with_features.columns:
        df_with_features = df_with_features.drop(columns=['target'])
    
    matching_cols = set(df_with_features.columns) & set(df_alchemy.columns)
    print(f"  Matching columns: {len(matching_cols)}/{len(df_alchemy.columns)}")
    
    # Check if values match for a few features
    print("\nValue comparison for sample features:")
    numeric_cols = df_with_features.select_dtypes(include=[np.number]).columns
    numeric_matching = list(set(numeric_cols) & matching_cols)[:3]
    
    for col in numeric_matching:
        if col in df_with_features.columns and col in df_alchemy.columns:
            try:
                matches = np.allclose(df_with_features[col].fillna(0), df_alchemy[col].fillna(0))
                print(f"  {col}: {'✓ Values match' if matches else '✗ Values differ'}")
            except Exception as e:
                print(f"  {col}: Could not compare - {e}")