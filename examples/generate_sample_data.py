#!/usr/bin/env python3
"""
Generate sample datasets that benefit from feature engineering.
These datasets are designed to showcase the value of engineered features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)


def generate_customer_churn_data(n_samples=2000):
    """
    Generate a customer churn dataset where engineered features are crucial.
    Key patterns:
    - Churn is highly related to usage ratios and trends
    - Time-based patterns (seasonality, recency)
    - Interaction effects between features
    - Non-linear relationships
    """
    
    # Generate customer IDs
    customer_ids = [f"CUST_{i:05d}" for i in range(n_samples)]
    
    # Account age (days) - older accounts less likely to churn
    account_age = np.random.exponential(scale=365, size=n_samples).clip(30, 1825)
    
    # Generate time-based features
    signup_dates = pd.date_range(
        end=datetime.now() - timedelta(days=30),
        periods=n_samples,
        freq='h'
    ).to_numpy()
    np.random.shuffle(signup_dates)
    
    # Monthly charges - bimodal distribution
    basic_customers = np.random.binomial(1, 0.6, n_samples).astype(bool)
    monthly_charges = np.where(
        basic_customers,
        np.random.normal(35, 10, n_samples).clip(20, 60),
        np.random.normal(85, 20, n_samples).clip(60, 150)
    )
    
    # Total charges (correlated with account age but not perfectly)
    total_charges = monthly_charges * (account_age / 30) * np.random.uniform(0.8, 1.2, n_samples)
    
    # Usage patterns (GB per month) - key for churn
    avg_monthly_usage = np.where(
        basic_customers,
        np.random.gamma(2, 5, n_samples),
        np.random.gamma(4, 10, n_samples)
    )
    
    # Peak usage - some customers have spiky usage
    peak_usage = avg_monthly_usage * np.random.uniform(1.5, 5, n_samples)
    
    # Support tickets - more tickets indicate problems
    support_tickets_monthly = np.random.poisson(
        lam=np.where(basic_customers, 0.3, 0.8),
        size=n_samples
    )
    
    # Contract type
    contract_types = np.random.choice(
        ['Month-to-month', 'One year', 'Two year'],
        n_samples,
        p=[0.5, 0.3, 0.2]
    )
    
    # Payment method
    payment_methods = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
        n_samples,
        p=[0.35, 0.15, 0.25, 0.25]
    )
    
    # Last interaction (days ago) - recent interaction reduces churn
    last_interaction = np.random.exponential(scale=30, size=n_samples).clip(0, 365)
    
    # Service add-ons
    has_online_security = np.random.binomial(1, 0.3, n_samples)
    has_tech_support = np.random.binomial(1, 0.3, n_samples)
    has_streaming = np.random.binomial(1, 0.4, n_samples)
    
    # Calculate churn based on complex relationships
    churn_score = (
        # Usage efficiency matters (high charges, low usage = likely churn)
        0.3 * (monthly_charges / (avg_monthly_usage + 1)) / 10 +
        
        # Support ticket ratio
        0.25 * (support_tickets_monthly / (account_age / 365)) +
        
        # Contract type effect
        0.2 * (contract_types == 'Month-to-month').astype(float) +
        
        # Payment method risk
        0.15 * (payment_methods == 'Electronic check').astype(float) +
        
        # Recency effect (non-linear)
        0.1 * np.exp(-last_interaction / 60) +
        
        # Service stickiness (add-ons reduce churn)
        -0.1 * (has_online_security + has_tech_support + has_streaming) / 3 +
        
        # Account age benefit (logarithmic)
        -0.1 * np.log1p(account_age / 365) +
        
        # Random noise
        np.random.normal(0, 0.1, n_samples)
    )
    
    # Convert to binary churn
    churn_probability = 1 / (1 + np.exp(-4 * (churn_score - 0.5)))
    churn = np.random.binomial(1, churn_probability, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'signup_date': signup_dates,
        'account_age_days': account_age.astype(int),
        'monthly_charges': np.round(monthly_charges, 2),
        'total_charges': np.round(total_charges, 2),
        'avg_monthly_usage_gb': np.round(avg_monthly_usage, 1),
        'peak_monthly_usage_gb': np.round(peak_usage, 1),
        'support_tickets_monthly': support_tickets_monthly,
        'contract_type': contract_types,
        'payment_method': payment_methods,
        'last_interaction_days': last_interaction.astype(int),
        'has_online_security': has_online_security,
        'has_tech_support': has_tech_support,
        'has_streaming': has_streaming,
        'churn': churn
    })
    
    return df


def generate_sales_forecasting_data(n_samples=1500):
    """
    Generate sales data where time-based and interaction features are crucial.
    Key patterns:
    - Strong seasonal patterns
    - Product interactions
    - Price elasticity effects
    - Promotional impacts
    """
    
    # Generate dates
    dates = pd.date_range(start='2022-01-01', periods=n_samples, freq='D')
    
    # Store information
    n_stores = 20
    stores = np.random.randint(1, n_stores + 1, n_samples)
    store_types = np.random.choice(['A', 'B', 'C'], n_stores)
    store_type_map = {i+1: store_types[i] for i in range(n_stores)}
    
    # Product categories
    categories = np.random.choice(
        ['Electronics', 'Clothing', 'Food', 'Home', 'Sports'],
        n_samples,
        p=[0.25, 0.20, 0.30, 0.15, 0.10]
    )
    
    # Base demand varies by category
    category_base_demand = {
        'Electronics': 50,
        'Clothing': 80,
        'Food': 200,
        'Home': 40,
        'Sports': 30
    }
    
    # Price (with category-specific ranges)
    category_price_range = {
        'Electronics': (50, 500),
        'Clothing': (20, 150),
        'Food': (5, 50),
        'Home': (15, 200),
        'Sports': (25, 300)
    }
    
    prices = []
    for cat in categories:
        min_p, max_p = category_price_range[cat]
        prices.append(np.random.uniform(min_p, max_p))
    prices = np.array(prices)
    
    # Promotions (more frequent on weekends)
    is_weekend = pd.to_datetime(dates).dayofweek.isin([5, 6])
    promotion_prob = np.where(is_weekend, 0.3, 0.1)
    is_promotion = np.random.binomial(1, promotion_prob)
    discount_percentage = np.where(is_promotion, np.random.uniform(0.1, 0.4, n_samples), 0)
    
    # Competitor pricing (relative to our price)
    competitor_price_ratio = np.random.normal(1.0, 0.15, n_samples).clip(0.7, 1.5)
    
    # Weather (affects different categories differently)
    temperature = 20 + 15 * np.sin(2 * np.pi * pd.to_datetime(dates).dayofyear / 365) + \
                  np.random.normal(0, 5, n_samples)
    is_rainy = np.random.binomial(1, 0.2, n_samples)
    
    # Calculate sales with complex interactions
    sales = []
    for i in range(n_samples):
        date = pd.to_datetime(dates[i])
        cat = categories[i]
        
        # Base demand
        base = category_base_demand[cat]
        
        # Seasonal effect (different for each category)
        if cat == 'Clothing':
            seasonal = 1 + 0.5 * np.sin(2 * np.pi * (date.dayofyear - 80) / 365)
        elif cat == 'Food':
            seasonal = 1 + 0.2 * np.sin(2 * np.pi * date.dayofyear / 365)
        else:
            seasonal = 1
        
        # Day of week effect
        dow_effect = 1.2 if date.dayofweek in [5, 6] else 1.0
        
        # Price elasticity (category-specific)
        elasticity = {
            'Electronics': -1.5,
            'Clothing': -1.2,
            'Food': -0.8,
            'Home': -1.0,
            'Sports': -1.3
        }[cat]
        
        price_effect = (prices[i] / np.mean(category_price_range[cat])) ** elasticity
        
        # Promotion effect (non-linear)
        promo_effect = 1 + 2 * discount_percentage[i] ** 0.7 if is_promotion[i] else 1
        
        # Competition effect
        competition_effect = competitor_price_ratio[i] ** 0.5
        
        # Weather effect
        if cat == 'Food' and temperature[i] > 25:
            weather_effect = 1.3
        elif cat == 'Sports' and not is_rainy[i] and temperature[i] > 20:
            weather_effect = 1.4
        else:
            weather_effect = 1.0
        
        # Store type effect
        store_effect = {'A': 1.2, 'B': 1.0, 'C': 0.8}[store_type_map[stores[i]]]
        
        # Calculate final sales
        sales_value = (
            base * seasonal * dow_effect * price_effect * 
            promo_effect * competition_effect * weather_effect * store_effect *
            np.random.lognormal(0, 0.2)
        )
        
        sales.append(max(0, int(sales_value)))
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'store_id': stores,
        'store_type': [store_type_map[s] for s in stores],
        'product_category': categories,
        'price': np.round(prices, 2),
        'is_promotion': is_promotion,
        'discount_percentage': np.round(discount_percentage, 2),
        'competitor_price_ratio': np.round(competitor_price_ratio, 2),
        'temperature': np.round(temperature, 1),
        'is_rainy': is_rainy,
        'units_sold': sales
    })
    
    return df


def generate_simple_demo_data(n_samples=500):
    """Generate a simple dataset for quick demos"""
    
    # Features that have clear engineered feature opportunities
    x1 = np.random.normal(50, 15, n_samples)
    x2 = np.random.exponential(scale=10, size=n_samples)
    x3 = np.random.uniform(0, 100, n_samples)
    
    # Categorical with clear groups
    category = np.random.choice(['A', 'B', 'C', 'D'], n_samples, p=[0.4, 0.3, 0.2, 0.1])
    
    # Time feature
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    
    # Target with non-linear relationships
    # Includes: quadratic term, ratio, interaction, and categorical encoding effect
    category_effect = {'A': 0, 'B': 5, 'C': 10, 'D': -5}
    cat_values = [category_effect[c] for c in category]
    
    target = (
        0.5 * x1 +                    # Linear
        0.01 * x1**2 +                # Quadratic  
        10 * np.log1p(x2) +           # Log transform
        -0.2 * x1 * x3 / 100 +        # Interaction
        5 * (x2 / (x3 + 1)) +         # Ratio
        np.array(cat_values) +        # Categorical effect
        10 * np.sin(2 * np.pi * np.arange(n_samples) / 50) +  # Seasonal
        np.random.normal(0, 5, n_samples)  # Noise
    )
    
    df = pd.DataFrame({
        'feature_1': np.round(x1, 2),
        'feature_2': np.round(x2, 2), 
        'feature_3': np.round(x3, 2),
        'category': category,
        'date': dates,
        'target': np.round(target, 2)
    })
    
    return df


if __name__ == "__main__":
    print("Generating customer churn dataset...")
    churn_df = generate_customer_churn_data(2000)
    churn_df.to_parquet(os.path.join(data_dir, 'customer_churn.parquet'), index=False)
    print(f"✓ Saved customer churn data: {churn_df.shape}")
    
    print("\nGenerating sales forecasting dataset...")
    sales_df = generate_sales_forecasting_data(1500)
    sales_df.to_parquet(os.path.join(data_dir, 'sales_forecast.parquet'), index=False)
    print(f"✓ Saved sales forecast data: {sales_df.shape}")
    
    print("\nGenerating simple demo dataset...")
    demo_df = generate_simple_demo_data(500)
    demo_df.to_parquet(os.path.join(data_dir, 'simple_demo.parquet'), index=False)
    print(f"✓ Saved simple demo data: {demo_df.shape}")
    
    print("\nDataset generation complete!")
    print("\nThese datasets are designed to benefit from engineered features:")
    print("- customer_churn.parquet: Benefits from ratios, time-based features, and interactions")
    print("- sales_forecast.parquet: Benefits from seasonal decomposition and category interactions")
    print("- simple_demo.parquet: Clear non-linear patterns that benefit from polynomial and interaction features")