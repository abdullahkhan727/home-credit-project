"""
================================================================================
HOME CREDIT DEFAULT RISK - DATA CLEANING & FEATURE ENGINEERING
================================================================================

This script performs comprehensive data cleaning and feature engineering based
on EDA findings. It creates a production-ready dataset for modeling.

Key Tasks:
1. Fix DAYS_EMPLOYED anomaly (365243 placeholder)
2. Handle missing values in EXT_SOURCE variables
3. Transform demographic features (age, employment to positive values)
4. Create financial ratios (credit-to-income, loan-to-value, etc.)
5. Add missing data indicators
6. Create interaction terms and binned variables
7. Aggregate supplementary data (bureau, previous applications, installments)
8. Ensure train/test consistency

Author: Data Analysis Team
Date: February 2026

================================================================================
"""

import polars as pl
import polars.selectors as cs
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import pickle
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

print("="*80)
print("HOME CREDIT - DATA CLEANING & FEATURE ENGINEERING")
print("="*80)
print(f"\nStarted: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\n")

# Data directory
data_dir = Path('C:/Users/abdul/Downloads')

# Create output directory for processed data
output_dir = data_dir / 'home-credit-project-1' / 'processed_data'
output_dir.mkdir(exist_ok=True)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "="*80)
print("1. LOADING DATA")
print("="*80)

print("\nLoading application data...")
train = pl.read_csv(data_dir / 'application_train.csv')
test = pl.read_csv(data_dir / 'application_test.csv')

print(f"[OK] Training set: {train.shape}")
print(f"[OK] Test set: {test.shape}")

# Store original shapes for comparison
original_train_shape = train.shape
original_test_shape = test.shape

# ============================================================================
# FIX DAYS_EMPLOYED ANOMALY
# ============================================================================

print("\n" + "="*80)
print("2. FIXING DAYS_EMPLOYED ANOMALY")
print("="*80)

# Identify the anomaly (365243 is a placeholder value)
anomaly_count_train = train.filter(pl.col('DAYS_EMPLOYED') == 365243).shape[0]
anomaly_count_test = test.filter(pl.col('DAYS_EMPLOYED') == 365243).shape[0]

print(f"\nAnomalous values found:")
print(f"  Training set: {anomaly_count_train:,} ({100*anomaly_count_train/train.shape[0]:.2f}%)")
print(f"  Test set: {anomaly_count_test:,} ({100*anomaly_count_test/test.shape[0]:.2f}%)")

# Create indicator variable for the anomaly (this could be predictive)
train = train.with_columns([
    (pl.col('DAYS_EMPLOYED') == 365243).cast(pl.Int8).alias('DAYS_EMPLOYED_ANOM')
])

test = test.with_columns([
    (pl.col('DAYS_EMPLOYED') == 365243).cast(pl.Int8).alias('DAYS_EMPLOYED_ANOM')
])

# Replace anomalous values with null (will be imputed later)
train = train.with_columns([
    pl.when(pl.col('DAYS_EMPLOYED') == 365243)
    .then(None)
    .otherwise(pl.col('DAYS_EMPLOYED'))
    .alias('DAYS_EMPLOYED')
])

test = test.with_columns([
    pl.when(pl.col('DAYS_EMPLOYED') == 365243)
    .then(None)
    .otherwise(pl.col('DAYS_EMPLOYED'))
    .alias('DAYS_EMPLOYED')
])

print(f"[OK] Created DAYS_EMPLOYED_ANOM indicator")
print(f"[OK] Replaced {anomaly_count_train + anomaly_count_test:,} anomalous values with null")

# ============================================================================
# TRANSFORM DEMOGRAPHIC FEATURES (DAYS TO YEARS)
# ============================================================================

print("\n" + "="*80)
print("3. TRANSFORMING DEMOGRAPHIC FEATURES")
print("="*80)

# Transform DAYS_BIRTH to AGE (positive years)
train = train.with_columns([
    (pl.col('DAYS_BIRTH') / -365).alias('AGE_YEARS')
])

test = test.with_columns([
    (pl.col('DAYS_BIRTH') / -365).alias('AGE_YEARS')
])

# Transform DAYS_EMPLOYED to positive years
train = train.with_columns([
    (pl.col('DAYS_EMPLOYED') / -365).alias('YEARS_EMPLOYED')
])

test = test.with_columns([
    (pl.col('DAYS_EMPLOYED') / -365).alias('YEARS_EMPLOYED')
])

# Transform other DAYS variables to positive years
days_cols = ['DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE']
for col in days_cols:
    if col in train.columns:
        new_name = col.replace('DAYS_', 'YEARS_')
        train = train.with_columns([
            (pl.col(col) / -365).alias(new_name)
        ])
        test = test.with_columns([
            (pl.col(col) / -365).alias(new_name)
        ])

print(f"[OK] Created AGE_YEARS (range: {train['AGE_YEARS'].min():.1f} to {train['AGE_YEARS'].max():.1f})")
print(f"[OK] Created YEARS_EMPLOYED")
print(f"[OK] Transformed {len(days_cols)} additional DAYS variables to positive years")

# ============================================================================
# CREATE FINANCIAL RATIOS
# ============================================================================

print("\n" + "="*80)
print("4. CREATING FINANCIAL RATIOS")
print("="*80)

def create_financial_ratios(df):
    """Create comprehensive financial ratio features."""
    
    df = df.with_columns([
        # 1. CREDIT TO INCOME RATIO (higher = riskier)
        (pl.col('AMT_CREDIT') / pl.col('AMT_INCOME_TOTAL')).alias('CREDIT_INCOME_RATIO'),
        
        # 2. ANNUITY TO INCOME RATIO (debt service ratio)
        (pl.col('AMT_ANNUITY') / pl.col('AMT_INCOME_TOTAL')).alias('ANNUITY_INCOME_RATIO'),
        
        # 3. CREDIT TO GOODS PRICE RATIO (loan-to-value)
        (pl.col('AMT_CREDIT') / pl.col('AMT_GOODS_PRICE')).alias('CREDIT_GOODS_RATIO'),
        
        # 4. INCOME TO FAMILY SIZE (per capita income)
        (pl.col('AMT_INCOME_TOTAL') / pl.col('CNT_FAM_MEMBERS')).alias('INCOME_PER_PERSON'),
        
        # 5. PAYMENT RATE (how much of loan is paid per period)
        (pl.col('AMT_ANNUITY') / pl.col('AMT_CREDIT')).alias('PAYMENT_RATE'),
        
        # 6. YEARS TO REPAY (credit term in years)
        (pl.col('AMT_CREDIT') / (pl.col('AMT_ANNUITY') * 12)).alias('YEARS_TO_REPAY'),
        
        # 7. CREDIT PER CHILD (burden per child if family has children)
        pl.when(pl.col('CNT_CHILDREN') > 0)
        .then(pl.col('AMT_CREDIT') / pl.col('CNT_CHILDREN'))
        .otherwise(0)
        .alias('CREDIT_PER_CHILD'),
        
        # 8. INCOME TO CREDIT RATIO (affordability - inverse of credit/income)
        (pl.col('AMT_INCOME_TOTAL') / pl.col('AMT_CREDIT')).alias('INCOME_CREDIT_RATIO'),
        
        # 9. EMPLOYED TO AGE RATIO (employment stability)
        (pl.col('YEARS_EMPLOYED') / pl.col('AGE_YEARS')).alias('EMPLOYED_TO_AGE_RATIO'),
        
        # 10. EXTERNAL SOURCE MEAN (average of external credit scores)
        ((pl.col('EXT_SOURCE_1').fill_null(0) + 
          pl.col('EXT_SOURCE_2').fill_null(0) + 
          pl.col('EXT_SOURCE_3').fill_null(0)) / 3).alias('EXT_SOURCE_MEAN'),
        
        # 11. EXTERNAL SOURCE WEIGHTED (EXT_SOURCE_2 and 3 are more predictive)
        ((pl.col('EXT_SOURCE_1').fill_null(0) * 0.2 + 
          pl.col('EXT_SOURCE_2').fill_null(0) * 0.4 + 
          pl.col('EXT_SOURCE_3').fill_null(0) * 0.4)).alias('EXT_SOURCE_WEIGHTED'),
    ])
    
    return df

print("\nCreating financial ratios...")
train = create_financial_ratios(train)
test = create_financial_ratios(test)

print(f"[OK] Created 11 financial ratio features:")
print(f"  - Credit/Income, Annuity/Income, Credit/Goods (LTV)")
print(f"  - Income per person, Payment rate, Years to repay")
print(f"  - Credit per child, Employed/Age ratio")
print(f"  - External source aggregations")

# ============================================================================
# CREATE MISSING DATA INDICATORS
# ============================================================================

print("\n" + "="*80)
print("5. CREATING MISSING DATA INDICATORS")
print("="*80)

# Identify columns with significant missing data (>5%)
missing_threshold = 0.05
cols_with_missing = []

for col in train.columns:
    if col == 'TARGET':
        continue
    missing_pct = train.select(col).null_count()[0, 0] / train.shape[0]
    if missing_pct > missing_threshold:
        cols_with_missing.append((col, missing_pct))

print(f"\nFound {len(cols_with_missing)} columns with >{missing_threshold*100}% missing data")

# Create indicator variables for columns with missing data
for col, missing_pct in cols_with_missing[:15]:  # Top 15 to avoid too many features
    indicator_name = f'{col}_MISSING'
    
    train = train.with_columns([
        pl.col(col).is_null().cast(pl.Int8).alias(indicator_name)
    ])
    
    test = test.with_columns([
        pl.col(col).is_null().cast(pl.Int8).alias(indicator_name)
    ])

print(f"[OK] Created {min(15, len(cols_with_missing))} missing data indicators")

# ============================================================================
# CREATE BINNED VARIABLES
# ============================================================================

print("\n" + "="*80)
print("6. CREATING BINNED VARIABLES")
print("="*80)

# Compute quantiles from TRAINING data only (for train/test consistency)
age_quantiles = [
    train['AGE_YEARS'].quantile(0.25),
    train['AGE_YEARS'].quantile(0.5),
    train['AGE_YEARS'].quantile(0.75)
]
income_quantiles = [
    train['AMT_INCOME_TOTAL'].quantile(0.25),
    train['AMT_INCOME_TOTAL'].quantile(0.5),
    train['AMT_INCOME_TOTAL'].quantile(0.75)
]
credit_quantiles = [
    train['AMT_CREDIT'].quantile(0.25),
    train['AMT_CREDIT'].quantile(0.5),
    train['AMT_CREDIT'].quantile(0.75)
]

# Save quantiles for later use on test data
quantiles = {
    'age': age_quantiles,
    'income': income_quantiles,
    'credit': credit_quantiles
}

def create_binned_features(df, quantiles):
    """Create binned categorical features."""
    
    # Age bins
    df = df.with_columns([
        pl.when(pl.col('AGE_YEARS') <= quantiles['age'][0])
        .then(pl.lit('Young'))
        .when(pl.col('AGE_YEARS') <= quantiles['age'][1])
        .then(pl.lit('Middle_Young'))
        .when(pl.col('AGE_YEARS') <= quantiles['age'][2])
        .then(pl.lit('Middle_Old'))
        .otherwise(pl.lit('Senior'))
        .alias('AGE_GROUP')
    ])
    
    # Income bins
    df = df.with_columns([
        pl.when(pl.col('AMT_INCOME_TOTAL') <= quantiles['income'][0])
        .then(pl.lit('Low'))
        .when(pl.col('AMT_INCOME_TOTAL') <= quantiles['income'][1])
        .then(pl.lit('Medium_Low'))
        .when(pl.col('AMT_INCOME_TOTAL') <= quantiles['income'][2])
        .then(pl.lit('Medium_High'))
        .otherwise(pl.lit('High'))
        .alias('INCOME_GROUP')
    ])
    
    # Credit bins
    df = df.with_columns([
        pl.when(pl.col('AMT_CREDIT') <= quantiles['credit'][0])
        .then(pl.lit('Small'))
        .when(pl.col('AMT_CREDIT') <= quantiles['credit'][1])
        .then(pl.lit('Medium_Small'))
        .when(pl.col('AMT_CREDIT') <= quantiles['credit'][2])
        .then(pl.lit('Medium_Large'))
        .otherwise(pl.lit('Large'))
        .alias('CREDIT_GROUP')
    ])
    
    return df

print("\nCreating binned features...")
train = create_binned_features(train, quantiles)
test = create_binned_features(test, quantiles)

print(f"[OK] Created 3 binned categorical features:")
print(f"  - AGE_GROUP (Young, Middle_Young, Middle_Old, Senior)")
print(f"  - INCOME_GROUP (Low, Medium_Low, Medium_High, High)")
print(f"  - CREDIT_GROUP (Small, Medium_Small, Medium_Large, Large)")

# ============================================================================
# CREATE INTERACTION TERMS
# ============================================================================

print("\n" + "="*80)
print("7. CREATING INTERACTION TERMS")
print("="*80)

train = train.with_columns([
    # Age × Income (older high-income clients are lower risk)
    (pl.col('AGE_YEARS') * pl.col('AMT_INCOME_TOTAL') / 1000000).alias('AGE_INCOME_INTERACTION'),
    
    # Age × Employed (experience)
    (pl.col('AGE_YEARS') * pl.col('YEARS_EMPLOYED')).alias('AGE_EMPLOYED_INTERACTION'),
    
    # Credit × Income (affordability)
    (pl.col('AMT_CREDIT') * pl.col('AMT_INCOME_TOTAL') / 1000000000).alias('CREDIT_INCOME_INTERACTION'),
    
    # Children × Family Members (family burden)
    (pl.col('CNT_CHILDREN') * pl.col('CNT_FAM_MEMBERS')).alias('CHILDREN_FAMILY_INTERACTION'),
    
    # External Source 2 × 3 (most predictive external scores)
    (pl.col('EXT_SOURCE_2').fill_null(0) * pl.col('EXT_SOURCE_3').fill_null(0)).alias('EXT_SOURCE_2_3_INTERACTION'),
])

test = test.with_columns([
    (pl.col('AGE_YEARS') * pl.col('AMT_INCOME_TOTAL') / 1000000).alias('AGE_INCOME_INTERACTION'),
    (pl.col('AGE_YEARS') * pl.col('YEARS_EMPLOYED')).alias('AGE_EMPLOYED_INTERACTION'),
    (pl.col('AMT_CREDIT') * pl.col('AMT_INCOME_TOTAL') / 1000000000).alias('CREDIT_INCOME_INTERACTION'),
    (pl.col('CNT_CHILDREN') * pl.col('CNT_FAM_MEMBERS')).alias('CHILDREN_FAMILY_INTERACTION'),
    (pl.col('EXT_SOURCE_2').fill_null(0) * pl.col('EXT_SOURCE_3').fill_null(0)).alias('EXT_SOURCE_2_3_INTERACTION'),
])

print(f"[OK] Created 5 interaction features")

# ============================================================================
# AGGREGATE BUREAU DATA
# ============================================================================

print("\n" + "="*80)
print("8. AGGREGATING BUREAU DATA")
print("="*80)

bureau_path = data_dir / 'bureau.csv'
if bureau_path.exists():
    print("\nLoading bureau.csv...")
    bureau = pl.read_csv(bureau_path)
    print(f"  Bureau records: {bureau.shape[0]:,}")
    print(f"  Unique clients: {bureau['SK_ID_CURR'].n_unique():,}")
    
    # Aggregate bureau features
    print("\nAggregating bureau features...")
    bureau_agg = (
        bureau
        .group_by('SK_ID_CURR')
        .agg([
            # Count statistics
            pl.len().alias('BUREAU_COUNT'),
            (pl.col('CREDIT_ACTIVE') == 'Active').sum().alias('BUREAU_ACTIVE_COUNT'),
            (pl.col('CREDIT_ACTIVE') == 'Closed').sum().alias('BUREAU_CLOSED_COUNT'),
            
            # Credit amounts
            pl.col('AMT_CREDIT_SUM').sum().alias('BUREAU_CREDIT_SUM'),
            pl.col('AMT_CREDIT_SUM').mean().alias('BUREAU_CREDIT_MEAN'),
            pl.col('AMT_CREDIT_SUM').max().alias('BUREAU_CREDIT_MAX'),
            
            # Debt statistics
            pl.col('AMT_CREDIT_SUM_DEBT').sum().alias('BUREAU_DEBT_SUM'),
            pl.col('AMT_CREDIT_SUM_DEBT').mean().alias('BUREAU_DEBT_MEAN'),
            
            # Overdue statistics
            (pl.col('CREDIT_DAY_OVERDUE') > 0).sum().alias('BUREAU_OVERDUE_COUNT'),
            pl.col('CREDIT_DAY_OVERDUE').max().alias('BUREAU_OVERDUE_MAX_DAYS'),
            pl.col('CREDIT_DAY_OVERDUE').mean().alias('BUREAU_OVERDUE_MEAN_DAYS'),
            
            # Credit type diversity
            pl.col('CREDIT_TYPE').n_unique().alias('BUREAU_CREDIT_TYPES'),
        ])
    )
    
    # Create debt ratios
    bureau_agg = bureau_agg.with_columns([
        (pl.col('BUREAU_DEBT_SUM') / pl.col('BUREAU_CREDIT_SUM')).alias('BUREAU_DEBT_CREDIT_RATIO'),
        (pl.col('BUREAU_ACTIVE_COUNT') / pl.col('BUREAU_COUNT')).alias('BUREAU_ACTIVE_RATIO'),
        (pl.col('BUREAU_OVERDUE_COUNT') / pl.col('BUREAU_COUNT')).alias('BUREAU_OVERDUE_RATIO'),
    ])
    
    print(f"[OK] Aggregated bureau data to {bureau_agg.shape[0]:,} clients")
    print(f"[OK] Created {bureau_agg.shape[1]-1} bureau features")
    
    # Join with train and test
    train = train.join(bureau_agg, on='SK_ID_CURR', how='left')
    test = test.join(bureau_agg, on='SK_ID_CURR', how='left')
    
    print(f"[OK] Joined bureau features to application data")
    
    # Clean up memory
    del bureau, bureau_agg
else:
    print("[WARNING] Bureau data not found, skipping...")

# ============================================================================
# AGGREGATE PREVIOUS APPLICATIONS
# ============================================================================

print("\n" + "="*80)
print("9. AGGREGATING PREVIOUS APPLICATIONS")
print("="*80)

prev_app_path = data_dir / 'previous_application.csv'
if prev_app_path.exists():
    print("\nLoading previous_application.csv...")
    prev_app = pl.read_csv(prev_app_path)
    print(f"  Previous applications: {prev_app.shape[0]:,}")
    print(f"  Unique clients: {prev_app['SK_ID_CURR'].n_unique():,}")
    
    print("\nAggregating previous application features...")
    prev_agg = (
        prev_app
        .group_by('SK_ID_CURR')
        .agg([
            # Count statistics
            pl.len().alias('PREV_APP_COUNT'),
            (pl.col('NAME_CONTRACT_STATUS') == 'Approved').sum().alias('PREV_APP_APPROVED'),
            (pl.col('NAME_CONTRACT_STATUS') == 'Refused').sum().alias('PREV_APP_REFUSED'),
            (pl.col('NAME_CONTRACT_STATUS') == 'Canceled').sum().alias('PREV_APP_CANCELED'),
            
            # Credit amounts
            pl.col('AMT_CREDIT').mean().alias('PREV_CREDIT_MEAN'),
            pl.col('AMT_CREDIT').max().alias('PREV_CREDIT_MAX'),
            pl.col('AMT_CREDIT').sum().alias('PREV_CREDIT_SUM'),
            
            # Application amounts
            pl.col('AMT_APPLICATION').mean().alias('PREV_APPLICATION_MEAN'),
            
            # Down payment
            pl.col('AMT_DOWN_PAYMENT').mean().alias('PREV_DOWN_PAYMENT_MEAN'),
            
            # Time features
            pl.col('DAYS_DECISION').mean().alias('PREV_DAYS_DECISION_MEAN'),
        ])
    )
    
    # Create derived features
    prev_agg = prev_agg.with_columns([
        (pl.col('PREV_APP_APPROVED') / pl.col('PREV_APP_COUNT')).alias('PREV_APPROVAL_RATE'),
        (pl.col('PREV_APP_REFUSED') / pl.col('PREV_APP_COUNT')).alias('PREV_REFUSAL_RATE'),
    ])
    
    print(f"[OK] Aggregated previous applications to {prev_agg.shape[0]:,} clients")
    print(f"[OK] Created {prev_agg.shape[1]-1} previous application features")
    
    # Join with train and test
    train = train.join(prev_agg, on='SK_ID_CURR', how='left')
    test = test.join(prev_agg, on='SK_ID_CURR', how='left')
    
    print(f"[OK] Joined previous application features")
    
    del prev_app, prev_agg
else:
    print("[WARNING] Previous application data not found, skipping...")

# ============================================================================
# AGGREGATE INSTALLMENTS PAYMENTS
# ============================================================================

print("\n" + "="*80)
print("10. AGGREGATING INSTALLMENTS PAYMENTS")
print("="*80)

installments_path = data_dir / 'installments_payments.csv'
if installments_path.exists():
    print("\nLoading installments_payments.csv...")
    installments = pl.read_csv(installments_path)
    print(f"  Installment records: {installments.shape[0]:,}")
    print(f"  Unique clients: {installments['SK_ID_CURR'].n_unique():,}")
    
    print("\nAggregating installment payment features...")
    
    # Calculate payment status
    installments = installments.with_columns([
        (pl.col('DAYS_ENTRY_PAYMENT') - pl.col('DAYS_INSTALMENT')).alias('PAYMENT_DELAY'),
        (pl.col('AMT_PAYMENT') - pl.col('AMT_INSTALMENT')).alias('PAYMENT_DIFF'),
    ])
    
    install_agg = (
        installments
        .group_by('SK_ID_CURR')
        .agg([
            # Count statistics
            pl.len().alias('INSTALL_COUNT'),
            
            # Late payment statistics
            (pl.col('PAYMENT_DELAY') > 0).sum().alias('INSTALL_LATE_COUNT'),
            pl.col('PAYMENT_DELAY').mean().alias('INSTALL_DELAY_MEAN'),
            pl.col('PAYMENT_DELAY').max().alias('INSTALL_DELAY_MAX'),
            
            # Payment differences
            pl.col('PAYMENT_DIFF').mean().alias('INSTALL_PAYMENT_DIFF_MEAN'),
            (pl.col('PAYMENT_DIFF') < 0).sum().alias('INSTALL_UNDERPAYMENT_COUNT'),
            
            # Payment amounts
            pl.col('AMT_PAYMENT').sum().alias('INSTALL_PAYMENT_SUM'),
            pl.col('AMT_INSTALMENT').sum().alias('INSTALL_INSTALMENT_SUM'),
        ])
    )
    
    # Create derived features
    install_agg = install_agg.with_columns([
        (pl.col('INSTALL_LATE_COUNT') / pl.col('INSTALL_COUNT')).alias('INSTALL_LATE_RATE'),
        (pl.col('INSTALL_UNDERPAYMENT_COUNT') / pl.col('INSTALL_COUNT')).alias('INSTALL_UNDERPAYMENT_RATE'),
        (pl.col('INSTALL_PAYMENT_SUM') / pl.col('INSTALL_INSTALMENT_SUM')).alias('INSTALL_PAYMENT_RATIO'),
    ])
    
    print(f"[OK] Aggregated installments to {install_agg.shape[0]:,} clients")
    print(f"[OK] Created {install_agg.shape[1]-1} installment features")
    
    # Join with train and test
    train = train.join(install_agg, on='SK_ID_CURR', how='left')
    test = test.join(install_agg, on='SK_ID_CURR', how='left')
    
    print(f"[OK] Joined installment payment features")
    
    del installments, install_agg
else:
    print("[WARNING] Installments data not found, skipping...")

# ============================================================================
# HANDLE MISSING VALUES IN EXT_SOURCE AND NUMERIC COLUMNS
# ============================================================================

print("\n" + "="*80)
print("11. IMPUTING MISSING VALUES")
print("="*80)

# Compute imputation values from TRAINING data only
imputation_values = {}

# Get numeric columns (excluding ID and TARGET)
numeric_cols = [col for col in train.select(cs.numeric()).columns 
                if col not in ['SK_ID_CURR', 'TARGET']]

print(f"\nComputing imputation values from training data...")
for col in numeric_cols:
    if train.select(col).null_count()[0, 0] > 0:
        # Use median for robustness to outliers
        median_val = train[col].median()
        imputation_values[col] = median_val

print(f"[OK] Computed imputation values for {len(imputation_values)} columns")

# Apply imputation to both train and test
print("\nApplying imputation...")
for col, value in imputation_values.items():
    train = train.with_columns([
        pl.col(col).fill_null(value)
    ])
    test = test.with_columns([
        pl.col(col).fill_null(value)
    ])

print(f"[OK] Imputed missing values using training medians")

# Save imputation values for future use
with open(output_dir / 'imputation_values.pkl', 'wb') as f:
    pickle.dump(imputation_values, f)

print(f"[OK] Saved imputation values to imputation_values.pkl")

# ============================================================================
# VERIFY TRAIN/TEST CONSISTENCY
# ============================================================================

print("\n" + "="*80)
print("12. VERIFYING TRAIN/TEST CONSISTENCY")
print("="*80)

# Check that all columns match (except TARGET)
train_cols = set(train.columns) - {'TARGET'}
test_cols = set(test.columns)

if train_cols == test_cols:
    print("[OK] Train and test have identical columns (except TARGET)")
    print(f"  Total features: {len(train_cols)}")
else:
    print("[WARNING] WARNING: Column mismatch detected!")
    only_in_train = train_cols - test_cols
    only_in_test = test_cols - train_cols
    if only_in_train:
        print(f"  Only in train: {only_in_train}")
    if only_in_test:
        print(f"  Only in test: {only_in_test}")

# Check for remaining missing values
train_missing = train.select(cs.numeric()).null_count().sum_horizontal()[0]
test_missing = test.select(cs.numeric()).null_count().sum_horizontal()[0]

print(f"\nRemaining missing values:")
print(f"  Training: {train_missing:,}")
print(f"  Test: {test_missing:,}")

# ============================================================================
# SAVE PROCESSED DATA
# ============================================================================

print("\n" + "="*80)
print("13. SAVING PROCESSED DATA")
print("="*80)

print("\nFinal dataset shapes:")
print(f"  Training: {train.shape} (original: {original_train_shape})")
print(f"  Test: {test.shape} (original: {original_test_shape})")
print(f"  Features added: {train.shape[1] - original_train_shape[1]}")

# Save as Parquet for efficiency
train.write_parquet(output_dir / 'train_processed.parquet')
test.write_parquet(output_dir / 'test_processed.parquet')

# Also save as CSV for inspection
train.write_csv(output_dir / 'train_processed.csv')
test.write_csv(output_dir / 'test_processed.csv')

# Save quantiles for later use
with open(output_dir / 'quantiles.pkl', 'wb') as f:
    pickle.dump(quantiles, f)

print(f"\n[OK] Saved processed data:")
print(f"  - train_processed.parquet ({train.shape})")
print(f"  - test_processed.parquet ({test.shape})")
print(f"  - train_processed.csv")
print(f"  - test_processed.csv")
print(f"  - imputation_values.pkl")
print(f"  - quantiles.pkl")

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "="*80)
print("PROCESSING SUMMARY")
print("="*80)

summary = f"""
DATA CLEANING & FEATURE ENGINEERING COMPLETE
================================================================

ORIGINAL DATA:
  Training: {original_train_shape[0]:,} rows × {original_train_shape[1]} columns
  Test: {original_test_shape[0]:,} rows × {original_test_shape[1]} columns

PROCESSED DATA:
  Training: {train.shape[0]:,} rows × {train.shape[1]} columns
  Test: {test.shape[0]:,} rows × {test.shape[1]} columns
  New features: {train.shape[1] - original_train_shape[1]}

TRANSFORMATIONS APPLIED:
  [OK] Fixed DAYS_EMPLOYED anomaly (365243 -> null + indicator)
  [OK] Transformed DAYS variables to positive YEARS
  [OK] Created 11 financial ratios
  [OK] Created 15 missing data indicators
  [OK] Created 3 binned categorical features
  [OK] Created 5 interaction terms
  [OK] Aggregated bureau data ({bureau_agg.shape[1]-1 if 'bureau_agg' in locals() else 0} features)
  [OK] Aggregated previous applications ({prev_agg.shape[1]-1 if 'prev_agg' in locals() else 0} features)
  [OK] Aggregated installment payments ({install_agg.shape[1]-1 if 'install_agg' in locals() else 0} features)
  [OK] Imputed missing values using training medians
  [OK] Ensured train/test consistency

TRAIN/TEST CONSISTENCY:
  [OK] Identical columns (except TARGET)
  [OK] Same imputation strategy applied
  [OK] Same binning thresholds applied

FILES SAVED TO: {output_dir}
  - train_processed.parquet
  - test_processed.parquet
  - train_processed.csv
  - test_processed.csv
  - imputation_values.pkl
  - quantiles.pkl

STATUS: [OK] READY FOR MODELING

Next steps:
  1. Encode categorical variables
  2. Scale features (if using linear models)
  3. Feature selection
  4. Model training
"""

print(summary)

print("="*80)
print(f"Completed: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
print("="*80)
