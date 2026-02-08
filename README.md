# Home Credit Default Risk Analysis

**Predictive modeling project for loan default risk assessment**

---

## 📊 Project Overview

This project develops a machine learning model to predict loan default risk for Home Credit Group. Using historical loan application data and supplementary credit history, we build a classifier to identify clients likely to default, enabling better lending decisions and reduced financial losses.

### Business Problem
Home Credit aims to predict which clients are likely to default on their loans. Default is a rare event (~8% of loans), creating a highly imbalanced classification problem requiring specialized modeling approaches.

### Key Metrics
- **Primary:** ROC-AUC Score (accounts for class imbalance)
- **Secondary:** Precision, Recall, F1-Score
- **Business Impact:** Reduction in default losses while maintaining approval rates

---

## 📁 Project Structure

```
home-credit-project-1/
├── README.md                          # This file
├── data_preprocessing.py              # Main data preparation pipeline
├── EDA_Report_Final.html              # Comprehensive EDA report with source code
├── PREPROCESSING_REPORT.md            # Data preprocessing documentation
├── processed_data/                    # Processed datasets (not in repo)
│   ├── train_processed.parquet        # Training data (307,511 × 200)
│   ├── test_processed.parquet         # Test data (48,744 × 199)
│   ├── imputation_values.pkl          # Saved imputation values
│   └── quantiles.pkl                  # Saved binning thresholds
└── case_template.qmd                  # Project requirements
```

---

## 🎯 Key Findings from EDA

### Data Quality
- **Overall Quality Score:** 91.8/100 (Excellent)
- **Missing Data:** 2.26% overall (manageable)
- **Class Imbalance:** 91.96% repaid vs 8.04% defaulted (11.39:1 ratio)
- **Data Issues:** Fixed DAYS_EMPLOYED anomaly (365,243 placeholder value)

### Strongest Predictors
1. **Employment Stability** (DAYS_EMPLOYED: r = -0.065)
2. **Client Age** (older clients = lower risk)
3. **Income Level** (higher income = protective)
4. **Credit Affordability** (loan size relative to income)
5. **External Credit Scores** (EXT_SOURCE variables)

### Critical Insights
- Baseline accuracy of 91.96% is misleading due to class imbalance
- Must use ROC-AUC, Precision-Recall, and F1-Score for evaluation
- Weak individual correlations (|r| < 0.08) indicate need for multivariate models
- Supplementary data (bureau history, payment patterns) will significantly improve predictions

---

## 🔧 Data Preprocessing & Feature Engineering

### Data Quality Fixes
✅ **Fixed DAYS_EMPLOYED Anomaly**
- Identified 64,648 records (18%) with placeholder value 365,243
- Created indicator variable `DAYS_EMPLOYED_ANOM`
- Replaced anomalies with null for proper imputation

✅ **Handled Missing Values**
- Computed imputation values from training data only (110 columns)
- Applied median imputation for numeric features
- Created 15 missing data indicators for high-missing columns

✅ **Transformed Demographic Features**
- Converted negative DAYS to positive YEARS (more interpretable)
- AGE_YEARS, YEARS_EMPLOYED, YEARS_REGISTRATION, etc.

### Feature Engineering (78 new features created)

#### 1. Financial Ratios (11 features)
- `CREDIT_INCOME_RATIO` - Credit to income (affordability)
- `ANNUITY_INCOME_RATIO` - Debt service ratio
- `CREDIT_GOODS_RATIO` - Loan-to-value (LTV)
- `PAYMENT_RATE` - Payment burden
- `YEARS_TO_REPAY` - Loan term estimation
- `INCOME_PER_PERSON` - Per capita income
- `EMPLOYED_TO_AGE_RATIO` - Employment stability
- `EXT_SOURCE_MEAN` - Average external credit score
- `EXT_SOURCE_WEIGHTED` - Weighted external score

#### 2. Missing Data Indicators (15 features)
Binary flags for columns with >5% missing data

#### 3. Binned Categories (3 features)
- `AGE_GROUP` (Young, Middle_Young, Middle_Old, Senior)
- `INCOME_GROUP` (Low, Medium_Low, Medium_High, High)
- `CREDIT_GROUP` (Small, Medium_Small, Medium_Large, Large)

#### 4. Interaction Terms (5 features)
- Age × Income, Age × Employed
- Credit × Income, Children × Family
- External Source 2 × 3 interaction

#### 5. Bureau Data Aggregates (16 features)
From 1.7M bureau records → per-client features:
- Credit counts, amounts, and debt ratios
- Overdue statistics and delinquency patterns
- Active vs closed credit ratios

#### 6. Previous Applications (13 features)
From 1.7M previous applications → per-client features:
- Approval rates and refusal history
- Average credit amounts
- Application patterns

#### 7. Installment Payments (12 features)
From 13.6M payment records → per-client features:
- Late payment rates
- Payment delay patterns
- Underpayment indicators

### Train/Test Consistency
✅ Identical columns (except TARGET in test)  
✅ Same imputation strategy applied  
✅ Same binning thresholds used  
✅ Reproducible transformations saved

---

## 📊 Dataset Statistics

| Metric | Original | Processed | Change |
|--------|----------|-----------|--------|
| **Training Rows** | 307,511 | 307,511 | No loss |
| **Training Columns** | 122 | 200 | +78 features |
| **Test Rows** | 48,744 | 48,744 | No loss |
| **Test Columns** | 121 | 199 | +78 features |
| **Missing Values** | 2.26% | 0% numeric | Imputed |

---

## 🚀 Running the Data Preparation Pipeline

### Prerequisites
```bash
pip install polars pandas numpy scikit-learn
```

### Execute Pipeline
```bash
python data_preprocessing.py
```

### Pipeline Steps
1. Load application_train.csv and application_test.csv
2. Fix DAYS_EMPLOYED anomaly
3. Transform demographic features (DAYS → YEARS)
4. Create 11 financial ratios
5. Create 15 missing data indicators
6. Create 3 binned categorical features
7. Create 5 interaction terms
8. Aggregate bureau.csv (1.7M records)
9. Aggregate previous_application.csv (1.7M records)
10. Aggregate installments_payments.csv (13.6M records)
11. Impute missing values using training medians
12. Verify train/test consistency
13. Save processed data and configuration files

### Output Files
- `processed_data/train_processed.parquet` (307,511 × 200)
- `processed_data/test_processed.parquet` (48,744 × 199)
- `processed_data/imputation_values.pkl`
- `processed_data/quantiles.pkl`

---

## 📈 Expected Model Performance

Based on comprehensive feature engineering:

| Model Type | Expected ROC-AUC | Notes |
|-----------|------------------|-------|
| Baseline (majority class) | 0.50 | Random guessing |
| Simple Logistic Regression | 0.60-0.65 | Basic model |
| XGBoost (application data) | 0.70-0.75 | Good model |
| **XGBoost + engineered features** | **0.76-0.82** | **Strong model** |
| Optimized ensemble | 0.83+ | Excellent model |

---

## 🎯 Next Steps

### 1. Encode Categorical Variables
```python
categorical_cols = ['AGE_GROUP', 'INCOME_GROUP', 'CREDIT_GROUP', 
                   'NAME_CONTRACT_TYPE', 'CODE_GENDER', etc.]
# Use one-hot encoding or label encoding as appropriate
```

### 2. Model Training (Recommended Approach)
```python
import xgboost as xgb

# XGBoost handles class imbalance well
model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.01,
    n_estimators=1000,
    scale_pos_weight=11.39,  # Handle 11.39:1 imbalance
    eval_metric='auc',
    random_state=42
)

model.fit(X_train, y_train,
         eval_set=[(X_val, y_val)],
         early_stopping_rounds=50)
```

### 3. Handle Class Imbalance
- Use `scale_pos_weight=11.39` in XGBoost
- Or use `class_weight='balanced'` in sklearn models
- Consider SMOTE oversampling if needed

### 4. Evaluation
```python
from sklearn.metrics import roc_auc_score, classification_report

y_pred_proba = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred_proba)
print(f"ROC-AUC: {auc:.4f}")
```

---

## 📚 Documentation

- **EDA_Report_Final.html** - Complete exploratory data analysis with integrated Python source code
- **PREPROCESSING_REPORT.md** - Detailed preprocessing documentation and technical summary
- **data_preprocessing.py** - Production-ready data preparation pipeline with full annotations

---

## 🔍 Key Technical Decisions

### Why These Features?
1. **Financial Ratios:** Capture affordability and debt burden (critical for credit risk)
2. **Bureau Aggregates:** Historical payment behavior is the strongest default predictor
3. **Missing Indicators:** Missingness patterns are often predictive (e.g., no bureau history = first-time borrower)
4. **Interaction Terms:** Capture non-linear relationships (e.g., age + income interaction)
5. **Binned Variables:** Create non-linear boundaries and categorical segments

### Why XGBoost?
- Handles missing values naturally
- Robust to outliers
- Captures complex non-linear relationships
- Excellent with imbalanced data (via scale_pos_weight)
- Provides feature importance for interpretability

### Train/Test Consistency
All imputation values, binning thresholds, and transformation parameters are computed from **training data only** and applied consistently to test data. This prevents data leakage and ensures valid model evaluation.

---

## 📊 Feature Importance (Expected Top 10)

Based on EDA and domain knowledge:

1. **EXT_SOURCE_MEAN** - External credit scores
2. **BUREAU_OVERDUE_RATIO** - Historical delinquency
3. **INSTALL_LATE_RATE** - Payment punctuality
4. **CREDIT_INCOME_RATIO** - Affordability
5. **YEARS_EMPLOYED** - Employment stability
6. **AGE_YEARS** - Client maturity
7. **BUREAU_DEBT_CREDIT_RATIO** - Current debt burden
8. **PREV_APPROVAL_RATE** - Application history
9. **ANNUITY_INCOME_RATIO** - Debt service ratio
10. **EXT_SOURCE_WEIGHTED** - Weighted credit scores

---

## 🛠️ Technologies Used

- **Python 3.11**
- **Polars** - Fast DataFrame library for data processing
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning utilities
- **XGBoost** - Gradient boosting (recommended for modeling)

---

## 📝 Project Status

- ✅ **Phase 1: EDA** - Complete (comprehensive analysis in EDA_Report_Final.html)
- ✅ **Phase 2: Data Preprocessing** - Complete (data_preprocessing.py)
- ⏳ **Phase 3: Feature Selection** - Pending
- ⏳ **Phase 4: Model Training** - Pending
- ⏳ **Phase 5: Model Evaluation** - Pending
- ⏳ **Phase 6: Model Deployment** - Pending

---

## 👤 Author

**Abdul**  
Data Analysis Project - February 2026

---

## 📄 License

This project is for educational purposes as part of a data analysis course.

---

## 🙏 Acknowledgments

- Home Credit Group for providing the dataset
- Course instructors for guidance and requirements
- Open-source community for excellent tools (Polars, XGBoost, etc.)

---

## 📧 Contact

For questions or collaboration opportunities, please open an issue in this repository.

---

*Last Updated: February 7, 2026*
