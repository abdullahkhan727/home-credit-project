# 🏦 Home Credit Default Risk: ML Credit Scoring Solution

**Predicting loan default risk to increase profitability and reduce financial losses**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0.0-green.svg)](https://github.com/microsoft/LightGBM)
[![Kaggle Score](https://img.shields.io/badge/Kaggle%20AUC-0.764-orange.svg)](https://www.kaggle.com/competitions/home-credit-default-risk)

**Author:** Abdul | **Project Date:** January - March 2026

---

## 📊 Business Problem and Project Objective

### The Problem

Home Credit Group, a consumer finance provider, faces a critical business challenge: **how to accurately predict which loan applicants are likely to default while maintaining competitive approval rates**. The company serves customers with little or no credit history, making traditional credit scoring methods inadequate.

**Current Situation:**
- Default rate: 8% of loans (significant financial losses)
- Manual underwriting: Slow, inconsistent, expensive
- Limited credit history: 26M Americans are "credit invisible"
- Conservative lending: Missing profitable customers
- Competitive pressure: Need to approve 80%+ to maintain market share

**Business Impact of Defaults:**
- **Financial loss:** $3,200 per default (principal + collection costs)
- **Opportunity cost:** Overly conservative = losing customers to competitors
- **Regulatory risk:** Fair lending compliance requirements
- **Reputation damage:** High default rates affect brand perception

### Project Objective

**Primary Goal:** Build a machine learning model that predicts loan default probability to optimize lending decisions.

**Success Criteria:**
1. **Accuracy:** Achieve ROC-AUC > 0.75 on held-out test data
2. **Business Value:** Increase profitability vs. current system
3. **Fairness:** No discriminatory patterns by protected demographics
4. **Explainability:** Provide clear reasons for denied applications (regulatory requirement)
5. **Deployability:** Production-ready with monitoring framework

---

## 🎯 Our Solution to the Business Problem

### Model Architecture

**Algorithm Selected:** LightGBM (Light Gradient Boosting Machine)

**Why LightGBM:**
- Best performance among 5 models tested (0.778 validation AUC)
- Fastest training time (3 minutes vs 8+ for alternatives)
- Excellent handling of class imbalance (8% default rate)
- Built-in missing value handling (41% missing in some features)
- Production-proven at scale

### Key Technical Components

#### 1. Comprehensive Data Integration
Integrated 4 supplementary data sources:
- **Bureau credit history:** 1.7M records → Payment behavior patterns
- **Previous applications:** 1.7M records → Application history trends
- **Installment payments:** 13.6M records → Payment punctuality
- **POS cash balance:** 10M records → Credit utilization

#### 2. Advanced Feature Engineering (78 New Features)
Created features capturing:
- **Financial ratios (11):** Credit/income, debt/credit, payment/income
- **Bureau aggregates (16):** Overdue ratios, active credit counts, credit length
- **Previous app aggregates (13):** Approval rates, refusal patterns, amount trends
- **Installment aggregates (12):** Late payment rates, DPD metrics, payment volatility
- **Missing indicators (15):** Systematic missing data patterns

**Impact:** Features improved model AUC from 0.702 to 0.778 (+10.8%)

#### 3. Class Imbalance Strategy
Addressed severe imbalance (11.39:1 ratio):
- `scale_pos_weight=11.39` to weight minority class
- Stratified train-test split maintaining class distribution
- AUC metric (robust to imbalance) instead of accuracy
- Tested SMOTE and undersampling (class weights performed best)

#### 4. Cost-Sensitive Threshold Optimization
- Industry-researched costs: $850 profit/repaid, $3,200 loss/default
- Tested 5 thresholds: 0.05, 0.08, 0.10, 0.12, 0.15, 0.20
- **Optimal: 0.12** maximizes expected profit ($6.1M per 10K applications)
- Sensitivity analysis shows $300K revenue impact per threshold point

#### 5. Model Explainability (SHAP)
Top 5 features (46.6% of importance):
1. **EXT_SOURCE_3 (14.2%):** External credit bureau score
2. **EXT_SOURCE_2 (11.8%):** Alternative credit score
3. **EXT_SOURCE_MEAN (8.9%):** Average credit scores
4. **BUREAU_OVERDUE_RATIO (6.7%):** Historical delinquency
5. **CREDIT_INCOME_RATIO (5.4%):** Loan affordability

**Insight:** Credit bureau data = 35% of importance → Identifies thin-file gap

#### 6. Fairness and Compliance Framework
- **Fairness analysis:** Identified 5% gender approval gap
- **Adverse action mapping:** ECOA/FCRA-compliant denial reasons
- **Monitoring plan:** Monthly audits, quarterly reviews
- **Bias mitigation:** Recommendations for gender-specific calibration

### Performance Achieved

| Metric | Validation | Kaggle Test | Target |
|--------|------------|-------------|--------|
| **ROC-AUC** | 0.778 | 0.764 | >0.75 ✅ |
| **Precision (Default)** | 41% | - | >40% ✅ |
| **Recall (Default)** | 68% | - | >65% ✅ |
| **Approval Rate** | 82% | - | >80% ✅ |

**All objectives met!**

---

## 👤 My Contribution to the Project

As this was an **individual project**, I was responsible for all components:

### 1. Exploratory Data Analysis (EDA)
**My Work:**
- Analyzed 307,511 loan applications with 122 features
- Identified data quality issues (91.8/100 quality score)
- Discovered DAYS_EMPLOYED anomaly (365,243 placeholder)
- Found strong predictors: EXT_SOURCE scores, DAYS_BIRTH, REGION_RATING
- Visualized class imbalance (91.96% vs 8.04%)
- Documented missing patterns (EXT_SOURCE_3: 41% missing)

**Deliverable:** `EDA_Report_Final.html` - Comprehensive analysis with 20+ visualizations

### 2. Data Preprocessing and Feature Engineering
**My Work:**
- Built production pipeline processing 4 supplementary tables
- Engineered 78 new features from 13.6M+ records
- Created aggregation functions for bureau, previous apps, installments
- Implemented missing value imputation (median, persisted for test)
- Developed one-hot encoding with drop_first for categoricals
- Validated data quality and feature distributions

**Impact:** Improved model AUC by 10.8% (0.702 → 0.778)

**Deliverable:** `data_preprocessing.py` - Production-ready pipeline

### 3. Model Development and Selection
**My Work:**
- Established baseline (majority class: 0.500 AUC)
- Compared 5 algorithms: Logistic Regression, Random Forest, XGBoost, LightGBM, Neural Network
- Tested 3 class imbalance strategies: SMOTE, undersampling, class weights
- Performed hyperparameter tuning (randomized search, 20 iterations, 3-fold CV)
- Selected LightGBM based on AUC-speed tradeoff
- Generated Kaggle submission (0.764 public score)

**Deliverable:** `model_development.ipynb` - Systematic model comparison

### 4. Model Explainability and Analysis
**My Work:**
- Computed SHAP values on 1,000-row sample
- Identified top 10 predictive features
- Interpreted feature importance for business stakeholders
- Analyzed feature interactions and dependencies
- Documented model behavior for different applicant profiles

**Key Finding:** Credit bureau dependency (35%) identifies product opportunity for alternative scoring

### 5. Business Analysis and Threshold Optimization
**My Work:**
- Researched realistic lending economics (CFPB, Federal Reserve, TransUnion)
- Cost assumptions: $850 profit/repaid, $3,200 loss/default, 15% recovery
- Built cost-benefit model across 5 threshold scenarios
- Identified optimal threshold (0.12) maximizing profit
- Quantified annual impact: $27M additional profit (300K applications)
- Performed sensitivity analysis showing revenue implications

**Impact:** Translated model performance to business value

### 6. Fairness Analysis and Compliance
**My Work:**
- Analyzed approval rates by gender and education
- Identified 5% gender approval gap (Female 79% vs Male 84%)
- Interesting finding: Females have *lower* default rate (7.8% vs 8.4%)
- Calculated adverse impact ratio: 0.94 (passes 0.80 threshold)
- Developed feature-to-reason translation for adverse acti
