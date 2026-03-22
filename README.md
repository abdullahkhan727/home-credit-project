# 🏦 Home Credit Default Risk: End-to-End ML Project

**Predicting loan default risk using gradient boosting and feature engineering**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0.0-green.svg)](https://github.com/microsoft/LightGBM)
[![Kaggle Score](https://img.shields.io/badge/Kaggle%20AUC-0.764-orange.svg)](https://www.kaggle.com/competitions/home-credit-default-risk)

**Author:** Abdul | **Project Date:** January - March 2026

---

## 📊 Project Overview

This project tackles a real-world credit risk problem: **predicting which loan applicants are likely to default**. Using data from Home Credit Group, I built a machine learning pipeline that could save the company **$27 million annually** while maintaining competitive approval rates.

### Business Impact

**The model enables Home Credit to:**

- Reduce default rate by **35%** (from 8.0% to 5.2% among approved loans)
- Maintain **82% approval rate** for market competitiveness
- Generate **$27M additional annual profit** (based on 300K applications)
- Provide **regulatory-compliant explanations** for denied applications

**Key Achievement:** Validation ROC-AUC of **0.778** and Kaggle test score of **0.764**, demonstrating strong generalization.

---

## 🎯 Key Technical Accomplishments

### 1. Comprehensive Feature Engineering (78 New Features)
Created 78 engineered features from supplementary data sources, improving model AUC by **10.8%**:

- **Bureau credit history aggregates:** 16 features (overdue ratios, active credit metrics)
- **Previous application patterns:** 13 features (approval rates, refusal history)
- **Installment payment behavior:** 12 features (late payment rates, DPD metrics)
- **Financial ratios:** 11 features (credit/income, debt/credit, annuity/income)
- **Missing data indicators:** 15 features (capturing data quality patterns)

**Impact:** Application data alone achieved 0.702 AUC; with supplementary features: **0.778 AUC** (+10.8%)

### 2. Systematic Model Comparison
Evaluated 5 models with rigorous cross-validation:

| Model | Validation AUC | Training Time | Selected |
|-------|----------------|---------------|----------|
| Baseline (Majority Class) | 0.500 | <1s | ✗ |
| Logistic Regression | 0.681 | 30s | ✗ |
| Random Forest | 0.738 | 12m | ✗ |
| XGBoost | 0.766 | 8m | ✗ |
| **LightGBM** | **0.778** | **3m** | **✓** |

**Selection Rationale:** LightGBM achieved best AUC (0.778) with fastest training time and excellent handling of class imbalance (8% default rate).

### 3. Cost-Sensitive Threshold Optimization
Used realistic lending economics to optimize decision threshold:

- **Industry-researched costs:** $850 profit per repaid loan, $3,200 loss per default
- **Tested 5 thresholds:** 0.05 to 0.20
- **Optimal threshold: 0.12** maximizes expected profit at $6.1M per 10K applications

### 4. Model Explainability (SHAP)
Top 5 predictive features account for 46.6% of model importance:

1. **EXT_SOURCE_3** (14.2%) - External credit bureau score
2. **EXT_SOURCE_2** (11.8%) - Alternative credit bureau score
3. **EXT_SOURCE_MEAN** (8.9%) - Average external credit scores
4. **BUREAU_OVERDUE_RATIO** (6.7%) - Historical delinquency rate
5. **CREDIT_INCOME_RATIO** (5.4%) - Loan affordability measure

**Insight:** Credit bureau data accounts for 35% of importance - model heavily dependent on external scores.

### 5. Fairness and Compliance Analysis
Conducted thorough fairness audit:

- **Gender disparity:** 5% approval gap (Female 79% vs Male 84%)
- **Interesting finding:** Females have *lower* default rate yet lower approval
- Developed **ECOA/FCRA-compliant adverse action mapping**

---

## 💼 Interview Talking Points

### For Technical Interviews:
> "I built a LightGBM model achieving 0.778 AUC on a highly imbalanced dataset (8% defaults). Key was engineering 78 features from supplementary data - this alone improved AUC by 10.8%. I handled class imbalance through scale_pos_weight and systematically compared 5 models."

### For Business Interviews:
> "My model would generate $27 million additional annual profit by reducing defaults 35% while maintaining 82% approval rate. I optimized the threshold using realistic lending economics - $850 profit per good loan vs $3,200 loss per default."

### For Ethics Questions:
> "I identified a 5% gender approval disparity through fairness analysis. Interestingly, female applicants have lower default rates, suggesting possible bias. I developed ECOA-compliant adverse action mapping and recommended monthly monitoring."

### For Data Questions:
> "I engineered 78 features by aggregating 1.7M bureau records, 1.7M previous applications, and 13.6M installment payments. External credit scores account for 35% of importance, identifying a product opportunity for thin-file applicants."

---

## 📂 Project Portfolio

### 1. [model_development.ipynb](model_development.ipynb) - Model Development

**Systematic model selection and optimization**

**Contents:**
- Baseline establishment
- Model comparison (5 algorithms)
- Class imbalance strategies
- Hyperparameter tuning
- Feature importance analysis
- Kaggle submission

**Key Results:**
- LightGBM selected (0.778 AUC)
- Class weights best for imbalance
- 78 features improved AUC 10.8%

**Skills:** Model selection, hyperparameter optimization, imbalanced data, cross-validation

---

### 2. [model_card.qmd](model_card.qmd) - Production Model Card

**Business stakeholder documentation**

**Contents:**
- Executive summary ($27M impact)
- Model details & performance
- Decision threshold analysis (5 scenarios)
- SHAP explainability (top 10 features)
- Adverse action mapping (ECOA/FCRA)
- Fairness analysis (gender/education)
- Limitations & risks (8 categories)

**Key Findings:**
- Optimal threshold: 0.12
- Gender disparity: 5% gap
- Credit bureau dependency: 35%
- Annual impact: $27M profit

**Skills:** Business communication, cost-benefit analysis, fairness auditing, regulatory compliance

---

### 3. [data_preprocessing.py](data_preprocessing.py) - Data Pipeline

**Production-ready feature engineering**

**Contents:**
- Multi-table data loading
- 78 feature engineering functions
- Aggregation pipelines
- Missing value imputation
- Categorical encoding
- Data validation

**Features Created:**
- Financial ratios (11)
- Bureau aggregates (16)
- Previous app aggregates (13)
- Installment aggregates (12)
- Missing indicators (15)

**Skills:** Data engineering, feature engineering, pipeline development, quality assurance

---

## 📈 Results

### Model Performance

| Metric | Validation | Kaggle Test |
|--------|------------|-------------|
| **ROC-AUC** | 0.778 | 0.764 |
| **Precision (Default)** | 41% | - |
| **Recall (Default)** | 68% | - |
| **Approval Rate** | 82% | - |

### Business Value

| Scenario | Approval | Default Rate | Annual Profit |
|----------|----------|--------------|---------------|
| Without Model | 92% | 8.0% | $156M |
| **With Model** | **82%** | **5.2%** | **$183M** |
| **Improvement** | -10 pts | -2.8 pts | **+$27M** |

---

## 🔍 Key Insights

### 1. External Credit Scores Dominate (35% importance)
- **Strength:** Strong prediction for established credit
- **Weakness:** Poor performance for thin-file applicants
- **Opportunity:** Alternative credit scoring product

### 2. Payment History Predicts Defaults (11.5% combined)
- Bureau overdue ratio: 6.7%
- Installment late rate: 4.8%
- **Actionable:** Maintain clean payment records

### 3. Fairness Requires Monitoring
- 5% gender gap despite lower female default rate
- Suggests possible proxy discrimination
- Monthly audits recommended

### 4. Supplementary Data Essential (+10.8% AUC)
- Application only: 0.702 AUC
- With supplementary: 0.778 AUC
- Investment in data infrastructure pays off

---

## 🎓 Skills Demonstrated

**Machine Learning:**
- Gradient boosting (LightGBM, XGBoost)
- Hyperparameter optimization
- Class imbalance handling
- Model explainability (SHAP)

**Data Engineering:**
- Large-scale data processing (13.6M records)
- Feature engineering
