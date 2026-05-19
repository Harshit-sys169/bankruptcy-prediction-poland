![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-0.41%2B-lightblue)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Very Advanced](https://img.shields.io/badge/Complexity-Very%20Advanced-darkred)

# Corporate Bankruptcy Prediction in Poland

End-to-end machine learning project for predicting corporate bankruptcy with advanced techniques for handling real-world challenges: imbalanced data, asymmetric misclassification costs, and model interpretability.

## Project Overview

Built a sophisticated bankruptcy prediction system that goes beyond standard classification to address practical business concerns. The analysis demonstrates production-grade ML practices including proper handling of class imbalance, ensemble methods, business-aware cost optimization, and SHAP model interpretability.

## Dataset
- **Source:** Polish bankruptcy prediction dataset
- **Target:** Bankruptcy (binary: company bankrupt or survived)
- **Challenge:** Highly imbalanced (rare bankruptcies in real data)
- **Features:** Financial ratios and company metrics

## Core Components

### 1. Imbalanced Data Handling (`bankruptcy_poland_advanced.py`)

**The Imbalance Problem:**
- Real bankruptcy datasets: 95%+ non-bankrupt, <5% bankrupt
- Naive classifier: "predict never bankrupt" achieves 95% accuracy (useless)
- Standard metrics (accuracy) misleading

**Solutions Implemented:**
- **SMOTE (Synthetic Minority Oversampling):**
  * Creates synthetic minority examples via interpolation
  * Reduces overfitting risk vs random oversampling
  * Effective for small sample sizes
  
- **Random Oversampling:**
  * Duplicates minority examples
  * Simpler than SMOTE
  * Works when risk of overfitting is low

**Evaluation Metrics for Imbalance:**
- **F1 Score:** Harmonic mean of precision/recall (primary metric)
- **ROC-AUC:** Performance across all thresholds
- **Balanced Accuracy:** Average recall for each class
- **Precision-Recall Curves:** More informative than ROC for imbalanced data

### 2. Ensemble Methods Comparison (`bankruptcy_poland_advanced.py`)

**Random Forest:**
- Reduces variance through ensemble averaging
- Class weighting handles imbalance
- Feature importance from tree splits

**Gradient Boosting:**
- Sequentially focuses on misclassified examples
- Naturally handles imbalance by emphasizing hard cases
- Often best performance on imbalanced data

**AdaBoost:**
- Designed specifically for difficult-to-classify examples
- Weights samples, boosting the hardest ones
- Effective for binary classification

**Logistic Regression (Baseline):**
- Class weighting for imbalance
- Interpretable coefficients
- Fast training and prediction

### 3. Cost-Sensitive Learning (`cost_sensitive_learning.py`)

**Business Reality:**
- False Negative (predicting non-bankrupt when bankrupt) >> False Positive cost
- Missed bankruptcies have huge business impact
- Predicting bankruptcy unnecessarily has lower cost

**Implementation:**
- Asymmetric misclassification costs
- Threshold optimization based on business costs
- Compare different cost ratios
- Find optimal decision threshold

**Decision Threshold Adjustment:**
- Standard: predict bankruptcy if probability > 0.5
- Cost-aware: adjust to probability > 0.3 (more conservative)
- Business metric: cost per misclassification

### 4. Model Interpretability (`bankruptcy_poland_advanced.py`)

**Feature Importance:**
- Tree-based: Which features split nodes most often
- Shows relative importance from model's perspective

**SHAP Values:**
- Game-theoretic approach to feature importance
- Per-prediction explanation: which features pushed score up/down
- Consistent, theoretically sound
- Can explain any model (TreeExplainer for trees, KernelExplainer for any)

**Confusion Matrix Analysis:**
- Understanding error patterns
- False Positive vs False Negative trade-offs
- Precision-Recall by class

### 5. Business Metrics (`business_metrics.py`)

**Financial Impact:**
- Not just accuracy, but dollars saved
- Bankruptcy detection value
- False alarm costs
- Break-even analysis

**Cost Curves:**
- How costs change with decision threshold
- Find optimal threshold for specific business situation
- Scenario analysis for different cost ratios

## Project Structure
```
bankruptcy-prediction-poland/
├── bankruptcy_poland_advanced.py    # Ensemble + SHAP + metrics
├── cost_sensitive_learning.py       # Asymmetric costs
├── business_metrics.py              # Financial impact
├── requirements.txt
└── README.md
```

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install shap  # Optional but recommended
   ```

2. **Prepare data:**
   - Place data in `data/` directory

3. **Run main analysis:**
   ```bash
   python bankruptcy_poland_advanced.py
   ```
   Output:
   - Model comparison (accuracy, F1, ROC-AUC)
   - Confusion matrices
   - ROC and Precision-Recall curves
   - SHAP values and interpretability

4. **Run cost-sensitive analysis:**
   ```bash
   python cost_sensitive_learning.py
   ```
   Output:
   - Impact of class weighting
   - Threshold optimization
   - Business metric comparison

5. **Run business metrics:**
   ```bash
   python business_metrics.py
   ```
   Output:
   - Cost-benefit analysis
   - Optimal threshold for business
   - ROI calculations

## Key Results

**Model Comparison:**
- Gradient Boosting typically outperforms on imbalanced data
- Ensemble methods beat single algorithms
- Proper metrics (F1, ROC-AUC) show true performance

**Imbalance Handling:**
- SMOTE significantly improves minority class detection
- Class weighting effective for ensemble methods
- Balanced evaluation metrics essential

**Cost-Sensitive Optimization:**
- Optimal threshold rarely at 0.5
- Depends on specific business cost structure
- Conservative predictions better for bankruptcy

**Interpretability:**
- SHAP values show which features drive predictions
- Helps stakeholders trust and understand model
- Can explain individual predictions

## Technologies Used
- **Data Processing:** pandas, numpy
- **Machine Learning:** scikit-learn
- **Ensemble Methods:** Random Forest, Gradient Boosting, AdaBoost
- **Imbalance:** imblearn (SMOTE, oversampling)
- **Interpretability:** SHAP
- **Visualization:** matplotlib, seaborn

## Key Competencies Demonstrated

1. **Handling Class Imbalance**
   - Why standard accuracy fails
   - SMOTE vs oversampling trade-offs
   - Appropriate evaluation metrics

2. **Ensemble Methods**
   - Understanding when each method works best
   - Combining predictions from multiple models
   - Feature importance from ensembles

3. **Business-Aware ML**
   - Translating cost metrics to threshold
   - Understanding that statistics ≠ business value
   - Cost-benefit analysis

4. **Model Interpretability**
   - Why black-box models need explanation
   - SHAP game-theoretic approach
   - Per-prediction explanations

5. **Real-World Problem Solving**
   - Working with messy, imbalanced data
   - Building systems that balance multiple objectives
   - Explaining complex models to stakeholders

## Lessons Learned

1. **Accuracy is misleading:** Always use appropriate metrics for imbalanced data
2. **F1 score > accuracy:** For imbalanced classification
3. **Ensemble beats individual models:** Diversity is strength
4. **SHAP enables trust:** Interpretability drives adoption
5. **Cost-awareness matters:** Statistical optimality ≠ business optimality
6. **Threshold matters more than algorithm:** For real-world deployment

## References
- Class Imbalance: He, H., & Garcia, E. A. (2009). Learning from imbalanced data. IEEE Transactions on Knowledge and Data Engineering
- SHAP: Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions
- Cost-Sensitive Learning: Elkan, C. (2001). The Foundations of Cost-Sensitive Learning. International Joint Conference on Artificial Intelligence
- Ensemble Methods: Breiman, L. (1996). Bagging predictors. Machine Learning; Schapire, R. E. (1990). The Strength of Weak Learnability
- Imbalanced Learning: Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research

---

**Author:** Harshit  
**GitHub:** https://github.com/Harshit-sys169
