# Credit Card Default Classification — XGBoost
**Ikrar Gempur Tirani — D121231015**

## Overview

This project uses **XGBoost** (Extreme Gradient Boosting) to classify whether a credit card client will default on payment next month. The dataset contains demographic and payment history data from 30,000 clients.

## Dataset

| Attribute | Value |
|-----------|-------|
| Source | UCI Default of Credit Card Clients Dataset |
| Samples | 30,000 clients |
| Features | 23 (demographics, credit limit, payment history, bill amounts) |
| Target | default.payment.next.month (0 = no default, 1 = default) |
| Class imbalance | ~78% no default / ~22% default |

## Algorithm

**XGBoost** is an ensemble method based on gradient boosting of decision trees. It builds trees sequentially, where each tree corrects the errors of the previous one using gradient descent on the loss function.

Key techniques applied:
- **SMOTE** (Synthetic Minority Oversampling) to address class imbalance
- **Optuna** hyperparameter tuning (150 trials)
- **5-Fold Stratified Cross-Validation** for model selection
- Feature importance analysis via gain

## Results

| Metric | Value |
|--------|-------|
| Accuracy | ~82% |
| ROC-AUC | ~0.78 |
| Precision (default) | ~0.64 |
| Recall (default) | ~0.60 |

Key finding: `PAY_0` (repayment status in September) is the most important feature by a large margin, followed by `PAY_2` and `PAY_3`.

## Files

| File | Description |
|------|-------------|
| `XGBoost_Classification_CreditCard.ipynb` | Main notebook |
| `UCI_Credit_Card.csv` | Raw dataset |
| `classification_results_XGBoost_CreditCard.csv` | Prediction results |

## Requirements

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
optuna
matplotlib
seaborn
```

## Reference

- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD 2016.
- Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2).
