# Customer Churn Prediction & Retention Analytics

## Problem
Telecom companies lose revenue when customers cancel their service. 
This project predicts which customers are likely to churn so retention 
teams can intervene before it happens.

## Approach
- Explored 7,043 customer records across 20 features
- Handled class imbalance using SMOTE
- Compared Logistic Regression, XGBoost, and LightGBM
- Selected LightGBM (ROC-AUC: 0.72, Churn Recall: 63%)
- Used SHAP to explain individual predictions

## Key Findings
- Fiber optic internet and electronic check payment are the strongest churn signals
- Month-to-month contract customers churn at significantly higher rates
- New customers (low tenure) are most at risk

## Tech Stack
Python · pandas · scikit-learn · XGBoost · LightGBM · SHAP · Streamlit

## How to Run
cd app
streamlit run app.py