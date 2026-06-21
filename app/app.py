import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and features
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, 'model.pkl'))
features = joblib.load(os.path.join(BASE_DIR, 'features.pkl'))

# Page config
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("Customer Churn Risk Predictor")
st.markdown("Enter customer details below to predict churn risk and understand the key drivers.")

# ── Sidebar inputs ──────────────────────────────────────────
st.sidebar.header("Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 0.0, 120.0, 65.0)
total_charges = st.sidebar.slider("Total Charges ($)", 0.0, 9000.0, 1000.0)

# ── Build input dataframe ────────────────────────────────────
input_dict = {
    'gender': 1 if gender == 'Male' else 0,
    'SeniorCitizen': 1 if senior == 'Yes' else 0,
    'Partner': 1 if partner == 'Yes' else 0,
    'Dependents': 1 if dependents == 'Yes' else 0,
    'tenure': tenure,
    'PhoneService': 1 if phone == 'Yes' else 0,
    'PaperlessBilling': 1 if paperless == 'Yes' else 0,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'MultipleLines_No phone service': 1 if multiple_lines == 'No phone service' else 0,
    'MultipleLines_Yes': 1 if multiple_lines == 'Yes' else 0,
    'InternetService_Fiber optic': 1 if internet == 'Fiber optic' else 0,
    'InternetService_No': 1 if internet == 'No' else 0,
    'OnlineSecurity_No internet service': 1 if online_security == 'No internet service' else 0,
    'OnlineSecurity_Yes': 1 if online_security == 'Yes' else 0,
    'OnlineBackup_No internet service': 1 if online_backup == 'No internet service' else 0,
    'OnlineBackup_Yes': 1 if online_backup == 'Yes' else 0,
    'DeviceProtection_No internet service': 1 if device_protection == 'No internet service' else 0,
    'DeviceProtection_Yes': 1 if device_protection == 'Yes' else 0,
    'TechSupport_No internet service': 1 if tech_support == 'No internet service' else 0,
    'TechSupport_Yes': 1 if tech_support == 'Yes' else 0,
    'StreamingTV_No internet service': 1 if streaming_tv == 'No internet service' else 0,
    'StreamingTV_Yes': 1 if streaming_tv == 'Yes' else 0,
    'StreamingMovies_No internet service': 1 if streaming_movies == 'No internet service' else 0,
    'StreamingMovies_Yes': 1 if streaming_movies == 'Yes' else 0,
    'Contract_One year': 1 if contract == 'One year' else 0,
    'Contract_Two year': 1 if contract == 'Two year' else 0,
    'PaymentMethod_Credit card (automatic)': 1 if payment == 'Credit card (automatic)' else 0,
    'PaymentMethod_Electronic check': 1 if payment == 'Electronic check' else 0,
    'PaymentMethod_Mailed check': 1 if payment == 'Mailed check' else 0,
}

input_df = pd.DataFrame([input_dict])[features]

# ── Predict ──────────────────────────────────────────────────
prob = model.predict_proba(input_df)[0][1]
prediction = model.predict(input_df)[0]

# Risk level
if prob >= 0.7:
    risk = "🔴 High Risk"
    color = "red"
elif prob >= 0.4:
    risk = "🟠 Medium Risk"
    color = "orange"
else:
    risk = "🟢 Low Risk"
    color = "green"

# ── Display results ───────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Prediction")
    st.markdown(f"### {risk}")
    st.metric("Churn Probability", f"{prob:.1%}")

with col2:
    st.subheader("Risk Gauge")
    fig, ax = plt.subplots(figsize=(4, 0.5))
    ax.barh(["Risk"], [prob], color=color, height=0.4)
    ax.barh(["Risk"], [1 - prob], left=[prob], color="#e0e0e0", height=0.4)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.4, 0.7, 1.0])
    ax.set_xticklabels(["0%", "40%", "70%", "100%"])
    ax.axis('on')
    st.pyplot(fig)

# ── SHAP explanation ──────────────────────────────────────────
st.subheader("Why is this customer at risk?")
explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(input_df)

fig2, ax2 = plt.subplots(figsize=(8, 4))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_vals[0],
        base_values=explainer.expected_value,
        data=input_df.iloc[0],
        feature_names=features
    ),
    show=False
)
st.pyplot(plt.gcf())