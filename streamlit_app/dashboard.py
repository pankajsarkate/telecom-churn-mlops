import streamlit as st
import requests

st.title("📞 Telecom Churn Predictor")

gender = st.selectbox("Gender", [0, 1])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has Partner", [0, 1])
dependents = st.selectbox("Has Dependents", [0, 1])
tenure = st.slider("Tenure (months)", 0, 72, 12)
charges = st.slider("Monthly Charges", 20, 120, 60)
calls = st.slider("Support Calls", 0, 5, 1)
contract = st.selectbox("Contract Type", [0, 1, 2])

payload = {
    "tenure_months": tenure,
    "monthly_charges": charges,
    "support_calls": calls,
    "senior_citizen": senior,
    "gender": gender,
    "partner": partner,
    "dependents": dependents,
    "contract_type": contract
}

if st.button("Predict", key="predict_button"):
    try:
        response = requests.post("http://localhost:8000/predict", json=payload)
        st.write(response.json())
    except requests.exceptions.ConnectionError:
        st.error("❌ FastAPI server is not running at localhost:8000")
