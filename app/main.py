from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("models/telecom_model.pkl")
scaler = joblib.load("models/scaler.pkl")

app = FastAPI()  # ✅ THIS MUST BE DEFINED before @app.post

class Customer(BaseModel):
    tenure_months: int
    monthly_charges: float
    support_calls: int
    senior_citizen: int
    gender: int
    partner: int
    dependents: int
    contract_type: int

@app.post("/predict")
def predict_churn(customer: Customer):
    X = np.array([[customer.tenure_months,
                   customer.monthly_charges,
                   customer.support_calls,
                   customer.senior_citizen,
                   customer.gender,
                   customer.partner,
                   customer.dependents,
                   customer.contract_type]])
    
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0][1]
    
    return {"churn_probability": round(prob, 3)}
