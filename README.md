# 📊 Telecom Churn Prediction — End-to-End Data Science & MLOps Project

## 🧭 Business Objective  
Telecom companies lose millions in revenue due to **customer churn** — when subscribers discontinue their service.  
The goal of this project is to **predict the likelihood of churn** using machine learning so the business can:
- Identify at-risk customers early  
- Design targeted retention campaigns  
- Reduce overall churn rates  

---

## 🏗️ Project Architecture  

```mermaid
flowchart LR
  A[Data Sources: CRM, Billing, Network KPIs, Customer Support] --> B[Data Pipeline (ETL)]
  B --> C[Data Wrangling & Preprocessing]
  C --> D[Feature Engineering & Selection]
  D --> E[Model Training: ML + ANN]
  E --> F[Evaluation (ROC-AUC, Precision, Recall, F1)]
  F --> G[API (FastAPI) + Frontend (Streamlit)]
  G --> H[Deployment: Railway / GitHub Actions]
  H --> I[Monitoring & Drift Detection]
