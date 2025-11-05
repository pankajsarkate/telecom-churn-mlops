import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import os

# Create dummy telecom data
df = pd.DataFrame({
    'customer_id': ['CUST' + str(i).zfill(4) for i in range(1000)],
    'gender': np.random.choice(['Male', 'Female'], 1000),
    'senior_citizen': np.random.randint(0, 2, 1000),
    'partner': np.random.choice(['Yes', 'No'], 1000),
    'dependents': np.random.choice(['Yes', 'No'], 1000),
    'tenure_months': np.random.randint(1, 72, 1000),
    'monthly_charges': np.round(np.random.uniform(20, 120, 1000), 2),
    'support_calls': np.random.randint(0, 6, 1000),
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], 1000),
    'churn': np.random.choice(['Yes', 'No'], 1000, p=[0.3, 0.7])
})

# Encode categorical columns
df['churn'] = df['churn'].map({'Yes': 1, 'No': 0})
cats = ['gender', 'partner', 'dependents', 'contract_type']
le = LabelEncoder()
for col in cats:
    df[col] = le.fit_transform(df[col])

# Prepare features and target
X = df.drop(columns=['customer_id', 'churn'])
y = df['churn']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE
smote = SMOTE()
X_res, y_res = smote.fit_resample(X_train_scaled, y_train)

# Train model
rf = RandomForestClassifier()
rf.fit(X_res, y_res)

# Evaluate
print(classification_report(y_test, rf.predict(X_test_scaled)))
print("ROC AUC:", roc_auc_score(y_test, rf.predict_proba(X_test_scaled)[:, 1]))

# Save model and scaler
os.makedirs("models", exist_ok=True)
joblib.dump(rf, 'models/telecom_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
