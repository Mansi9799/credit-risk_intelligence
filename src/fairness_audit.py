import os
import joblib
import pandas as pd
import numpy as np
import json
from sklearn.metrics import confusion_matrix

# Configuration
DATA_PATH = "processed_data\application_features.parquet"
MODEL_PATH = "models\xgboost_model.joblib"
ENCODER_PATH = "models\tree_label_encoders.joblib"
OUTPUT_DIR = "fairness_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class CreditRiskModelWrapper:
    def __init__(self, model_path, encoder_path):
        self.model = joblib.load(model_path)
        self.encoders = joblib.load(encoder_path) if os.path.exists(encoder_path) else {}
        
    def predict_proba(self, X):
        X_encoded = X.copy()
        for col, enc in self.encoders.items():
            if col in X_encoded.columns:
                known_classes = set(enc.classes_)
                X_encoded[col] = X_encoded[col].apply(lambda x: str(x) if str(x) in known_classes else enc.classes_[0])
                X_encoded[col] = enc.transform(X_encoded[col].astype(str))
                
        for col in X_encoded.columns:
            X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce').fillna(0)
            
        return self.model.predict_proba(X_encoded)

def calculate_fairness_metrics(y_true, y_pred, sensitive_attr):
    """
    Calculates fairness metrics for a binary sensitive attribute (e.g., Male vs Female).
    y_pred: 1 if Default (Declined), 0 if Non-Default (Approved)
    """
    metrics = {}
    groups = sensitive_attr.unique()
    
    group_metrics = {}
    for g in groups:
        mask = (sensitive_attr == g)
        y_t = y_true[mask]
        y_p = y_pred[mask]
        
        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        if cm.shape == (2,2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0,0,0,0 # fallback
            
        approval_rate = (tn + fn) / len(y_t) if len(y_t) > 0 else 0  # Predict 0 means Approved
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        group_metrics[str(g)] = {
            "Approval_Rate": approval_rate,
            "True_Positive_Rate (Default correctly caught)": tpr,
            "False_Positive_Rate (Safe person wrongly declined)": fpr,
            "Population_Count": len(y_t)
        }
    
    # Calculate Disparate Impact (Demographic Parity Ratio)
    # Usually Ratio of Approval Rate of unprivileged group / privileged group
    # We will just print all rates so the auditor can see
    
    return group_metrics

def main():
    print("Loading data for Fairness Audit...")
    df = pd.read_parquet(DATA_PATH, engine='pyarrow')
    
    # Use a sample for speed
    df_sample = df.sample(n=50000, random_state=42).copy()
    
    # The original raw categorical columns were encoded, but we want the raw demographic attributes.
    # We will rely on 'CODE_GENDER' and 'DAYS_BIRTH'
    if 'CODE_GENDER' not in df_sample.columns or 'DAYS_BIRTH' not in df_sample.columns:
        print("Sensitive attributes not found in the dataset.")
        return
        
    print("Loading Model...")
    wrapper = CreditRiskModelWrapper(MODEL_PATH, ENCODER_PATH)
    
    features = [c for c in df.columns if c not in ['SK_ID_CURR', 'TARGET']]
    X = df_sample[features]
    y_true = df_sample['TARGET']
    
    print("Generating Predictions...")
    probas = wrapper.predict_proba(X)[:, 1]
    y_pred = (probas > 0.5).astype(int)  # 1 = Decline, 0 = Approve
    
    print("Auditing Gender (CODE_GENDER)...")
    gender_audit = calculate_fairness_metrics(y_true, y_pred, df_sample['CODE_GENDER'])
    
    print("Auditing Age (Converting DAYS_BIRTH to Cohorts)...")
    age_years = abs(df_sample['DAYS_BIRTH']) / 365.25
    age_cohorts = pd.cut(age_years, bins=[18, 30, 45, 60, 100], labels=['18-30', '31-45', '46-60', '60+'])
    age_audit = calculate_fairness_metrics(y_true, y_pred, age_cohorts)
    
    report = {
        "Gender_Audit": gender_audit,
        "Age_Cohort_Audit": age_audit
    }
    
    with open(os.path.join(OUTPUT_DIR, "fairness_audit_report.json"), "w") as f:
        json.dump(report, f, indent=4)
        
    print("\n--- Fairness Audit Summary ---")
    print("\nGender Bias Analysis (Approval Rate):")
    for k, v in gender_audit.items():
        if v['Population_Count'] > 100:
            print(f" - {k}: {v['Approval_Rate']*100:.2f}% Approved")
            
    print("\nAge Bias Analysis (Approval Rate):")
    for k, v in age_audit.items():
        if v['Population_Count'] > 100:
            print(f" - {k}: {v['Approval_Rate']*100:.2f}% Approved")
            
    print("\nStage 10: Fairness & Disparate Impact Audit complete.")

if __name__ == "__main__":
    main()
