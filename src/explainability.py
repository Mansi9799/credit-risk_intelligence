import os
import joblib
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
import json

# Configuration
DATA_PATH = "processed_data\application_features.parquet"
MODEL_PATH = "models\xgboost_model.joblib"
OUTPUT_DIR = "explainability_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_NAME_MAPPING = {
    "EXT_SOURCE_1": "External Credit Score (Bureau 1)",
    "EXT_SOURCE_2": "External Credit Score (Bureau 2)",
    "EXT_SOURCE_3": "External Credit Score (Bureau 3)",
    "EXT_SOURCE_MEAN": "Average External Credit Score",
    "EXT_SOURCE_MIN": "Minimum External Credit Score",
    "DEBT_TO_INCOME_RATIO": "Debt-to-Income Ratio",
    "ANNUITY_TO_INCOME_RATIO": "Annuity-to-Income Ratio",
    "PAYMENT_BURDEN_INDEX": "Payment Burden Index",
    "INST_AVG_DPD": "Average Days Past Due (Historical)",
    "INST_PCT_LATE": "Percentage of Late Payments",
    "INST_TOTAL_DEFICIT": "Total Payment Deficit (Historical)",
    "AMT_CREDIT": "Total Loan Amount",
    "AMT_ANNUITY": "Loan Annuity Amount",
    "DAYS_EMPLOYED": "Length of Employment",
    "YEARS_EMPLOYED": "Years of Employment",
    "DAYS_BIRTH": "Client Age"
}

def get_readable_feature_name(feature_name):
    if feature_name in FEATURE_NAME_MAPPING:
        return FEATURE_NAME_MAPPING[feature_name]
    return feature_name.replace("_", " ").title()

def load_data_and_model(sample_size=1000):
    print("Loading XGBoost Model...")
    model = joblib.load(MODEL_PATH)
    
    print("Loading Feature Dataset...")
    df = pd.read_parquet(DATA_PATH, engine='pyarrow')
    
    print("Applying Label Encoders...")
    encoder_path = "models\tree_label_encoders.joblib"
    if os.path.exists(encoder_path):
        encoders = joblib.load(encoder_path)
        for col, enc in encoders.items():
            if col in df.columns:
                known_classes = set(enc.classes_)
                df[col] = df[col].apply(lambda x: str(x) if str(x) in known_classes else enc.classes_[0])
                df[col] = enc.transform(df[col].astype(str))
    
    features = [c for c in df.columns if c not in ['SK_ID_CURR', 'TARGET']]
    X = df[features]
    y = df['TARGET']
    
    # We will just take a sample to compute SHAP values quickly for the demo
    # Let's focus on a mix of defaults and non-defaults
    df_sample = df.sample(n=sample_size, random_state=42)
    X_sample = df_sample[features]
    
    return model, X_sample, df_sample['SK_ID_CURR'], df_sample['TARGET']

def generate_global_shap(model, X_sample):
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    print("Generating Global SHAP Summary Plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_global_shap_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()
    return explainer, shap_values

def generate_adverse_action_notices(explainer, shap_values, X_sample, client_ids, y_true, top_n=4):
    print("Generating Adverse Action Notices...")
    adverse_notices = []
    
    for idx in range(len(X_sample)): 
        client_id = client_ids.iloc[idx]
        actual_default = y_true.iloc[idx]
        
        # Get SHAP values for this individual
        client_shap_values = shap_values[idx]
        
        # We look for features that contributed most to the POSITIVE class (default risk)
        # So we sort by highest positive SHAP values
        feature_contributions = list(zip(X_sample.columns, client_shap_values, X_sample.iloc[idx]))
        feature_contributions.sort(key=lambda x: x[1], reverse=True)
        
        # Filter only those that strictly increased risk (shap_value > 0)
        risk_drivers = [fc for fc in feature_contributions if fc[1] > 0]
        top_reasons = risk_drivers[:top_n]
        
        reasons_formatted = []
        for feat, val, actual_val in top_reasons:
            reasons_formatted.append({
                "Feature": get_readable_feature_name(feat),
                "RawFeatureName": feat,
                "SHAP_Contribution": round(float(val), 4),
                "FeatureValue": round(float(actual_val), 4) if isinstance(actual_val, (int, float, np.number)) else str(actual_val)
            })
            
        notice = {
            "Client_ID": int(client_id),
            "Actual_Default": int(actual_default),
            "Risk_Score_LogOdds": round(float(sum(client_shap_values) + explainer.expected_value), 4),
            "Adverse_Action_Reasons": reasons_formatted
        }
        adverse_notices.append(notice)
        
    with open(os.path.join(OUTPUT_DIR, "adverse_action_notices.json"), "w") as f:
        json.dump(adverse_notices, f, indent=4)
        
    print(f"Generated notices for {len(adverse_notices)} clients.")
    # Show notices for top 3 highest risk clients from the sample
    adverse_notices.sort(key=lambda x: x['Risk_Score_LogOdds'], reverse=True)
    
    for notice in adverse_notices[:3]:
        print(f"\n--- Client {notice['Client_ID']} ---")
        print(f"Risk Score (LogOdds): {notice['Risk_Score_LogOdds']} | Actual Default: {notice['Actual_Default']}")
        print("Top Risk-Driving Factors:")
        for r in notice['Adverse_Action_Reasons']:
            print(f" - {r['Feature']}: (Value: {r['FeatureValue']}, Impact: +{r['SHAP_Contribution']})")

if __name__ == "__main__":
    model, X_sample, client_ids, y_true = load_data_and_model(sample_size=100) # Fast 100 for review
    explainer, shap_values = generate_global_shap(model, X_sample)
    generate_adverse_action_notices(explainer, shap_values, X_sample, client_ids, y_true)
    print("\nStage 7: SHAP Explainability complete.")
