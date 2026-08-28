import os
import joblib
import pandas as pd
import dice_ml
import json

# Configuration
DATA_PATH = "processed_data\application_features.parquet"
MODEL_PATH = "models\xgboost_model.joblib"
ENCODER_PATH = "models\tree_label_encoders.joblib"
OUTPUT_DIR = "explainability_outputs"
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
        
        # DiCE converts numeric columns to objects internally, which XGBoost rejects.
        # We must cast everything back to numeric.
        for col in X_encoded.columns:
            X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce')
        
        # Ensure correct column order
        return self.model.predict_proba(X_encoded)
        
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

def main():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH, engine='pyarrow')
    
    print("Imputing missing values for DiCE compatibility...")
    for col in df.columns:
        if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(mode_val)
        else:
            df[col] = df[col].fillna(df[col].median())
            
    # We will sample 100 rows to keep it fast, but specifically target those who are classified as Risky (TARGET=1 or proba>0.5)
    # Actually, let's just use the wrapper
    model_wrapper = CreditRiskModelWrapper(MODEL_PATH, ENCODER_PATH)
    
    features = [c for c in df.columns if c not in ['SK_ID_CURR', 'TARGET']]
    X = df[features]
    
    # Identify high risk customers
    print("Identifying risky customers...")
    probas = model_wrapper.predict_proba(X.head(2000))[:, 1]
    risky_indices = [i for i, p in enumerate(probas) if p > 0.6] # >60% probability of default
    
    if not risky_indices:
        print("No high risk customers found in sample.")
        return
        
    # Take top 2 risky customers for demo
    target_customers = X.iloc[risky_indices[:2]]
    
    # Setup DiCE Data
    print("Configuring DiCE...")
    categorical_features = list(model_wrapper.encoders.keys())
    continuous_features = [c for c in features if c not in categorical_features]
    
    # To avoid memory issues and speed up, we pass a small subset of data to DiCE for background initialization
    df_dice = df.head(1000).copy()
    
    d = dice_ml.Data(dataframe=df_dice[features + ['TARGET']], 
                     continuous_features=continuous_features, 
                     outcome_name='TARGET')
                     
    # Setup DiCE Model
    # Since we are using a custom wrapper, we need to make it look like an sklearn model with a predict method
    # Or we use backend='sklearn' and DiCE will just call model.predict_proba
    m = dice_ml.Model(model=model_wrapper, backend="sklearn")
    
    # Setup DiCE Explainer using 'genetic' method which is much more robust
    exp = dice_ml.Dice(d, m, method="genetic")
    
    # Define actionable features (things a customer or underwriter can actually change)
    # We add more features to ensure a counterfactual can be found
    features_to_vary = ['AMT_CREDIT', 'AMT_ANNUITY', 'NAME_CONTRACT_TYPE', 'DEBT_TO_INCOME_RATIO', 'CREDIT_TERM_MONTHS', 'PAYMENT_BURDEN_INDEX', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    
    print("Generating Counterfactuals...")
    cf_results = []
    
    for i in range(len(target_customers)):
        query_instance = target_customers.iloc[[i]]
        print(f"\nProcessing Customer {i+1}...")
        
        try:
            dice_exp = exp.generate_counterfactuals(
                query_instance, 
                total_CFs=1, 
                desired_class="opposite",
                features_to_vary=features_to_vary
            )
            
            # Extract results
            cf_df = dice_exp.cf_examples_list[0].final_cfs_df
            
            print(f"Original Instance Predicted Probability: {model_wrapper.predict_proba(query_instance)[0][1]:.4f}")
            print("Counterfactuals (Minimum changes to flip to 'Approved'/'Low Risk'):")
            
            cf_data = []
            for j in range(len(cf_df)):
                changes = {}
                for col in features_to_vary:
                    orig_val = query_instance[col].values[0]
                    new_val = cf_df[col].values[j]
                    if orig_val != new_val:
                        changes[col] = f"{orig_val} -> {new_val}"
                        print(f" - Change {col}: {orig_val} -> {new_val}")
                cf_data.append(changes)
                
            cf_results.append({
                "Customer_Index": risky_indices[i],
                "Original_Risk_Prob": float(model_wrapper.predict_proba(query_instance)[0][1]),
                "Counterfactuals": cf_data
            })
            
        except Exception as e:
            print(f"Could not generate counterfactual for customer {i+1}: {e}")
            
    with open(os.path.join(OUTPUT_DIR, "counterfactuals.json"), "w") as f:
        json.dump(cf_results, f, indent=4)
        
    print("\nStage 8: Counterfactual Explanations complete.")

if __name__ == "__main__":
    main()
