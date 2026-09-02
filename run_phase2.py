import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.feature_engineering import FeatureEngineer
from src.models import CreditRiskModels

def main():
    print("🚀 Starting Phase 2: Feature Engineering & Baseline Models...")
    
    # 1. Load Cleaned Data
    app_clean = pd.read_parquet("processed_data/application_train_cleaned.parquet")
    inst_clean = pd.read_parquet("processed_data/installments_payments_cleaned.parquet")
    
    # 2. Feature Engineering (Generates the 167 Features)
    engineer = FeatureEngineer()
    features_df = engineer.build_feature_matrix(app_clean, inst_clean)
    
    # Save the 167-feature matrix
    features_df.to_parquet("processed_data/application_features.parquet")
    print(f"✅ Feature Engineering complete. Matrix shape: {features_df.shape}")
    
    # 3. Prepare Data for Modeling
    X = features_df.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
    # Fill NAs dynamically via median for Logistic/XGBoost compatibility
    X = X.fillna(X.median())
    y = features_df['TARGET']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Train Models
    print("🧠 Training Baseline and Champion Models...")
    trainer = CreditRiskModels()
    trainer.train_baseline_logistic(X_train, y_train)
    trainer.train_xgboost(X_train, y_train)
    trainer.train_lightgbm_champion(X_train, y_train)
    
    # 5. Evaluate and Save
    results = []
    for model_name in ['logistic_regression_pipeline', 'xgboost_model', 'lightgbm_model']:
        res = trainer.evaluate(model_name, X_test, y_test)
        results.append(res)
        
    results_df = pd.DataFrame(results)
    print("\n📊 Model Benchmark Results:")
    print(results_df)
    
    os.makedirs("model_outputs", exist_ok=True)
    results_df.to_csv("model_outputs/model_benchmark_comparison.csv", index=False)
    
    trainer.save_models()
    print("✅ Phase 2 Complete! Models saved to models/ directory.")

if __name__ == "__main__":
    main()
