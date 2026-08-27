import sys
import os

# Ensure the src module can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
from src.explainability import load_data_and_model, generate_global_shap, generate_adverse_action_notices
from src.counterfactuals import main as run_counterfactuals
from src.uplift_modeling import simulate_intervention_data, train_uplift_model, evaluate_uplift, DATA_PATH as UPLIFT_DATA_PATH
from src.fairness_audit import main as run_fairness_audit
from src.stress_testing import main as run_stress_testing

def main():
    print("======================================================")
    print("  PHASE 4: ADVANCED INTELLIGENCE & GOVERNANCE RUNNER  ")
    print("======================================================")
    
    print("\n[1/5] Running SHAP Explainability...")
    model, X_sample, client_ids, y_true = load_data_and_model(sample_size=100)
    explainer, shap_values = generate_global_shap(model, X_sample)
    generate_adverse_action_notices(explainer, shap_values, X_sample, client_ids, y_true)
    
    print("\n[2/5] Running DiCE Counterfactuals...")
    run_counterfactuals()
    
    print("\n[3/5] Running Uplift Modeling (EconML)...")
    print("Loading base dataset for Uplift...")
    df_uplift = pd.read_parquet(UPLIFT_DATA_PATH, engine='pyarrow')
    df_simulated = simulate_intervention_data(df_uplift, sample_size=10000) # Reduced sample for pipeline speed
    est, X_test, cate_pred, ite_test = train_uplift_model(df_simulated)
    evaluate_uplift(X_test, cate_pred, ite_test)
    
    print("\n[4/5] Running Fairness & Disparate Impact Audit...")
    run_fairness_audit()
    
    print("\n[5/5] Running CCAR / Basel Macroeconomic Stress Testing...")
    run_stress_testing()
    
    print("\n======================================================")
    print("          PHASE 4 EXECUTION FULLY COMPLETE            ")
    print("======================================================")

if __name__ == "__main__":
    main()
