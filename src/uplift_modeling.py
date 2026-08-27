import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from econml.metalearners import TLearner

# Configuration
DATA_PATH = r"C:\Users\MY PC\OneDrive\ドキュメント\Rainmeter\Desktop\credit-risk-intelligence\processed_data\application_features.parquet"
OUTPUT_DIR = r"C:\Users\MY PC\OneDrive\ドキュメント\Rainmeter\Desktop\credit-risk-intelligence\uplift_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_intervention_data(df, sample_size=50000):
    """
    Simulates a randomized control trial (RCT) where a subset of customers are offered 
    a "Credit Counseling & Loan Restructuring" intervention.
    """
    print(f"Simulating Intervention Data for {sample_size} customers...")
    df_sample = df.sample(n=sample_size, random_state=42).copy()
    
    # 1. Random Treatment Assignment (50% treated, 50% control)
    np.random.seed(42)
    df_sample['TREATMENT'] = np.random.binomial(1, 0.5, size=len(df_sample))
    
    # 2. Base Probability of Default (using our existing TARGET as a proxy for baseline risk)
    # We will synthetically adjust the TARGET based on treatment and customer profile
    
    # Identify profile characteristics
    # People who benefit most: Moderate Debt-to-Income, lower age, moderate DPD (Not perfect, but not terrible)
    # Lost Causes: Very high DPD, extremely high DTI
    # Sure Things: Very low DTI, excellent external scores
    
    # Fill NAs for simulation logic
    dti = df_sample['DEBT_TO_INCOME_RATIO'].fillna(df_sample['DEBT_TO_INCOME_RATIO'].median())
    dpd = df_sample['INST_AVG_DPD'].fillna(df_sample['INST_AVG_DPD'].median())
    ext_mean = df_sample['EXT_SOURCE_MEAN'].fillna(df_sample['EXT_SOURCE_MEAN'].median())
    
    # Calculate True Individual Treatment Effect (ITE)
    # Negative ITE means the intervention *reduces* the probability of default (which is good)
    true_ite = np.zeros(len(df_sample))
    
    for i in range(len(df_sample)):
        # Persuadables (Moderate Risk) get a big reduction in default risk (-15% probability)
        if (0.3 < dti.iloc[i] < 0.6) and (dpd.iloc[i] < 30) and (0.3 < ext_mean.iloc[i] < 0.6):
            true_ite[i] = -0.15 
        # Lost Causes (Extreme Risk) barely respond (-1% probability)
        elif (dti.iloc[i] > 0.6) or (dpd.iloc[i] > 60):
            true_ite[i] = -0.01
        # Sure Things (Low Risk) don't need it (-1% probability)
        else:
            true_ite[i] = -0.01
            
    # Calculate baseline probability (approximated for simulation)
    baseline_prob = df_sample['TARGET'].values * 0.8 + np.random.uniform(0, 0.2, size=len(df_sample))
    baseline_prob = np.clip(baseline_prob, 0.05, 0.95)
    
    # Calculate final probability given treatment
    final_prob = baseline_prob + (df_sample['TREATMENT'].values * true_ite)
    final_prob = np.clip(final_prob, 0, 1)
    
    # Realize the post-intervention outcome based on the probability
    df_sample['OUTCOME_DEFAULT'] = np.random.binomial(1, final_prob)
    df_sample['TRUE_ITE'] = true_ite
    
    return df_sample

def train_uplift_model(df):
    """
    Trains a T-Learner (Two-Model approach) to predict the Conditional Average Treatment Effect (CATE).
    
    Why this is different from just "Who is risky?":
    - A standard risk model predicts P(Default). It targets high-risk people.
    - An uplift model predicts P(Default | Control) - P(Default | Treatment). 
      It targets "Persuadables" — people whose risk *decreases the most* when given the intervention.
      If a customer is extremely high risk but WILL NOT respond to intervention (Lost Cause), 
      the uplift model will score them low, saving intervention budget!
    """
    print("Training Uplift Model (T-Learner)...")
    
    # Select numeric features for simplicity in this demo
    features = [c for c in df.columns if df[c].dtype in ['int64', 'float64', 'int32', 'float32'] 
                and c not in ['SK_ID_CURR', 'TARGET', 'TREATMENT', 'OUTCOME_DEFAULT', 'TRUE_ITE']]
                
    # Fill NA for EconML (LightGBM handles NAs natively but EconML wrappers sometimes prefer clean data)
    X = df[features].fillna(df[features].median())
    T = df['TREATMENT']
    Y = df['OUTCOME_DEFAULT']
    
    X_train, X_test, T_train, T_test, Y_train, Y_test, ite_train, ite_test = train_test_split(
        X, T, Y, df['TRUE_ITE'], test_size=0.2, random_state=42
    )
    
    # We use LightGBM classifiers as our base estimators for the T-Learner
    models = lgb.LGBMClassifier(n_estimators=100, max_depth=5, random_state=42)
    
    # Initialize T-Learner
    # T-Learner fits one model on control group, one on treatment group.
    # CATE = Model_Treatment(X) - Model_Control(X)
    est = TLearner(models=models)
    est.fit(Y_train, T_train, X=X_train)
    
    print("Predicting CATE (Uplift) on test set...")
    # Predict Conditional Average Treatment Effect (CATE)
    # A negative CATE means the intervention *reduced* default probability (Desirable)
    cate_pred = est.effect(X_test)
    
    return est, X_test, cate_pred, ite_test

def evaluate_uplift(X_test, cate_pred, ite_test):
    print("Evaluating Uplift Model...")
    
    # Create evaluation dataframe
    eval_df = X_test.copy()
    eval_df['Predicted_CATE'] = cate_pred
    eval_df['True_CATE'] = ite_test
    
    # We want to target people with the MOST NEGATIVE CATE (highest risk reduction)
    eval_df['Uplift_Score'] = -eval_df['Predicted_CATE'] # Positive score = Higher priority for intervention
    
    # Sort by Uplift Score to identify top candidates
    eval_df = eval_df.sort_values(by='Uplift_Score', ascending=False)
    
    # Segment into Tiers
    top_20_pct = int(len(eval_df) * 0.2)
    bottom_20_pct = int(len(eval_df) * 0.8)
    
    print("\n--- Uplift Segmentation Insights ---")
    
    print("\n[Segment 1: The 'Persuadables' (Top 20% Uplift Score)]")
    print("These customers have moderate risk, but the intervention dramatically reduces their default probability.")
    persuadables = eval_df.iloc[:top_20_pct]
    print(f"Average True Default Reduction: {abs(persuadables['True_CATE'].mean() * 100):.2f}%")
    print(f"Mean Debt-to-Income: {persuadables['DEBT_TO_INCOME_RATIO'].mean():.2f}")
    
    print("\n[Segment 2: The 'Lost Causes' / 'Sure Things' (Bottom 20% Uplift Score)]")
    print("Intervening here is a waste of budget. They either default anyway (Lost Causes) or won't default anyway (Sure Things).")
    lost_causes = eval_df.iloc[bottom_20_pct:]
    print(f"Average True Default Reduction: {abs(lost_causes['True_CATE'].mean() * 100):.2f}%")
    print(f"Mean Debt-to-Income: {lost_causes['DEBT_TO_INCOME_RATIO'].mean():.2f}")
    
    # Save Report
    report = {
        "Persuadables_True_Effect": float(persuadables['True_CATE'].mean()),
        "Persuadables_Mean_DTI": float(persuadables['DEBT_TO_INCOME_RATIO'].mean()),
        "LostCauses_True_Effect": float(lost_causes['True_CATE'].mean()),
        "LostCauses_Mean_DTI": float(lost_causes['DEBT_TO_INCOME_RATIO'].mean())
    }
    with open(os.path.join(OUTPUT_DIR, "uplift_segmentation_report.json"), "w") as f:
        json.dump(report, f, indent=4)
        
    # Plot Uplift Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(eval_df['Uplift_Score'] * 100, bins=50, color='blue', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title("Distribution of Predicted Uplift Scores (Default Risk Reduction %)")
    plt.xlabel("Predicted Decrease in Default Probability (%)")
    plt.ylabel("Number of Customers")
    plt.savefig(os.path.join(OUTPUT_DIR, "01_uplift_score_distribution.png"), dpi=300)
    plt.close()
    
    print("\nStage 9: Uplift Modeling complete.")

if __name__ == "__main__":
    print("Loading base dataset...")
    df = pd.read_parquet(DATA_PATH, engine='pyarrow')
    df_simulated = simulate_intervention_data(df, sample_size=50000)
    est, X_test, cate_pred, ite_test = train_uplift_model(df_simulated)
    evaluate_uplift(X_test, cate_pred, ite_test)
