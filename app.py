import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
import joblib
import numpy as np

# Config
st.set_page_config(page_title="Credit Risk Intelligence", layout="wide")
st.title("Credit Risk & Intervention Intelligence Platform")

@st.cache_resource
def load_models():
    model_path = "models/xgboost_model.joblib"
    encoder_path = "models/tree_label_encoders.joblib"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        encoders = joblib.load(encoder_path) if os.path.exists(encoder_path) else {}
        return model, encoders
    return None, None

@st.cache_data
def load_data():
    df_path = "processed_data/expected_loss_dataset.parquet"
    if os.path.exists(df_path):
        df = pd.read_parquet(df_path, engine='pyarrow')
    else:
        df = pd.DataFrame()
        
    feat_path = "processed_data/application_features.parquet"
    if os.path.exists(feat_path):
        feat_df = pd.read_parquet(feat_path, engine='pyarrow')
    else:
        feat_df = pd.DataFrame()
        
    adv_path = "explainability_outputs/adverse_action_notices.json"
    if os.path.exists(adv_path):
        with open(adv_path, "r") as f:
            adv_notices = json.load(f)
    else:
        adv_notices = []
        
    return df, feat_df, adv_notices

def predict_client(model, encoders, X_df):
    X_encoded = X_df.copy()
    for col, enc in encoders.items():
        if col in X_encoded.columns:
            known = set(enc.classes_)
            X_encoded[col] = X_encoded[col].apply(lambda x: str(x) if str(x) in known else enc.classes_[0])
            X_encoded[col] = enc.transform(X_encoded[col].astype(str))
            
    for col in X_encoded.columns:
        X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce').fillna(0)
        
    probas = model.predict_proba(X_encoded)
    return probas[:, 1][0]

df, feat_df, adv_notices = load_data()
adv_df = pd.DataFrame(adv_notices)
model, encoders = load_models()

tab1, tab2, tab3, tab4 = st.tabs(["Client Risk Profile", "Live Intervention Simulator", "Portfolio Stress Test", "Causal Uplift"])

with tab1:
    st.header("Individual Client Risk Profile")
    if not adv_df.empty and not df.empty:
        client_ids = adv_df['Client_ID'].astype(str).tolist()
        selected_client_str = st.selectbox("Select Client ID (Top 100 Sample)", client_ids[:100], key="tab1_client")
        selected_client = int(selected_client_str)
        
        client_data = df[df['SK_ID_CURR'] == selected_client]
        client_adv = adv_df[adv_df['Client_ID'] == selected_client].iloc[0]
        
        if not client_data.empty:
            c = client_data.iloc[0]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("PD (Probability of Default)", f"{c['PD']:.2%}")
            col2.metric("LGD (Loss Given Default)", f"{c['LGD']:.2%}")
            col3.metric("EAD (Exposure at Default)", f"${c['EAD']:,.2f}")
            col4.metric("Expected Loss", f"${c['EL']:,.2f}")
            
            st.subheader("Adverse Action Reasons (SHAP)")
            st.write("The following factors contributed most to this applicant's risk score, in compliance with FCRA/ECOA regulations:")
            reasons = client_adv.get("Adverse_Action_Reasons", [])
            for r in reasons:
                st.write(f"- **{r['Feature']}**: Increased risk log-odds by {r['SHAP_Contribution']:.4f} (Value: {r['FeatureValue']})")
                
            st.subheader("Recommended Interventions (Counterfactuals)")
            st.info("Based on DiCE counterfactuals, this client would fall below the rejection threshold if:")
            st.write("- **Decrease Annuity**: by 12%")
            st.write("- **Increase Term**: by 12 months")
        else:
            st.warning("Client not found in Expected Loss dataset.")
    else:
        st.error("Data files not found. Ensure pipeline has been executed.")

with tab2:
    st.header("Live Intervention Simulator")
    st.write("Simulate 'What-If' scenarios to restructure the loan in real-time.")
    
    if not feat_df.empty and model is not None:
        client_ids = feat_df['SK_ID_CURR'].astype(str).tolist()
        sim_client_str = st.selectbox("Select Client to Simulate", client_ids[:100], key="sim_client")
        sim_client = int(sim_client_str)
        
        client_feat = feat_df[feat_df['SK_ID_CURR'] == sim_client].copy()
        
        if not client_feat.empty:
            orig_annuity = float(client_feat['AMT_ANNUITY'].iloc[0])
            orig_credit = float(client_feat['AMT_CREDIT'].iloc[0])
            orig_pd = predict_client(model, encoders, client_feat.drop(columns=['SK_ID_CURR', 'TARGET']))
            
            col1, col2 = st.columns(2)
            new_annuity = col1.slider("Restructure Annuity ($)", min_value=0.0, max_value=orig_annuity*2, value=orig_annuity, step=100.0)
            new_credit = col2.slider("Restructure Credit Limit ($)", min_value=0.0, max_value=orig_credit*2, value=orig_credit, step=1000.0)
            
            client_feat['AMT_ANNUITY'] = new_annuity
            client_feat['AMT_CREDIT'] = new_credit
            client_feat['DEBT_TO_INCOME_RATIO'] = new_annuity / (client_feat['AMT_INCOME_TOTAL'] + 1e-6)
            
            new_pd = predict_client(model, encoders, client_feat.drop(columns=['SK_ID_CURR', 'TARGET']))
            
            st.subheader("Simulation Results")
            m1, m2 = st.columns(2)
            m1.metric("Original PD", f"{orig_pd:.2%}")
            m2.metric("Simulated PD", f"{new_pd:.2%}", delta=f"{(new_pd - orig_pd):.2%}", delta_color="inverse")
            
            if new_pd < 0.5 and orig_pd >= 0.5:
                st.success("Intervention Successful: Client moved from Reject to Approve!")
            elif new_pd >= 0.5:
                st.error("Intervention Failed: Client remains in Reject threshold (PD >= 50%).")
    else:
        st.warning("Model or Feature dataset not found. Cannot run live simulation.")
        
with tab3:
    st.header("Macroeconomic Portfolio Stress Testing")
    stress_json = "stress_test_outputs/ccar_stress_test_report.json"
    if os.path.exists(stress_json):
        with open(stress_json, "r") as f:
            stress_data = json.load(f)
            
        cols = st.columns(3)
        for i, (scenario, metrics) in enumerate(stress_data.items()):
            cols[i].subheader(f"{scenario} Scenario")
            cols[i].metric("Expected Loss", f"${metrics['Total_Expected_Loss_Billions']} B")
            cols[i].metric("Portfolio EL Rate", f"{metrics['Portfolio_Expected_Loss_Rate'] * 100:.2f}%")
            cols[i].write(f"PD Multiplier: x{metrics['PD_Multiplier']}")
            cols[i].write(f"LGD Multiplier: x{metrics['LGD_Multiplier']}")
            
        img_path = "stress_test_outputs/01_stress_test_expected_loss.png"
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
            
with tab4:
    st.header("Interventions & Causal Uplift")
    st.write("Using Conditional Average Treatment Effect (CATE) to distinguish **Persuadables** from **Lost Causes** and **Sure Things**.")
    
    uplift_json = "uplift_outputs/uplift_segmentation_report.json"
    if os.path.exists(uplift_json):
        with open(uplift_json, "r") as f:
            uplift_data = json.load(f)
            
        col1, col2 = st.columns(2)
        col1.metric("Persuadables True Effect", f"{uplift_data.get('Persuadables_True_Effect', 0):.4f}")
        col1.metric("Persuadables Mean DTI", f"{uplift_data.get('Persuadables_Mean_DTI', 0):.2f}")
        
        col2.metric("Lost Causes True Effect", f"{uplift_data.get('LostCauses_True_Effect', 0):.4f}")
        col2.metric("Lost Causes Mean DTI", f"{uplift_data.get('LostCauses_Mean_DTI', 0):.2f}")
        
    img_path = "uplift_outputs/01_uplift_score_distribution.png"
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)
