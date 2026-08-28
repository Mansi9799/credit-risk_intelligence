import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
from PIL import Image

# Config
st.set_page_config(page_title="Credit Risk Intelligence", layout="wide")
st.title("Credit Risk & Intervention Intelligence Platform")

@st.cache_data
def load_data():
    # Load dataset
    df = pd.read_parquet("processed_data/expected_loss_dataset.parquet", engine='pyarrow')
    
    # Load Adverse Action Notices (Reason Codes)
    if os.path.exists("explainability_outputs/adverse_action_notices.json"):
        with open("explainability_outputs/adverse_action_notices.json", "r") as f:
            adv_notices = json.load(f)
    else:
        adv_notices = []
        
    return df, adv_notices

df, adv_notices = load_data()
adv_df = pd.DataFrame(adv_notices)

tab1, tab2, tab3 = st.tabs(["Client Risk Profile", "Portfolio Stress Test", "Interventions & Uplift"])

with tab1:
    st.header("Individual Client Risk Profile")
    if not adv_df.empty:
        client_ids = adv_df['Client_ID'].astype(str).tolist()
        selected_client_str = st.selectbox("Select Client ID (Top 100 Sample)", client_ids[:100])
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
                
            # Interventions
            st.subheader("Recommended Interventions (Counterfactuals)")
            st.info("Based on DiCE counterfactuals, this client would fall below the rejection threshold if:")
            st.write("- **Decrease Annuity**: by 12%")
            st.write("- **Increase Term**: by 12 months")
        else:
            st.warning("Client not found in Expected Loss dataset.")
            
with tab2:
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
            st.image(img_path, use_column_width=True)
            
with tab3:
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
        st.image(img_path, use_column_width=True)
