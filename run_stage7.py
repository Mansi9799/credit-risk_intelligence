"""
Credit Risk Intelligence System - Stage 7 Loss Modeling Execution Script
========================================================================
Executes Loss Given Default (LGD), Exposure at Default (EAD), and Expected
Loss (EL = PD x LGD x EAD) calculation and portfolio capital provisioning.

Pipeline Steps:
1. Ingests Master Feature Dataset (307,511 loans).
2. Generates calibrated default probabilities (PD) via LightGBM Classifier.
3. Fits and validates LGD Regressor (LGD in [0.05, 0.95]).
4. Computes Exposure at Default (EAD) with term amortization & revolving CCF.
5. Computes loan-level Expected Loss (EL) and portfolio aggregation.
6. Generates analytical visual artifacts and summary tables in loss_outputs/.
7. Persists models and enriched expected loss dataset.

Author: Portfolio Project
Stage: 7 - Loss Modeling & Expected Loss
"""

import os
import sys
import time
import json
import logging
import joblib
import pandas as pd
import numpy as np

# Safe standard output handling
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add src to path
sys.path.insert(0, os.path.abspath("src"))

from src.loss_modeling import LGDModel, EADModel, ExpectedLossCalculator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("Stage7Pipeline")


def run_stage7_loss_modeling():
    """
    Executes the end-to-end Stage 7 Loss Modeling & Expected Loss pipeline.
    """
    total_start = time.time()
    print("\n" + "=" * 90)
    print("=== CREDIT RISK INTELLIGENCE SYSTEM -- STAGE 7: LOSS MODELING & EXPECTED LOSS ===")
    print("=" * 90 + "\n")

    # -------------------------------------------------------------------------
    # 1. LOAD MASTER FEATURE DATASET
    # -------------------------------------------------------------------------
    features_path = "processed_data/application_features.parquet"
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature dataset not found at {features_path}. Run previous phases first.")

    logger.info(f"Loading master feature dataset from {features_path}...")
    df_features = pd.read_parquet(features_path)
    logger.info(f"Loaded {len(df_features):,} loan records with {df_features.shape[1]} columns.")

    # -------------------------------------------------------------------------
    # 2. GENERATE CALIBRATED DEFAULT PROBABILITIES (PD)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(">>> STEP 1: CALIBRATED PROBABILITY OF DEFAULT (PD) ESTIMATION")
    print("=" * 90)

    lgb_model_path = "models/lightgbm_model.joblib"
    encoders_path = "models/tree_label_encoders.joblib"
    metadata_path = "models/model_metadata.json"

    if os.path.exists(lgb_model_path) and os.path.exists(encoders_path) and os.path.exists(metadata_path):
        logger.info("Loading pre-trained LightGBM classification engine...")
        lgb_model = joblib.load(lgb_model_path)
        label_encoders = joblib.load(encoders_path)
        with open(metadata_path, "r") as f:
            meta = json.load(f)

        feature_cols = meta["feature_names"]
        X_tree = df_features[feature_cols].copy()
        
        # Apply categorical label encoders
        for col, encoder in label_encoders.items():
            if col in X_tree.columns:
                X_tree[col] = encoder.transform(X_tree[col].astype(str))

        logger.info("Predicting calibrated Probability of Default (PD) for all loans...")
        pd_scores = lgb_model.predict_proba(X_tree)[:, 1]
    else:
        # Fallback to empirical target-smoothed baseline PD if model artifact missing
        logger.warning("Pre-trained LightGBM model not found. Using baseline PD proxy.")
        ext_mean = df_features["EXT_SOURCE_MEAN"].fillna(df_features["EXT_SOURCE_MEAN"].median()).values
        pd_scores = np.clip(1.0 / (1.0 + np.exp(4.0 * (ext_mean - 0.5))), 0.01, 0.95)

    logger.info(
        f"PD calculation complete | Mean PD: {pd_scores.mean()*100:.2f}% | "
        f"Min PD: {pd_scores.min()*100:.2f}% | Max PD: {pd_scores.max()*100:.2f}%"
    )

    # -------------------------------------------------------------------------
    # 3. FIT & EVALUATE LGD (LOSS GIVEN DEFAULT) MODEL
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(">>> STEP 2: LGD (LOSS GIVEN DEFAULT) MODELING")
    print("=" * 90)

    lgd_engine = LGDModel(min_lgd=0.05, max_lgd=0.95, random_state=42)
    lgd_engine.fit(df_features)
    
    # Predict LGD across portfolio
    lgd_scores = lgd_engine.predict(df_features)
    
    # Save model artifact
    lgd_engine.save("models/lgd_model.joblib")

    # Feature Importance
    lgd_imp = lgd_engine.get_feature_importance()
    lgd_imp.to_csv("loss_outputs/lgd_feature_importance.csv", index=False)

    print("\n[TOP 8 LGD RISK DRIVERS & IMPORTANCE]:")
    print("-" * 55)
    print(f"{'Feature Name':<35} | {'Importance':<15}")
    print("-" * 55)
    for _, row in lgd_imp.head(8).iterrows():
        print(f"{row['feature']:<35} | {row['importance']:<15.4f}")
    print("-" * 55)

    # -------------------------------------------------------------------------
    # 4. COMPUTE EXPOSURE AT DEFAULT (EAD)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(">>> STEP 3: EXPOSURE AT DEFAULT (EAD) MODELING")
    print("=" * 90)

    ead_engine = EADModel(
        revolving_ccf=0.75,
        default_amortization_factor=0.70,
        accrued_interest_margin=0.03
    )

    # Load survival durations if available from Phase 3 survival dataset
    surv_path = "processed_data/survival_dataset.parquet"
    surv_durations = None
    if os.path.exists(surv_path):
        df_surv = pd.read_parquet(surv_path)
        if "duration" in df_surv.columns and len(df_surv) == len(df_features):
            surv_durations = df_surv["duration"].values
            logger.info("Incorporated Phase 3 survival event durations for dynamic EAD amortization.")

    ead_amounts, ead_factors = ead_engine.calculate_ead(df_features, survival_durations=surv_durations)

    # -------------------------------------------------------------------------
    # 5. EXPECTED LOSS (EL = PD x LGD x EAD) & PORTFOLIO CAPITAL PROVISIONING
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(">>> STEP 4: EXPECTED LOSS (EL = PD x LGD x EAD) & PORTFOLIO CAPITAL PROVISIONING")
    print("=" * 90)

    el_calc = ExpectedLossCalculator(output_dir="loss_outputs")
    loss_df = el_calc.calculate_expected_loss(
        df=df_features,
        pd_scores=pd_scores,
        lgd_scores=lgd_scores,
        ead_amounts=ead_amounts
    )

    # Persist full enriched dataset
    loss_parquet_path = "processed_data/expected_loss_dataset.parquet"
    loss_df.to_parquet(loss_parquet_path)
    logger.info(f"Saved master Expected Loss dataset to {loss_parquet_path}")

    # Portfolio Summary Metrics
    summary = el_calc.summarize_portfolio(loss_df)

    print("\n[INSTITUTIONAL PORTFOLIO EXPECTED LOSS & CAPITAL PROVISIONING SUMMARY]:")
    print("=" * 80)
    print(f"  * Total Active Loans Evaluated  : {summary['total_loans']:,}")
    print(f"  * Total Original Credit Limit   : ${summary['total_original_credit_limit']:,.2f}")
    print(f"  * Total Portfolio EAD           : ${summary['total_portfolio_ead']:,.2f}")
    print(f"  * Total Expected Loss (EL)      : ${summary['total_portfolio_expected_loss']:,.2f}")
    print(f"  * Portfolio Expected Loss Rate  : {summary['portfolio_el_rate_pct']:.2f}%")
    print(f"  * Portfolio Mean PD             : {summary['mean_pd']*100:.2f}%")
    print(f"  * Portfolio Mean LGD            : {summary['mean_lgd']*100:.2f}%")
    print(f"  * Mean EAD per Loan             : ${summary['mean_ead']:,.2f}")
    print(f"  * Mean Expected Loss per Loan   : ${summary['mean_el_per_loan']:,.2f}")
    print("=" * 80)

    print("\n[RISK TIER STRATIFICATION & LOSS CONCENTRATION]:")
    print("-" * 80)
    print(f"{'Risk Tier':<12} | {'Loan Count':<12} | {'Total EAD ($M)':<15} | {'Total EL ($M)':<15} | {'EL Share %':<12} | {'Mean PD %':<10}")
    print("-" * 80)
    for row in summary["tier_breakdown"]:
        print(
            f"{row['RISK_TIER']:<12} | {row['Loan_Count']:<12,d} | "
            f"${row['Total_EAD']/1e6:<14.2f}M | ${row['Total_EL']/1e6:<14.2f}M | "
            f"{row['EL_Share_Pct']:<11.2f}% | {row['Mean_PD']*100:<9.2f}%"
        )
    print("-" * 80)

    # -------------------------------------------------------------------------
    # 6. VISUALIZATIONS & ARTIFACTS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(">>> STEP 5: VISUAL ARTIFACT GENERATION")
    print("=" * 90)

    plots = el_calc.generate_loss_visualizations(loss_df)
    for p in plots:
        print(f"  * Saved figure: {p}")

    total_time = time.time() - total_start
    print("\n" + "=" * 90)
    print(f"=== STAGE 7 LOSS MODELING COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS! ===")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    run_stage7_loss_modeling()
