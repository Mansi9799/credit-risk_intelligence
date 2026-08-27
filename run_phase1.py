"""
Credit Risk Intelligence System - Phase 1 Execution Script
==========================================================
Executes the full Phase 1 Data Foundation pipeline:
- STAGE 1: Data Loading, Dtype Optimization, and Missing Value Summaries
- STAGE 2: Exploratory Data Analysis (Distributions, Cohort Default Rates, Repayment Correlations)
- STAGE 3: Institutional Data Cleaning & Anomaly Remediation with Interview Rationales

Author: Portfolio Project
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np

# Configure safe standard output encoding for cross-platform compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure src module is in path
sys.path.insert(0, os.path.abspath("src"))

from src.data_loader import DataLoader
from src.eda import CreditRiskEDA
from src.data_cleaner import DataCleaner, CLEANING_RATIONALE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("Phase1Pipeline")


def run_phase1_pipeline(sample_installments_nrows: int = 1_500_000):
    """
    Executes the entire Phase 1 pipeline across Stage 1, Stage 2, and Stage 3.

    Parameters
    ----------
    sample_installments_nrows : int
        Number of installment transactions to load for repayment behavior analysis.
        Loading ~1.5 million rows provides statistical power while running in seconds.
    """
    start_total_time = time.time()

    print("\n" + "=" * 85)
    print("=== CREDIT RISK INTELLIGENCE SYSTEM -- PHASE 1: DATA FOUNDATION PIPELINE ===")
    print("=" * 85 + "\n")

    # =========================================================================
    # STAGE 1: ENVIRONMENT & DATA LOADING
    # =========================================================================
    print(">>> STAGE 1: ENVIRONMENT SETUP & DATA LOADING")
    print("-" * 85)

    loader = DataLoader(data_dir="data")

    # 1. Load Application Dataset (Full 307,511 rows)
    t0 = time.time()
    app_df = loader.load_application_train(nrows=None, optimize_memory=True)
    logger.info(f"Loaded full application_train.csv in {time.time() - t0:.2f}s")

    # 2. Structural Inspection & Summary
    app_summary = DataLoader.inspect_dataset(app_df, name="Home Credit Application Train")

    # 3. Missing Value Summary (Top 25 Columns)
    missing_table = DataLoader.get_missing_summary(app_df, top_n=25)
    print("\n[TOP 25 MISSING VALUE COLUMNS (application_train)]:")
    print("-" * 85)
    print(f"{'Feature Name':<35} | {'Missing Count':<15} | {'Missing %':<12} | {'Dtype':<10}")
    print("-" * 85)
    for idx, row in missing_table.iterrows():
        print(f"{idx:<35} | {int(row['Missing_Count']):<15,d} | {row['Missing_Percentage']:<11.2f}% | {str(row['Dtype']):<10}")
    print("-" * 85)

    # 4. Load Installments Dataset for Repayment Profiling
    t0 = time.time()
    logger.info(f"Loading installment payment transactions (sample={sample_installments_nrows:,} rows)...")
    inst_df = loader.load_installments(nrows=sample_installments_nrows, optimize_memory=True)
    logger.info(f"Loaded installments dataset in {time.time() - t0:.2f}s")
    DataLoader.inspect_dataset(inst_df, name="Installment Payments Transactions")

    # =========================================================================
    # STAGE 2: EXPLORATORY DATA ANALYSIS (EDA)
    # =========================================================================
    print("\n" + "=" * 85)
    print(">>> STAGE 2: EXPLORATORY DATA ANALYSIS & RISK STRATIFICATION")
    print("=" * 85)

    eda = CreditRiskEDA(output_dir="eda_outputs")

    # 1. Target & Class Imbalance Analysis
    target_stats = eda.plot_target_distribution(app_df)
    print(f"\n[TARGET CLASS DISTRIBUTION]:")
    print(f"  * Non-Default (TARGET=0) : {target_stats['non_default_count']:,} ({100 - target_stats['default_rate_pct']:.2f}%)")
    print(f"  * Default     (TARGET=1) : {target_stats['default_count']:,} ({target_stats['default_rate_pct']:.2f}%)")
    print(f"  * Imbalance Ratio        : 1 : {(target_stats['non_default_count'] / target_stats['default_count']):.2f}")

    # 2. Continuous Financial Distributions (Credit Limit, Income, Annuity, Age)
    eda.plot_continuous_distributions(app_df)

    # 3. Segment-Level Default Rates (Age Groups, Credit Bands, Education, Contract Type)
    segment_metrics = eda.plot_segment_default_rates(app_df)

    # 4. External Credit Agency Scores (EXT_SOURCE_1, 2, 3)
    eda.plot_external_credit_scores(app_df)

    # 5. Granular Repayment Behavior Analysis & Target Correlation
    client_repay, merged_repay = eda.analyze_repayment_behavior(app_df, inst_df)

    print("\n[SUCCESS] All EDA visualizations and analytical tables saved to 'eda_outputs/'.")

    # =========================================================================
    # STAGE 3: DATA CLEANING & ANOMALY REMEDIATION
    # =========================================================================
    print("\n" + "=" * 85)
    print(">>> STAGE 3: DATA CLEANING & ANOMALY REMEDIATION (INTERVIEW-READY)")
    print("=" * 85)

    cleaner = DataCleaner(processed_dir="processed_data")

    # 1. Clean Application Dataset
    clean_app, app_clean_report = cleaner.clean_application_data(app_df)

    # 2. Clean Installment Payments Dataset
    clean_inst, inst_clean_report = cleaner.clean_installments_data(inst_df)

    # 3. Export Cleaned Datasets
    cleaner.export_cleaned_data(clean_app, clean_inst)

    # Print Cleaned Transformation Summary
    print("\n[CLEANING DECISIONS & JUSTIFICATIONS]:")
    print("=" * 85)
    for transform in app_clean_report["transformations"]:
        print(f"\n* [TRANSFORMATION] {transform['step']}")
        print(f"  Rationale & Interview Defense:")
        print(f"  \"{transform['rationale']}\"")
    print("=" * 85)

    print(f"\n- Initial Application Missing Cells: {app_clean_report['initial_missing_cells']:,}")
    print(f"- Final Application Missing Cells  : {app_clean_report['final_missing_cells']:,}")
    print(f"- Cleaned Application Shape        : {clean_app.shape[0]:,} rows x {clean_app.shape[1]} columns")

    total_time = time.time() - start_total_time
    print("\n" + "=" * 85)
    print(f"=== PHASE 1 COMPLETED SUCCESSFULLY in {total_time:.2f} seconds! ===")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    run_phase1_pipeline()
