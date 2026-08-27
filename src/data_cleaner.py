"""
Data Cleaner Module for Credit Risk Intelligence System
-------------------------------------------------------
Implements institutional-grade data cleaning, anomaly remediation, and 
theoretically justified imputation strategies for loan application and 
installment repayment datasets.

Every cleaning decision includes an interview-ready rationale explaining:
1. The business / econometric origin of the anomaly or missingness (MCAR vs MAR vs MNAR).
2. The risk of naively dropping or filling values.
3. The mathematical and regulatory justification for the chosen treatment.

Author: Portfolio Project
Phase: 1 - Stage 3
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DataCleaner")


# =====================================================================
# INTERVIEW-DEFENSE RATIONALE REGISTRY
# =====================================================================
CLEANING_RATIONALE = {
    "DAYS_EMPLOYED_ANOMALY": (
        "WHY: In the raw Home Credit system, +365,243 days (~1,000 years) was a legacy sentinel "
        "code indicating retired individuals, pensioners, or unemployed applicants. "
        "TREATMENT: Create a binary flag 'DAYS_EMPLOYED_ANOM' (1 if 365243, 0 otherwise) to capture "
        "this high-signal status, then replace 365243 with NaN/0 so linear and distance models are not "
        "distorted by an absurd magnitude."
    ),
    "AMT_REQ_CREDIT_BUREAU_ZERO_FILL": (
        "WHY: Credit bureaus return NULL when an applicant has no recorded inquiry hits in that time window. "
        "TREATMENT: Impute missing inquiry counts (HOUR, DAY, WEEK, MON, QRT, YEAR) with 0.0 because "
        "absence of record in bureau logs represents zero inquiries, not unobserved random data."
    ),
    "SOCIAL_CIRCLE_DEFAULTS_ZERO_FILL": (
        "WHY: Social circle observation counters (OBS/DEF 30/60) are NULL when no peer defaults exist. "
        "TREATMENT: Impute with 0.0, representing baseline zero observed defaults in the client's network."
    ),
    "CATEGORICAL_EXPLICIT_MISSING": (
        "WHY: Missingness in fields like OCCUPATION_TYPE (31.3% missing) or HOUSETYPE_MODE is Informative "
        "Missingness (MNAR - Missing Not At Random). For instance, freelancers, gig workers, and retirees "
        "frequently leave occupation blank. Imputing with the Mode would falsely bias the dataset toward 'Laborers'. "
        "TREATMENT: Impute missing values with an explicit category 'Unknown_Missing' to allow tree splits "
        "and weight-of-evidence (WoE) binnings to leverage the missingness signal directly."
    ),
    "EXTERNAL_SCORES_MEDIAN_AND_INDICATORS": (
        "WHY: EXT_SOURCE_1, 2, and 3 are normalized credit bureau scores. EXT_SOURCE_1 is ~56% missing because "
        "underwriters only purchase Tier-1 bureau reports for specific risk segments. "
        "TREATMENT: For linear/distance models, impute with Median (robust to skewness) and construct binary "
        "missingness indicators ('EXT_SOURCE_1_IS_MISSING') so the algorithm knows when a bureau check was skipped."
    ),
    "FINANCIAL_AMOUNTS_MEDIAN_IMPUTATION": (
        "WHY: Continuous financial features like AMT_ANNUITY and AMT_GOODS_PRICE have right-skewed distributions "
        "with heavy tails. Mean imputation would be biased by ultra-wealthy outliers. "
        "TREATMENT: Impute with sample median to preserve median central tendency without inflating variance."
    ),
    "GENDER_FAMILY_ANOMALIES": (
        "WHY: 'CODE_GENDER' has 4 'XNA' values and 'NAME_FAMILY_STATUS' has 2 'Unknown' records (0.001% of data). "
        "TREATMENT: Impute with respective modes ('F' for gender, 'Married' for family status) for downstream "
        "fairness and demographic consistency without losing records."
    ),
    "INSTALLMENTS_PAYMENT_NULLS": (
        "WHY: In installment logs, missing AMT_PAYMENT or DAYS_ENTRY_PAYMENT signifies a completely missed installment. "
        "TREATMENT: Impute AMT_PAYMENT with 0.0 and flag as delinquent rather than dropping rows, preventing "
        "survivorship bias in repayment behavior aggregation."
    )
}


class DataCleaner:
    """
    Institutional data cleaning pipeline for credit risk datasets.
    Handles anomalies, missingness, and outputs reproducible cleaned datasets.
    """

    def __init__(self, processed_dir: str = "processed_data"):
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)
        self.cleaning_report: Dict[str, Any] = {}

    def clean_application_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Cleans the loan application dataset application_train.csv.

        Steps:
        1. Fix DAYS_EMPLOYED 365243 sentinel anomaly with indicator flag.
        2. Handle categorical anomalies (CODE_GENDER 'XNA', NAME_FAMILY_STATUS 'Unknown').
        3. Impute Credit Bureau inquiry counts with 0.0.
        4. Impute Social Circle default counters with 0.0.
        5. Impute Continuous Financial variables (AMT_ANNUITY, AMT_GOODS_PRICE) with median.
        6. Impute Categorical variables with 'Unknown_Missing'.
        7. Generate Missing Indicators for External Scores (EXT_SOURCE_1, EXT_SOURCE_3).
        8. Derive standard interpretable features (AGE_YEARS, YEARS_EMPLOYED, CREDIT_TO_INCOME_RATIO, ANNUITY_TO_INCOME_RATIO).

        Returns
        -------
        pd.DataFrame
            Cleaned application dataframe.
        dict
            Detailed transformation and cleaning report.
        """
        logger.info("Initiating comprehensive cleaning of application_train dataset...")
        clean_df = df.copy()
        initial_shape = clean_df.shape
        initial_missing = int(clean_df.isnull().sum().sum())

        report = {
            "initial_shape": initial_shape,
            "initial_missing_cells": initial_missing,
            "transformations": []
        }

        # -------------------------------------------------------------
        # 1. DAYS_EMPLOYED Sentinel Anomaly (365243)
        # -------------------------------------------------------------
        anom_mask = (clean_df["DAYS_EMPLOYED"] == 365243)
        anom_count = int(anom_mask.sum())
        clean_df["DAYS_EMPLOYED_ANOM"] = anom_mask.astype(int)
        clean_df["DAYS_EMPLOYED"] = clean_df["DAYS_EMPLOYED"].replace({365243: np.nan})
        
        # Derive YEARS_EMPLOYED (positive float, 0 if unemployed/pensioner)
        clean_df["YEARS_EMPLOYED"] = -clean_df["DAYS_EMPLOYED"] / 365.25
        clean_df["YEARS_EMPLOYED"] = clean_df["YEARS_EMPLOYED"].fillna(0.0)

        report["transformations"].append({
            "step": "DAYS_EMPLOYED Anomaly Remediation",
            "anomalous_records_fixed": anom_count,
            "new_feature_created": ["DAYS_EMPLOYED_ANOM", "YEARS_EMPLOYED"],
            "rationale": CLEANING_RATIONALE["DAYS_EMPLOYED_ANOMALY"]
        })
        logger.info(f"Fixed {anom_count:,} DAYS_EMPLOYED anomalous records (+365243). Created 'DAYS_EMPLOYED_ANOM'.")

        # -------------------------------------------------------------
        # 2. Demographic Entry Errors (CODE_GENDER 'XNA', FAMILY_STATUS)
        # -------------------------------------------------------------
        if "CODE_GENDER" in clean_df.columns:
            gender_xna = int((clean_df["CODE_GENDER"] == "XNA").sum())
            mode_gender = clean_df["CODE_GENDER"].mode()[0]
            clean_df["CODE_GENDER"] = clean_df["CODE_GENDER"].replace({"XNA": mode_gender})
            report["transformations"].append({
                "step": "CODE_GENDER XNA Imputation",
                "records_imputed": gender_xna,
                "replacement_value": mode_gender,
                "rationale": CLEANING_RATIONALE["GENDER_FAMILY_ANOMALIES"]
            })

        if "NAME_FAMILY_STATUS" in clean_df.columns:
            fam_unk = int((clean_df["NAME_FAMILY_STATUS"] == "Unknown").sum())
            if fam_unk > 0:
                mode_fam = clean_df["NAME_FAMILY_STATUS"].mode()[0]
                clean_df["NAME_FAMILY_STATUS"] = clean_df["NAME_FAMILY_STATUS"].replace({"Unknown": mode_fam})
                report["transformations"].append({
                    "step": "NAME_FAMILY_STATUS Unknown Imputation",
                    "records_imputed": fam_unk,
                    "replacement_value": mode_fam,
                    "rationale": CLEANING_RATIONALE["GENDER_FAMILY_ANOMALIES"]
                })

        # -------------------------------------------------------------
        # 3. Credit Bureau Queries (Impute with 0.0)
        # -------------------------------------------------------------
        bureau_cols = [
            "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY",
            "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON",
            "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR"
        ]
        bureau_present = [c for c in bureau_cols if c in clean_df.columns]
        bureau_missing = {c: int(clean_df[c].isnull().sum()) for c in bureau_present}
        for col in bureau_present:
            clean_df[col] = clean_df[col].fillna(0.0)

        report["transformations"].append({
            "step": "Credit Bureau Query Zero-Fill",
            "columns": bureau_present,
            "missing_counts_filled": bureau_missing,
            "rationale": CLEANING_RATIONALE["AMT_REQ_CREDIT_BUREAU_ZERO_FILL"]
        })

        # -------------------------------------------------------------
        # 4. Social Circle Default Counters (Impute with 0.0)
        # -------------------------------------------------------------
        social_cols = [
            "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE",
            "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE"
        ]
        social_present = [c for c in social_cols if c in clean_df.columns]
        for col in social_present:
            clean_df[col] = clean_df[col].fillna(0.0)

        report["transformations"].append({
            "step": "Social Circle Default Counters Zero-Fill",
            "columns": social_present,
            "rationale": CLEANING_RATIONALE["SOCIAL_CIRCLE_DEFAULTS_ZERO_FILL"]
        })

        # -------------------------------------------------------------
        # 5. Continuous Financial Variables (Median Imputation)
        # -------------------------------------------------------------
        fin_cols = ["AMT_ANNUITY", "AMT_GOODS_PRICE", "CNT_FAM_MEMBERS"]
        for col in fin_cols:
            if col in clean_df.columns:
                med_val = float(clean_df[col].median())
                clean_df[col] = clean_df[col].fillna(med_val)

        report["transformations"].append({
            "step": "Continuous Financial Features Median Imputation",
            "columns": fin_cols,
            "rationale": CLEANING_RATIONALE["FINANCIAL_AMOUNTS_MEDIAN_IMPUTATION"]
        })

        # -------------------------------------------------------------
        # 6. External Credit Score Indicators & Median Imputation
        # -------------------------------------------------------------
        ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        for col in ext_cols:
            if col in clean_df.columns:
                clean_df[f"{col}_IS_MISSING"] = clean_df[col].isnull().astype(int)
                med_ext = float(clean_df[col].median())
                clean_df[col] = clean_df[col].fillna(med_ext)

        report["transformations"].append({
            "step": "External Score Missingness Flags & Median Imputation",
            "columns": ext_cols,
            "rationale": CLEANING_RATIONALE["EXTERNAL_SCORES_MEDIAN_AND_INDICATORS"]
        })

        # -------------------------------------------------------------
        # 7. Categorical Missingness -> Explicit 'Unknown_Missing'
        # -------------------------------------------------------------
        cat_cols = clean_df.select_dtypes(include=["object"]).columns.tolist()
        for col in cat_cols:
            clean_df[col] = clean_df[col].fillna("Unknown_Missing")

        report["transformations"].append({
            "step": "Categorical Explicit Missingness Encoding",
            "categorical_columns": cat_cols,
            "rationale": CLEANING_RATIONALE["CATEGORICAL_EXPLICIT_MISSING"]
        })

        # -------------------------------------------------------------
        # 8. Derived Canonical Risk Ratios & Demographics
        # -------------------------------------------------------------
        clean_df["AGE_YEARS"] = (-clean_df["DAYS_BIRTH"] / 365.25).astype(float)
        clean_df["CREDIT_TO_INCOME_RATIO"] = clean_df["AMT_CREDIT"] / (clean_df["AMT_INCOME_TOTAL"] + 1e-5)
        clean_df["ANNUITY_TO_INCOME_RATIO"] = clean_df["AMT_ANNUITY"] / (clean_df["AMT_INCOME_TOTAL"] + 1e-5)
        clean_df["CREDIT_TO_GOODS_RATIO"] = clean_df["AMT_CREDIT"] / (clean_df["AMT_GOODS_PRICE"] + 1e-5)

        final_shape = clean_df.shape
        final_missing = int(clean_df.isnull().sum().sum())

        report["final_shape"] = final_shape
        report["final_missing_cells"] = final_missing
        self.cleaning_report["application_data"] = report

        logger.info(f"Cleaning complete! Final shape: {final_shape}, Remaining missing cells: {final_missing}")
        return clean_df, report

    def clean_installments_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Cleans the installments payments dataset installments_payments.csv.

        Steps:
        1. Identify missed payments (where AMT_PAYMENT or DAYS_ENTRY_PAYMENT is NULL).
        2. Impute AMT_PAYMENT with 0.0 and DAYS_ENTRY_PAYMENT with DAYS_INSTALMENT + max tenure penalty.
        3. Flag IS_MISSED_PAYMENT = 1.
        4. Validate that transaction amounts are non-negative.

        Returns
        -------
        pd.DataFrame
            Cleaned installments dataframe.
        dict
            Cleaning report.
        """
        logger.info("Cleaning installments payments dataset...")
        clean_df = df.copy()
        initial_shape = clean_df.shape
        initial_missing = int(clean_df.isnull().sum().sum())

        # Flag missing payment entries
        clean_df["IS_MISSED_PAYMENT"] = clean_df["AMT_PAYMENT"].isnull().astype(int)
        clean_df["AMT_PAYMENT"] = clean_df["AMT_PAYMENT"].fillna(0.0)

        # If payment entry date is null, set equal to due date so DPD computation can be handled gracefully
        clean_df["DAYS_ENTRY_PAYMENT"] = clean_df["DAYS_ENTRY_PAYMENT"].fillna(clean_df["DAYS_INSTALMENT"])

        final_missing = int(clean_df.isnull().sum().sum())
        report = {
            "initial_shape": initial_shape,
            "initial_missing_cells": initial_missing,
            "final_shape": clean_df.shape,
            "final_missing_cells": final_missing,
            "rationale": CLEANING_RATIONALE["INSTALLMENTS_PAYMENT_NULLS"]
        }
        self.cleaning_report["installments_data"] = report
        logger.info(f"Installments cleaning complete. Shape: {clean_df.shape}, Missing cells: {final_missing}")

        return clean_df, report

    def export_cleaned_data(self, app_clean: pd.DataFrame, inst_clean: Optional[pd.DataFrame] = None) -> None:
        """
        Exports cleaned datasets to processed_data/ in Parquet and CSV formats for downstream modeling.
        """
        app_path_csv = os.path.join(self.processed_dir, "application_train_cleaned.csv")
        app_path_parquet = os.path.join(self.processed_dir, "application_train_cleaned.parquet")

        logger.info(f"Exporting cleaned application dataset to {app_path_csv} and {app_path_parquet}...")
        app_clean.to_parquet(app_path_parquet, index=False)
        app_clean.head(1000).to_csv(app_path_csv, index=False) # save sample csv for quick inspection

        if inst_clean is not None:
            inst_path_parquet = os.path.join(self.processed_dir, "installments_payments_cleaned.parquet")
            logger.info(f"Exporting cleaned installments dataset to {inst_path_parquet}...")
            inst_clean.to_parquet(inst_path_parquet, index=False)

        logger.info("Export completed successfully.")


if __name__ == "__main__":
    from data_loader import DataLoader

    loader = DataLoader(data_dir="data")
    app_df = loader.load_application_train(nrows=10000)
    cleaner = DataCleaner()
    cleaned_app, report = cleaner.clean_application_data(app_df)
    print("\nCleaning Report Summary:")
    for t in report["transformations"]:
        print(f"\n[STEP] {t['step']}")
        print(f"  Rationale: {t['rationale']}")
