"""
Exploratory Data Analysis (EDA) Module for Credit Risk Intelligence
------------------------------------------------------------------
Conducts deep statistical exploration, distributional profiling, segment-level 
default risk analysis, and transaction-level repayment behavior correlation.
Saves publication-quality figures and statistical summaries to eda_outputs/.

Author: Portfolio Project
Phase: 1 - Stage 2
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Tuple, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("EDA")

# Set visualization aesthetics
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

# Color Palette: Navy for Non-Default (0), Crimson for Default (1)
PALETTE_MAP = {0: "#2B5C8F", 1: "#D9383A"}
PALETTE_LIST = ["#2B5C8F", "#D9383A"]


class CreditRiskEDA:
    """
    Executes end-to-end Exploratory Data Analysis for credit risk intelligence,
    producing visual artifacts and analytical insight tables.
    """

    def __init__(self, output_dir: str = "eda_outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Initialized CreditRiskEDA. Outputs will be saved to: {self.output_dir}")

    def plot_target_distribution(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Visualizes class imbalance in loan default (TARGET: 0 vs 1).
        Computes exact default rate and minority class representation.
        """
        logger.info("Plotting target default rate and class distribution...")
        counts = df["TARGET"].value_counts().sort_index()
        pcts = df["TARGET"].value_counts(normalize=True).sort_index() * 100
        default_rate = float(pcts[1])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        # 1. Bar plot with counts & percentage labels
        bars = axes[0].bar(
            ["Non-Default (0)", "Default (1)"],
            counts.values,
            color=PALETTE_LIST,
            width=0.5,
            edgecolor="black",
            alpha=0.88
        )
        axes[0].set_title("Loan Default Class Distribution (Counts)", fontsize=13, fontweight="bold", pad=12)
        axes[0].set_ylabel("Number of Applicants", fontsize=11)
        axes[0].grid(axis="y", linestyle="--", alpha=0.6)

        for bar, count, pct in zip(bars, counts.values, pcts.values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (max(counts) * 0.015),
                f"{count:,}\n({pct:.2f}%)",
                ha="center", va="bottom", fontsize=11, fontweight="bold"
            )

        # 2. Donut plot highlighting portfolio default rate
        wedges, texts, autotexts = axes[1].pie(
            counts.values,
            labels=["Non-Default (0)", "Default (1)"],
            autopct="%1.2f%%",
            startangle=40,
            colors=PALETTE_LIST,
            explode=(0, 0.08),
            wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
            textprops=dict(fontsize=11)
        )
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
        axes[1].set_title(f"Overall Portfolio Default Rate: {default_rate:.2f}%", fontsize=13, fontweight="bold", pad=12)

        plt.suptitle("Portfolio Target Analysis: Severe Class Imbalance Profile", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, "01_class_imbalance_default_rate.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {out_path}")

        return {
            "total_records": len(df),
            "non_default_count": int(counts[0]),
            "default_count": int(counts[1]),
            "default_rate_pct": default_rate
        }

    def plot_continuous_distributions(self, df: pd.DataFrame) -> None:
        """
        Plots the distribution of key continuous financial variables:
        - AMT_CREDIT (Loan Credit Amount / Limit)
        - AMT_INCOME_TOTAL (Applicant Total Income)
        - AMT_ANNUITY (Loan Annuity / Installment)
        - AGE (Derived from DAYS_BIRTH)
        stratified by Default vs Non-Default.
        """
        logger.info("Plotting distributions for core financial and demographic variables...")
        plot_df = df.copy()

        # Derive Age in years from DAYS_BIRTH (stored as negative days relative to application)
        if "DAYS_BIRTH" in plot_df.columns and "AGE_YEARS" not in plot_df.columns:
            plot_df["AGE_YEARS"] = (-plot_df["DAYS_BIRTH"] / 365.25)

        variables = [
            ("AMT_CREDIT", "Loan Credit Amount ($)", True),
            ("AMT_INCOME_TOTAL", "Applicant Total Income ($)", True),
            ("AMT_ANNUITY", "Loan Annuity ($)", True),
            ("AGE_YEARS", "Applicant Age (Years)", False)
        ]

        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        axes = axes.flatten()

        for idx, (col, label, use_log) in enumerate(variables):
            ax = axes[idx]
            if col not in plot_df.columns:
                continue

            # Drop missing and extreme income outliers for visual stability (99th percentile cap for income)
            sub = plot_df[[col, "TARGET"]].dropna()
            if col == "AMT_INCOME_TOTAL":
                q99 = sub[col].quantile(0.99)
                sub = sub[sub[col] <= q99]

            sns.kdeplot(
                data=sub[sub["TARGET"] == 0],
                x=col,
                ax=ax,
                fill=True,
                color=PALETTE_MAP[0],
                label="Non-Default (0)",
                alpha=0.35,
                linewidth=2
            )
            sns.kdeplot(
                data=sub[sub["TARGET"] == 1],
                x=col,
                ax=ax,
                fill=True,
                color=PALETTE_MAP[1],
                label="Default (1)",
                alpha=0.35,
                linewidth=2
            )

            # Calculate and display medians
            med_0 = sub[sub["TARGET"] == 0][col].median()
            med_1 = sub[sub["TARGET"] == 1][col].median()

            ax.axvline(med_0, color=PALETTE_MAP[0], linestyle="--", linewidth=1.5, alpha=0.8)
            ax.axvline(med_1, color=PALETTE_MAP[1], linestyle="--", linewidth=1.5, alpha=0.8)

            ax.set_title(
                f"Distribution of {label}\n(Medians: Non-Default=${med_0:,.0f} | Default=${med_1:,.0f})",
                fontsize=11.5, fontweight="bold", pad=8
            )
            ax.set_xlabel(label, fontsize=10.5)
            ax.set_ylabel("Density", fontsize=10.5)
            ax.legend(loc="upper right", frameon=True)

            if use_log and col in ["AMT_CREDIT", "AMT_INCOME_TOTAL"]:
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${int(x):,}"))

        plt.suptitle("Distributional Shifts in Key Financial & Demographic Features by Default Status",
                     fontsize=15, fontweight="bold", y=1.01)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, "02_distributions_key_financial_variables.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {out_path}")

    def plot_segment_default_rates(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Computes and plots default rates broken down by critical risk segments:
        1. Age Groups (18-25, 25-35, 35-45, 45-55, 55-65, 65+)
        2. Credit Limit Bands (Quintiles / Defined Tiers)
        3. Education Level
        4. Contract Type
        """
        logger.info("Computing default rates across customer cohorts and risk segments...")
        plot_df = df.copy()

        # 1. Age Cohorts
        if "DAYS_BIRTH" in plot_df.columns:
            plot_df["AGE_YEARS"] = -plot_df["DAYS_BIRTH"] / 365.25
            age_bins = [18, 25, 35, 45, 55, 65, 100]
            age_labels = ["18-25", "25-34", "35-44", "45-54", "55-64", "65+"]
            plot_df["AGE_GROUP"] = pd.cut(plot_df["AGE_YEARS"], bins=age_bins, labels=age_labels, right=False)

        # 2. Credit Bands
        if "AMT_CREDIT" in plot_df.columns:
            credit_labels = ["Tier 1: <$250k", "Tier 2: $250k-$500k", "Tier 3: $500k-$750k", "Tier 4: $750k-$1M", "Tier 5: >$1M"]
            plot_df["CREDIT_BAND"] = pd.cut(
                plot_df["AMT_CREDIT"],
                bins=[0, 250000, 500000, 750000, 1000000, np.inf],
                labels=credit_labels,
                right=False
            )

        segment_summaries = {}

        # Plot 1: Age Group Default Rate
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Subplot 1: Age Groups
        if "AGE_GROUP" in plot_df.columns:
            age_summary = plot_df.groupby("AGE_GROUP", observed=False)["TARGET"].agg(
                Total="count", Default_Count="sum", Default_Rate="mean"
            ).reset_index()
            age_summary["Default_Rate_Pct"] = age_summary["Default_Rate"] * 100
            segment_summaries["age_groups"] = age_summary

            ax1 = axes[0, 0]
            bars1 = ax1.bar(
                age_summary["AGE_GROUP"].astype(str),
                age_summary["Default_Rate_Pct"],
                color="#3D6B99",
                edgecolor="black",
                alpha=0.85
            )
            ax1.set_title("Default Rate by Age Group\n(Younger Borrowers Exhibit Higher Default Propensity)", fontsize=11.5, fontweight="bold")
            ax1.set_xlabel("Age Cohort", fontsize=10.5)
            ax1.set_ylabel("Default Rate (%)", fontsize=10.5)
            ax1.set_ylim(0, max(age_summary["Default_Rate_Pct"]) * 1.25)
            for bar in bars1:
                yval = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.3, f"{yval:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=9.5)

        # Subplot 2: Credit Limit Bands
        if "CREDIT_BAND" in plot_df.columns:
            credit_summary = plot_df.groupby("CREDIT_BAND", observed=False)["TARGET"].agg(
                Total="count", Default_Count="sum", Default_Rate="mean"
            ).reset_index()
            credit_summary["Default_Rate_Pct"] = credit_summary["Default_Rate"] * 100
            segment_summaries["credit_bands"] = credit_summary

            ax2 = axes[0, 1]
            bars2 = ax2.bar(
                credit_summary["CREDIT_BAND"].astype(str),
                credit_summary["Default_Rate_Pct"],
                color="#5C82A6",
                edgecolor="black",
                alpha=0.85
            )
            ax2.set_title("Default Rate by Credit Limit Band", fontsize=11.5, fontweight="bold")
            ax2.set_xlabel("Credit Band", fontsize=10.5)
            ax2.set_ylabel("Default Rate (%)", fontsize=10.5)
            ax2.set_xticklabels(credit_summary["CREDIT_BAND"].astype(str), rotation=20, ha="right")
            ax2.set_ylim(0, max(credit_summary["Default_Rate_Pct"]) * 1.25)
            for bar in bars2:
                yval = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.3, f"{yval:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=9.5)

        # Subplot 3: Education Level
        if "NAME_EDUCATION_TYPE" in plot_df.columns:
            edu_summary = plot_df.groupby("NAME_EDUCATION_TYPE")["TARGET"].agg(
                Total="count", Default_Count="sum", Default_Rate="mean"
            ).reset_index().sort_values(by="Default_Rate", ascending=False)
            edu_summary["Default_Rate_Pct"] = edu_summary["Default_Rate"] * 100
            segment_summaries["education"] = edu_summary

            ax3 = axes[1, 0]
            bars3 = ax3.barh(
                edu_summary["NAME_EDUCATION_TYPE"],
                edu_summary["Default_Rate_Pct"],
                color="#7A9ABA",
                edgecolor="black",
                alpha=0.85
            )
            ax3.set_title("Default Rate by Education Level", fontsize=11.5, fontweight="bold")
            ax3.set_xlabel("Default Rate (%)", fontsize=10.5)
            ax3.set_ylabel("Education Type", fontsize=10.5)
            ax3.set_xlim(0, max(edu_summary["Default_Rate_Pct"]) * 1.2)
            for bar in bars3:
                xval = bar.get_width()
                ax3.text(xval + 0.3, bar.get_y() + bar.get_height()/2, f"{xval:.2f}%", ha="left", va="center", fontweight="bold", fontsize=9.5)

        # Subplot 4: Contract Type & Gender
        if "NAME_CONTRACT_TYPE" in plot_df.columns:
            contract_summary = plot_df.groupby("NAME_CONTRACT_TYPE")["TARGET"].agg(
                Total="count", Default_Count="sum", Default_Rate="mean"
            ).reset_index()
            contract_summary["Default_Rate_Pct"] = contract_summary["Default_Rate"] * 100
            segment_summaries["contract_type"] = contract_summary

            ax4 = axes[1, 1]
            bars4 = ax4.bar(
                contract_summary["NAME_CONTRACT_TYPE"],
                contract_summary["Default_Rate_Pct"],
                color=["#2B5C8F", "#E67E22"],
                edgecolor="black",
                alpha=0.85,
                width=0.4
            )
            ax4.set_title("Default Rate by Loan Contract Type\n(Cash Loans vs Revolving Loans)", fontsize=11.5, fontweight="bold")
            ax4.set_xlabel("Contract Type", fontsize=10.5)
            ax4.set_ylabel("Default Rate (%)", fontsize=10.5)
            ax4.set_ylim(0, max(contract_summary["Default_Rate_Pct"]) * 1.3)
            for bar in bars4:
                yval = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2, yval + 0.3, f"{yval:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=9.5)

        plt.suptitle("Credit Risk Stratification: Segment-Level Default Rate Benchmarks", fontsize=15, fontweight="bold", y=1.01)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, "03_segment_default_rates.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {out_path}")

        # Save segment metrics to CSV
        for key, sum_df in segment_summaries.items():
            sum_df.to_csv(os.path.join(self.output_dir, f"segment_summary_{key}.csv"), index=False)

        return segment_summaries

    def plot_external_credit_scores(self, df: pd.DataFrame) -> None:
        """
        Plots the distribution of external credit agency scores (EXT_SOURCE_1, 2, 3),
        demonstrating why normalized bureau scores are critical risk discriminators.
        """
        logger.info("Plotting external credit risk score distributions...")
        ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        available_ext = [c for c in ext_cols if c in df.columns]

        if not available_ext:
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, col in enumerate(available_ext):
            ax = axes[idx]
            sub = df[[col, "TARGET"]].dropna()

            sns.kdeplot(
                data=sub[sub["TARGET"] == 0],
                x=col,
                ax=ax,
                fill=True,
                color=PALETTE_MAP[0],
                label="Non-Default (0)",
                alpha=0.4,
                linewidth=2
            )
            sns.kdeplot(
                data=sub[sub["TARGET"] == 1],
                x=col,
                ax=ax,
                fill=True,
                color=PALETTE_MAP[1],
                label="Default (1)",
                alpha=0.4,
                linewidth=2
            )

            med_0 = sub[sub["TARGET"] == 0][col].median()
            med_1 = sub[sub["TARGET"] == 1][col].median()

            ax.axvline(med_0, color=PALETTE_MAP[0], linestyle="--", linewidth=1.5)
            ax.axvline(med_1, color=PALETTE_MAP[1], linestyle="--", linewidth=1.5)

            ax.set_title(
                f"{col}\n(Median 0: {med_0:.3f} | Median 1: {med_1:.3f})",
                fontsize=11.5, fontweight="bold", pad=8
            )
            ax.set_xlabel(f"{col} (Normalized Agency Score)", fontsize=10.5)
            ax.set_ylabel("Density", fontsize=10.5)
            ax.legend(loc="upper right")

        plt.suptitle("External Credit Scores: Powerful Discriminative Ability Between Default and Non-Default",
                     fontsize=14.5, fontweight="bold", y=1.03)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, "04_external_scores_distribution.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {out_path}")

    def analyze_repayment_behavior(
        self,
        app_df: pd.DataFrame,
        inst_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aggregates granular installment repayment transactions to client-level metrics:
        - DPD (Days Past Due = max(0, DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT))
        - Payment-to-Installment Ratio (AMT_PAYMENT / AMT_INSTALMENT)
        - Payment Deficit / Shortfall (AMT_INSTALMENT - AMT_PAYMENT)
        - Count of Late Installments (DPD > 0)
        - Count of Underpaid Installments (AMT_PAYMENT < AMT_INSTALMENT)
        - Missed Installment Flag (DAYS_ENTRY_PAYMENT is NaN or AMT_PAYMENT == 0)

        Correlates these behavioral features directly with loan default.
        """
        logger.info("Engineering client-level behavioral features from installment transaction records...")

        # Work on a copy of required columns
        inst = inst_df[[
            "SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"
        ]].copy()

        # 1. Payment Delay / DPD (Days Past Due: positive means payment was late)
        # Note: DAYS_INSTALMENT and DAYS_ENTRY_PAYMENT are negative numbers (days before current application)
        # Entry date - Due date: If entry was -10 and due was -20, entry was 10 days LATER than due date.
        inst["DPD"] = (inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]).clip(lower=0)
        inst["DAYS_EARLY"] = (inst["DAYS_INSTALMENT"] - inst["DAYS_ENTRY_PAYMENT"]).clip(lower=0)

        # 2. Payment Underpayment & Shortfall
        inst["PAYMENT_DEFICIT"] = (inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]).clip(lower=0)
        inst["PAYMENT_RATIO"] = np.where(
            inst["AMT_INSTALMENT"] > 0,
            inst["AMT_PAYMENT"] / inst["AMT_INSTALMENT"],
            1.0
        )
        inst["IS_LATE"] = (inst["DPD"] > 0).astype(int)
        inst["IS_UNDERPAID"] = (inst["PAYMENT_DEFICIT"] > 1.0).astype(int)

        # Aggregate to Client Level (SK_ID_CURR)
        logger.info("Aggregating transaction statistics per borrower (SK_ID_CURR)...")
        client_repay = inst.groupby("SK_ID_CURR").agg(
            TOTAL_INSTALLMENTS=("AMT_INSTALMENT", "count"),
            AVG_DPD=("DPD", "mean"),
            MAX_DPD=("DPD", "max"),
            COUNT_LATE_PAYMENTS=("IS_LATE", "sum"),
            PCT_LATE_PAYMENTS=("IS_LATE", "mean"),
            AVG_PAYMENT_RATIO=("PAYMENT_RATIO", "mean"),
            MIN_PAYMENT_RATIO=("PAYMENT_RATIO", "min"),
            TOTAL_PAYMENT_DEFICIT=("PAYMENT_DEFICIT", "sum"),
            AVG_PAYMENT_DEFICIT=("PAYMENT_DEFICIT", "mean"),
            COUNT_UNDERPAID=("IS_UNDERPAID", "sum"),
            AVG_DAYS_EARLY=("DAYS_EARLY", "mean")
        ).reset_index()

        # Merge with TARGET in application_train
        merged = app_df[["SK_ID_CURR", "TARGET", "AMT_CREDIT", "AMT_INCOME_TOTAL", "EXT_SOURCE_2", "EXT_SOURCE_3"]].merge(
            client_repay, on="SK_ID_CURR", how="inner"
        )

        logger.info(f"Merged client repayment features with application target. Shape: {merged.shape}")

        # Compute Correlation Matrix with TARGET
        corr_cols = [
            "TARGET", "AVG_DPD", "MAX_DPD", "COUNT_LATE_PAYMENTS", "PCT_LATE_PAYMENTS",
            "AVG_PAYMENT_RATIO", "MIN_PAYMENT_RATIO", "AVG_PAYMENT_DEFICIT",
            "TOTAL_PAYMENT_DEFICIT", "COUNT_UNDERPAID", "AVG_DAYS_EARLY",
            "AMT_CREDIT", "EXT_SOURCE_2", "EXT_SOURCE_3"
        ]
        available_corr_cols = [c for c in corr_cols if c in merged.columns]

        corr_pearson = merged[available_corr_cols].corr(method="pearson")
        corr_spearman = merged[available_corr_cols].corr(method="spearman")

        # Save Correlation Heatmap
        fig, ax = plt.subplots(figsize=(13, 10))
        sns.heatmap(
            corr_spearman,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            center=0,
            linewidths=0.5,
            cbar_kws={"label": "Spearman Rank Correlation Coefficient"},
            ax=ax
        )
        ax.set_title("Correlation Heatmap: Repayment Behavior vs Default Target & Financial Factors",
                     fontsize=13.5, fontweight="bold", pad=12)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, "05_repayment_default_correlation_matrix.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {out_path}")

        # Plot 6: Key Repayment Metrics Box/KDE Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        repay_metrics = [
            ("AVG_DPD", "Average Days Past Due (DPD)", False),
            ("PCT_LATE_PAYMENTS", "Percentage of Late Installments", False),
            ("AVG_PAYMENT_RATIO", "Average Payment-to-Installment Ratio", False),
            ("AVG_PAYMENT_DEFICIT", "Average Payment Shortfall ($)", True)
        ]

        for idx, (col, label, use_log) in enumerate(repay_metrics):
            ax = axes[idx // 2, idx % 2]
            sub = merged[[col, "TARGET"]].dropna()

            # Trim 95th percentile for extreme outliers in visual display
            q95 = sub[col].quantile(0.95)
            q05 = sub[col].quantile(0.01)
            sub_trimmed = sub[(sub[col] <= q95) & (sub[col] >= q05)]

            sns.boxplot(
                data=sub_trimmed,
                x="TARGET",
                y=col,
                palette=PALETTE_LIST,
                ax=ax,
                width=0.4,
                showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "yellow", "markeredgecolor": "black"}
            )
            ax.set_title(f"Comparison of {label} by Default Status", fontsize=11.5, fontweight="bold")
            ax.set_xticklabels(["Non-Default (0)", "Default (1)"])
            ax.set_xlabel("Target Status", fontsize=10.5)
            ax.set_ylabel(label, fontsize=10.5)

        plt.suptitle("Repayment Friction Analysis: Defaulters Exhibit Higher DPD & Payment Deficits",
                     fontsize=15, fontweight="bold", y=1.01)
        plt.tight_layout()

        out_path_metrics = os.path.join(self.output_dir, "06_repayment_behavior_vs_default.png")
        plt.savefig(out_path_metrics, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {out_path_metrics}")

        # Export correlation table to CSV
        corr_spearman.to_csv(os.path.join(self.output_dir, "repayment_spearman_correlation.csv"))
        corr_pearson.to_csv(os.path.join(self.output_dir, "repayment_pearson_correlation.csv"))

        return client_repay, merged
