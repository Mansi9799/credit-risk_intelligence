# Institutional Credit Risk & Intervention Intelligence System

An end-to-end, institutional-grade credit risk analytics platform designed for portfolio modeling, default prediction, survival analysis, explainable AI, causal uplift targeting, fairness auditing, and stress testing. 

## 🚀 Pipeline Overview (Phases 1–5)

This system moves beyond traditional binary yes/no credit classifiers to provide a real-world, regulatory-compliant risk and causal intervention platform.

1. **Phase 1: Data Foundation & Cleaning**: Data ingestion, severe class imbalance profiling, and robust anomaly remediation (Missing Not At Random handling, explicit encoding).
2. **Phase 2: Feature Engineering & Baseline Models**: Creation of 167 features (financial burden ratios, repayment velocity) and benchmark classification (Logistic, XGBoost, LightGBM Champion).
3. **Phase 3: Survival Modeling (Time-to-Default)**: Kaplan-Meier estimations and Regularized Cox Proportional Hazards for dynamic PD curves.
4. **Stage 7: LGD, EAD & Expected Loss**: Basel II/III Expected Loss formulation computing portfolio-wide risk.
5. **Phase 4 & 5: Advanced Intelligence & UI**: SHAP (Explainability), Counterfactuals (Interventions), Uplift Modeling, Fairness Auditing, CCAR Stress Testing, and an interactive Streamlit application.

---

## 🧠 Beyond Standard Classification

While standard binary classification (e.g., Logistic Regression or LightGBM predicting Default vs. Non-Default) provides a static snapshot of risk, institutional banking requires advanced methodologies to handle temporal dynamics and drive actionable decisions.

### 1. Why Survival Analysis (Cox PH)?
Standard classifiers ignore the **timing** of a default and suffer from **right-censoring** bias (e.g., a non-defaulter at month 12 might default at month 24, but a standard classifier treats them as a hard "0"). 
* **Time-to-Default**: Survival analysis models the *instantaneous hazard rate* over the entire loan lifecycle.
* **Term Structure**: Allows us to generate dynamic Probability of Default (PD) curves at 12M, 24M, 36M, etc.
* **Regulatory Compliance**: Cox PH guarantees risk monotonicity (e.g., rising DTI strictly elevates instantaneous hazard), necessary for OCC SR 11-7 model validation.

### 2. Why Causal Uplift Modeling?
Predicting that someone will default is only half the problem; determining *what to do about it* is the true objective.
* **The Problem with Standard Risk Models**: They treat all high-risk clients equally (rejection).
* **Uplift Modeling (CATE/T-Learner)**: Estimates the Conditional Average Treatment Effect of an intervention (e.g., debt restructuring, term extension, interest rate reduction).
* **Outcome**: Segments clients into **Persuadables** (those who can be saved by an intervention), **Lost Causes** (will default regardless), and **Sure Things** (will repay regardless). This maximizes ROI on loss-mitigation budgets.

---

## 📊 Evaluation Metrics Dictionary

Understanding the evaluation metrics used across the pipeline:

* **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**: Measures the model's ability to rank-order risk. An AUC of 0.77 means the model ranks a randomly chosen defaulter higher than a non-defaulter 77% of the time.
* **PR-AUC (Precision-Recall AUC)**: Critical for highly imbalanced datasets (our target is 92% negative). It measures the trade-off between capturing defaulters (Recall) without falsely flagging non-defaulters (Precision).
* **KS-Statistic (Kolmogorov-Smirnov)**: Measures the maximum degree of separation between the score distributions of defaulters and non-defaulters. Higher is better.
* **Brier Score**: Measures the accuracy of probabilistic predictions (Mean Squared Error of probabilities). Lower is better.
* **Harrell's C-Index (Concordance Index)**: The survival analysis equivalent of ROC-AUC. It measures whether the model correctly orders the *timing* of events, accounting for right-censored data.
* **Expected Loss (EL)**: The regulatory Basel II/III risk metric. $EL = PD \times LGD \times EAD$. It quantifies the expected dollar loss for the portfolio.

---

## 🛠️ Data Cleaning & Interview Defense Rationale Registry (Phase 1)

| Anomaly / Feature | Treatment Applied | Interview-Ready Business & Theoretical Rationale |
| :--- | :--- | :--- |
| `DAYS_EMPLOYED == 365243` | Created flag, replaced with 0/NaN. | Legacy banking sentinel value for pensioners/unemployed. Preserves predictive signal while preventing linear weight distortion. |
| `AMT_REQ_CREDIT_BUREAU_*` | Imputed with `0.0`. | Bureau APIs return NULL when no inquiry hits exist. Reflects 0 inquiries, not missing random data. |
| `OCCUPATION_TYPE` | Imputed with `'Unknown_Missing'`. | Missingness is **MNAR** (gig workers, informal labor often omit). Explicit encoding preserves signal for tree splits. |
| `EXT_SOURCE_1, 2, 3` | Imputed with Median + missing flags. | Missingness is MNAR (selective bureau pulls). Flags preserve underwriter screening signal. |
| Gender/Family Status | Imputed with respective modes. | Preserves entire dataset integrity ahead of ECOA/fairness audits. |

---

## 🖥️ Interactive Dashboard (Phase 5)

The intelligence layer is exposed via a Streamlit application featuring:
1. **Client Risk Profiles**: Individual PD, LGD, EAD, EL.
2. **Explainability**: SHAP-driven adverse action reasons (FCRA/ECOA compliant) and DiCE counterfactual interventions.
3. **Portfolio Stress Testing**: Macroeconomic shocks applied to PD/LGD simulating Baseline, Adverse, and Severely Adverse CCAR scenarios.
4. **Causal Interventions**: Uplift segmentation isolating Persuadables.

## 🚀 How to Run the Pipeline

```bash
# 1. Activate virtual environment
.\venv\Scripts\activate

# 2. Run Test Suite
pytest tests/ -v

# 3. Run Pipeline Stages
python run_phase1.py    # Phase 1: Data Foundation & EDA
python run_phase2.py    # Phase 2: Feature Engineering & Baseline Models
python run_phase3.py    # Phase 3: Survival Modeling (Time-to-Default)
python run_stage7.py    # Stage 7: LGD, EAD & Expected Loss
python run_phase4.py    # Phase 4: Advanced Intelligence & Governance

# 4. Launch Dashboard
streamlit run app.py
```
