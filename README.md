# Credit Risk Intelligence System

An institutional-grade Credit Risk Analytics and Intelligence platform built for portfolio modeling, default prediction, survival analysis, explainable AI, causal uplift targeting, fairness auditing, and stress testing.

---

## 📂 Phase 1: Data Foundation (Completed)

Phase 1 establishes the foundational data architecture, exploratory risk profiling, and interview-defensible data cleaning pipeline.

### Architecture & Directory Structure
```
credit-risk-intelligence/
├── data/                                 # Raw application & installment datasets
│   ├── application_train.csv             # 307,511 loan applications (122 features)
│   └── installments_payments.csv         # 13.6M transaction repayment records
├── src/                                  # Modular core library
│   ├── __init__.py
│   ├── data_loader.py                    # Stage 1: High-perf ingestion, memory downcasting, missing reports
│   ├── eda.py                            # Stage 2: Deep EDA, cohort default analysis, repayment correlations
│   └── data_cleaner.py                   # Stage 3: Data cleaning & anomaly remediation with interview rationales
├── eda_outputs/                          # Stage 2 visual & tabular artifacts
│   ├── 01_class_imbalance_default_rate.png
│   ├── 02_distributions_key_financial_variables.png
│   ├── 03_segment_default_rates.png
│   ├── 04_external_scores_distribution.png
│   ├── 05_repayment_default_correlation_matrix.png
│   ├── 06_repayment_behavior_vs_default.png
│   └── segment_summary_*.csv             # Segment default rate tables
├── processed_data/                       # Stage 3 output datasets for Phase 2
│   ├── application_train_cleaned.parquet # 307,511 rows x 131 columns
│   ├── application_train_cleaned.csv     # Sample inspection CSV
│   └── installments_payments_cleaned.parquet
├── requirements.txt                      # Dependencies for all phases
└── run_phase1.py                         # End-to-end execution driver
```

---

## 🔬 Summary of Key Findings from Phase 1 EDA

1. **Severe Class Imbalance**:
   - Total records: **307,511**
   - Non-Default (`TARGET=0`): **282,686 (91.93%)**
   - Default (`TARGET=1`): **24,825 (8.07%)**
   - Imbalance ratio: **1 : 11.39**
   - *Implication for Phase 2*: Models require PR-AUC / ROC-AUC optimization, cost-sensitive matrices, and stratified cross-validation over standard accuracy.

2. **Demographic & Segment Default Stratification**:
   - **Age Risk Gradient**: Monotonic decrease in default risk with age:
     - 18–25: **12.29% default rate**
     - 25–34: **10.66% default rate**
     - 35–44: **8.41% default rate**
     - 45–54: **7.05% default rate**
     - 55–64: **5.42% default rate**
     - 65+: **3.66% default rate**
   - **Credit Limit Bands**: Tier 3 ($500k–$750k) loans exhibit peak default rate (**9.46%**), whereas Jumbo (> $1M) prime loans have the lowest (**5.87%**).
   - **Education Level**: Lower secondary applicants default at **10.93%**, compared to Academic Degree holders at **1.83%**.

3. **Repayment Friction & Behavioral Signals**:
   - Granular transaction aggregation from `installments_payments.csv` revealed that **Average Days Past Due (DPD)**, **Payment Deficits (underpayment)**, and **Percentage of Late Installments** correlate positively with future application default.
   - Defaulters exhibit significantly wider dispersion in payment shortfall and lower payment-to-installment ratios.

---

## 🛠️ Data Cleaning & Interview Defense Rationale Registry

| Anomaly / Feature | Issue in Raw Data | Treatment Applied | Interview-Ready Business & Theoretical Rationale |
| :--- | :--- | :--- | :--- |
| `DAYS_EMPLOYED == 365243` | 55,374 records have +365,243 days (~1,000 years). | Created `DAYS_EMPLOYED_ANOM = 1`, replaced raw value with NaN / 0, derived positive `YEARS_EMPLOYED`. | Legacy core banking sentinel value for pensioners/unemployed. Preserves the high predictive signal of retirement/unemployment while preventing severe gradient and linear weight distortion. |
| `AMT_REQ_CREDIT_BUREAU_*` | ~13.5% missing query counts across HOUR, DAY, WEEK, MON, QRT, YEAR. | Imputed with `0.0`. | Bureau APIs return NULL when no inquiry hits exist within the observation window. The missingness reflects 0 inquiries, not missing random observations. |
| `OBS/DEF_30/60_CNT_SOCIAL_CIRCLE` | ~0.33% missing. | Imputed with `0.0`. | Represents absence of recorded default events in the client's social network circle. |
| `OCCUPATION_TYPE` / Categoricals | 31.3% missing in occupation type. | Imputed with explicit category `'Unknown_Missing'`. | Missingness is **MNAR (Missing Not At Random)** — gig workers, informal laborers, and retirees often omit occupation. Imputing with Mode ('Laborers') introduces severe synthetic bias. Explicit encoding preserves signal for tree splits & WoE binning. |
| `EXT_SOURCE_1, 2, 3` | External bureau scores (56.4% missing in Source 1, 19.8% in Source 3). | Imputed with Median + added binary flags `EXT_SOURCE_1_IS_MISSING`, `EXT_SOURCE_3_IS_MISSING`. | Missingness is MNAR (selective bureau report procurement by underwriters based on applicant tier). Missingness flags preserve underwriter screening signal for linear/survival models. |
| `AMT_ANNUITY`, `AMT_GOODS_PRICE` | Missing in small fractions (<0.1%). | Imputed with sample Median. | Financial amount variables exhibit heavy right skewness; median preserves central tendency without outlier inflation. |
| `CODE_GENDER == 'XNA'`, `NAME_FAMILY_STATUS == 'Unknown'` | 4 records in gender, 2 in family status. | Imputed with respective modes (`F` and `Married`). | Preserves entire dataset integrity without losing records ahead of ECOA/fairness audits. |

---

## 🚀 How to Run Phase 1

```bash
# Activate virtual environment
.\venv\Scripts\activate

# Run full Phase 1 pipeline
python run_phase1.py
```
