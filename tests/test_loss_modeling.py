"""
Unit & Integration Test Suite for Loss Modeling (Stage 7)
=========================================================
Tests LGD (Loss Given Default), EAD (Exposure at Default), and
Expected Loss (EL = PD x LGD x EAD) calculation mechanics.
"""

import os
import pytest
import numpy as np
import pandas as pd
from src.loss_modeling import LGDModel, EADModel, ExpectedLossCalculator


@pytest.fixture
def sample_loans_df():
    """
    Constructs a synthetic fixture dataframe representing diverse loan applications.
    """
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "SK_ID_CURR": np.arange(100001, 100001 + n),
        "TARGET": np.random.binomial(1, 0.1, n),
        "AMT_CREDIT": np.random.uniform(100000, 1000000, n),
        "AMT_GOODS_PRICE": np.random.uniform(80000, 950000, n),
        "AMT_ANNUITY": np.random.uniform(5000, 50000, n),
        "AMT_INCOME_TOTAL": np.random.uniform(50000, 300000, n),
        "NAME_CONTRACT_TYPE": np.random.choice(["Cash loans", "Revolving loans"], n, p=[0.85, 0.15]),
        "FLAG_OWN_REALTY": np.random.choice(["Y", "N"], n, p=[0.70, 0.30]),
        "FLAG_OWN_CAR": np.random.choice(["Y", "N"], n, p=[0.40, 0.60]),
        "CREDIT_TERM_MONTHS": np.random.uniform(12, 48, n),
        "CREDIT_TO_INCOME_RATIO": np.random.uniform(1.5, 6.0, n),
        "ANNUITY_TO_INCOME_RATIO": np.random.uniform(0.1, 0.4, n),
        "DEBT_TO_INCOME_RATIO": np.random.uniform(0.1, 0.5, n),
        "YEARS_EMPLOYED": np.random.uniform(0, 30, n),
        "AGE_YEARS": np.random.uniform(21, 65, n),
        "EXT_SOURCE_MEAN": np.random.uniform(0.2, 0.8, n),
        "INST_AVG_PAYMENT_RATIO": np.random.uniform(0.7, 1.2, n),
        "INST_PCT_UNDERPAID": np.random.uniform(0.0, 0.3, n),
        "INST_AVG_DPD": np.random.uniform(0.0, 15.0, n),
        "REGION_RATING_CLIENT_W_CITY": np.random.choice([1, 2, 3], n)
    })
    return df


class TestLGDModeling:
    """Test suite for Loss Given Default (LGD) mechanics."""

    def test_empirical_lgd_bounds_and_mechanics(self, sample_loans_df):
        """Verify empirical LGD values are strictly within [0.05, 0.95] and reflect collateral effects."""
        lgd = LGDModel.generate_empirical_lgd(sample_loans_df, random_state=42)
        
        assert len(lgd) == len(sample_loans_df)
        assert np.all(lgd >= 0.05), "LGD should be bounded below by 0.05"
        assert np.all(lgd <= 0.95), "LGD should be bounded above by 0.95"
        assert isinstance(lgd, np.ndarray)

        # Secured / Realty owners should on average have lower LGD than unsecured non-realty
        realty_mask = sample_loans_df["FLAG_OWN_REALTY"] == "Y"
        lgd_realty = lgd[realty_mask].mean()
        lgd_no_realty = lgd[~realty_mask].mean()
        assert lgd_realty < lgd_no_realty, "Asset ownership must decrease average LGD."

    def test_lgd_model_training_and_prediction(self, sample_loans_df):
        """Test LGD model training, prediction, and feature importance."""
        model = LGDModel(min_lgd=0.05, max_lgd=0.95, random_state=42)
        model.fit(sample_loans_df)

        preds = model.predict(sample_loans_df)
        assert len(preds) == len(sample_loans_df)
        assert np.all((preds >= 0.05) & (preds <= 0.95))

        # Check feature importance
        imp_df = model.get_feature_importance()
        assert not imp_df.empty
        assert "feature" in imp_df.columns
        assert "importance" in imp_df.columns
        assert imp_df["importance"].sum() > 0.99

    def test_lgd_model_save_and_load(self, sample_loans_df, tmp_path):
        """Test model serialization and exact deserialization reproduction."""
        model = LGDModel(min_lgd=0.05, max_lgd=0.95, random_state=42)
        model.fit(sample_loans_df)
        preds_orig = model.predict(sample_loans_df)

        save_file = os.path.join(tmp_path, "test_lgd_model.joblib")
        model.save(save_file)
        assert os.path.exists(save_file)

        loaded_model = LGDModel.load(save_file)
        preds_loaded = loaded_model.predict(sample_loans_df)
        np.testing.assert_allclose(preds_orig, preds_loaded, rtol=1e-5)


class TestEADModeling:
    """Test suite for Exposure at Default (EAD) mechanics."""

    def test_ead_revolving_ccf_and_cash_amortization(self, sample_loans_df):
        """Verify revolving loans use CCF and cash loans apply contractual amortization factor."""
        ead_engine = EADModel(revolving_ccf=0.75, default_amortization_factor=0.70, accrued_interest_margin=0.03)
        ead_amounts, ead_factors = ead_engine.calculate_ead(sample_loans_df)

        assert len(ead_amounts) == len(sample_loans_df)
        assert np.all(ead_amounts > 0), "EAD must be strictly positive"
        assert np.all(ead_factors > 0), "EAD factors must be strictly positive"

        # Revolving lines should have exact CCF = 0.75
        revolving_mask = sample_loans_df["NAME_CONTRACT_TYPE"] == "Revolving loans"
        np.testing.assert_allclose(ead_factors[revolving_mask], 0.75, rtol=1e-5)

        # Cash loans with default factor 0.70 and 3% interest margin
        cash_mask = ~revolving_mask
        expected_cash_factor = 0.70 * 1.03
        np.testing.assert_allclose(ead_factors[cash_mask], expected_cash_factor, rtol=1e-5)

    def test_ead_with_survival_durations(self, sample_loans_df):
        """Verify dynamic amortization with survival elapsed durations."""
        ead_engine = EADModel(revolving_ccf=0.75, accrued_interest_margin=0.0)
        durations = np.full(len(sample_loans_df), 12.0)  # defaulted at month 12
        
        ead_amounts, ead_factors = ead_engine.calculate_ead(sample_loans_df, survival_durations=durations)
        assert len(ead_amounts) == len(sample_loans_df)
        assert np.all(ead_factors <= 1.05)


class TestExpectedLossEngine:
    """Test suite for Expected Loss (EL = PD x LGD x EAD) calculation & portfolio summary."""

    def test_expected_loss_mathematical_identity(self, sample_loans_df):
        """Verify EL is mathematically exact: EL = PD * LGD * EAD."""
        n = len(sample_loans_df)
        pd_scores = np.random.uniform(0.01, 0.80, n)
        lgd_scores = np.random.uniform(0.20, 0.70, n)
        ead_amounts = sample_loans_df["AMT_CREDIT"].values * 0.72

        calc = ExpectedLossCalculator(output_dir="loss_outputs")
        loss_df = calc.calculate_expected_loss(sample_loans_df, pd_scores, lgd_scores, ead_amounts)

        # Verify mathematical identity
        expected_el = pd_scores * lgd_scores * ead_amounts
        np.testing.assert_allclose(loss_df["EXPECTED_LOSS"].values, expected_el, rtol=1e-4)

        # Zero PD must yield Zero EL
        zero_pd_df = calc.calculate_expected_loss(sample_loans_df, np.zeros(n), lgd_scores, ead_amounts)
        assert np.all(zero_pd_df["EXPECTED_LOSS"] == 0.0)

    def test_risk_tier_stratification(self, sample_loans_df):
        """Verify risk tiers correctly partition by PD (Prime < 0.20, Subprime >= 0.50)."""
        pd_scores = np.array([0.05, 0.30, 0.75] * (len(sample_loans_df) // 3 + 1))[:len(sample_loans_df)]
        lgd_scores = np.full(len(sample_loans_df), 0.45)
        ead_amounts = np.full(len(sample_loans_df), 100000.0)

        calc = ExpectedLossCalculator()
        loss_df = calc.calculate_expected_loss(sample_loans_df, pd_scores, lgd_scores, ead_amounts)

        prime_el = loss_df[loss_df["RISK_TIER"] == "Prime"]["EXPECTED_LOSS"].mean()
        subprime_el = loss_df[loss_df["RISK_TIER"] == "Subprime"]["EXPECTED_LOSS"].mean()
        assert subprime_el > prime_el, "Subprime tier must exhibit higher Expected Loss than Prime."

    def test_portfolio_summary_metrics(self, sample_loans_df, tmp_path):
        """Test portfolio aggregation dictionary and output generation."""
        calc = ExpectedLossCalculator(output_dir=str(tmp_path))
        n = len(sample_loans_df)
        pd_scores = np.random.uniform(0.05, 0.40, n)
        lgd_scores = np.random.uniform(0.30, 0.60, n)
        ead_amounts = sample_loans_df["AMT_CREDIT"].values * 0.70

        loss_df = calc.calculate_expected_loss(sample_loans_df, pd_scores, lgd_scores, ead_amounts)
        summary = calc.summarize_portfolio(loss_df)

        assert summary["total_loans"] == n
        assert summary["total_portfolio_expected_loss"] > 0
        assert 0.0 < summary["portfolio_el_rate_pct"] < 100.0
        assert len(summary["tier_breakdown"]) > 0

        # Check generated summary files
        assert os.path.exists(os.path.join(tmp_path, "expected_loss_tier_summary.csv"))
        assert os.path.exists(os.path.join(tmp_path, "expected_loss_portfolio_summary.json"))
