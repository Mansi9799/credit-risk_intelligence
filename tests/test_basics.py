import pandas as pd
import numpy as np
import pytest
from src.feature_engineering import calculate_dti, calculate_employment_ratio
from src.evaluation import evaluate_model

def test_calculate_dti():
    df = pd.DataFrame({
        'AMT_ANNUITY': [1000, 2000],
        'AMT_INCOME_TOTAL': [5000, 4000]
    })
    result = calculate_dti(df)
    assert 'DEBT_TO_INCOME_RATIO' in result.columns
    assert result['DEBT_TO_INCOME_RATIO'].iloc[0] == 0.2
    assert result['DEBT_TO_INCOME_RATIO'].iloc[1] == 0.5

def test_calculate_employment_ratio():
    df = pd.DataFrame({
        'DAYS_EMPLOYED': [-1000, -2000],
        'DAYS_BIRTH': [-10000, -8000]
    })
    result = calculate_employment_ratio(df)
    assert 'EMPLOYMENT_TO_AGE_RATIO' in result.columns
    assert result['EMPLOYMENT_TO_AGE_RATIO'].iloc[0] == 0.1
    assert result['EMPLOYMENT_TO_AGE_RATIO'].iloc[1] == 0.25

def test_evaluate_model():
    y_true = [0, 0, 1, 1]
    y_pred_proba = [0.1, 0.4, 0.6, 0.9]
    
    metrics = evaluate_model(y_true, y_pred_proba, threshold=0.5)
    
    assert "ROC_AUC" in metrics
    assert "F1_Score" in metrics
    assert metrics["ROC_AUC"] == 1.0  # Perfect separation
    assert metrics["F1_Score"] == 1.0
