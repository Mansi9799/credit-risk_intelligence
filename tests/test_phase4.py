import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.fairness_audit import calculate_fairness_metrics
from src.stress_testing import simulate_stress_test
from src.uplift_modeling import simulate_intervention_data

class TestPhase4:
    
    def test_fairness_metrics(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 0, 0, 1])
        sensitive_attr = pd.Series(['Male', 'Female', 'Male', 'Female', 'Female', 'Male'])
        
        metrics = calculate_fairness_metrics(y_true, y_pred, sensitive_attr)
        
        assert 'Male' in metrics
        assert 'Female' in metrics
        assert metrics['Male']['Population_Count'] == 3
        assert metrics['Female']['Population_Count'] == 3
        
        # Male: true=[0,1,1], pred=[0,1,1] => TP=2, TN=1, FP=0, FN=0 -> Approval (TN+FN)/Total = 1/3
        assert np.isclose(metrics['Male']['Approval_Rate'], 1.0 / 3.0)
        
    def test_stress_testing_mechanics(self):
        # Create dummy expected loss dataset
        df = pd.DataFrame({
            'EAD': [100000, 200000, 50000],
            'PD': [0.1, 0.2, 0.5],
            'LGD': [0.4, 0.5, 0.6]
        })
        
        results = simulate_stress_test(df)
        
        assert "Baseline" in results
        assert "Adverse" in results
        assert "Severely Adverse" in results
        
        baseline_el = results["Baseline"]["Total_Expected_Loss_Billions"]
        adverse_el = results["Adverse"]["Total_Expected_Loss_Billions"]
        severe_el = results["Severely Adverse"]["Total_Expected_Loss_Billions"]
        
        # Stressed EL should be strictly greater than or equal to Baseline EL
        assert adverse_el >= baseline_el
        assert severe_el >= adverse_el
        
    def test_uplift_simulation(self):
        df = pd.DataFrame({
            'TARGET': [0, 1, 0, 1, 0],
            'DEBT_TO_INCOME_RATIO': [0.2, 0.8, 0.4, 0.7, 0.1],
            'INST_AVG_DPD': [5, 90, 15, 60, 0],
            'EXT_SOURCE_MEAN': [0.8, 0.2, 0.5, 0.3, 0.9]
        })
        
        sim_df = simulate_intervention_data(df, sample_size=5)
        
        assert 'TREATMENT' in sim_df.columns
        assert 'OUTCOME_DEFAULT' in sim_df.columns
        assert 'TRUE_ITE' in sim_df.columns
        
        # ITE should be negative or zero (intervention reduces or maintains default prob)
        assert (sim_df['TRUE_ITE'] <= 0).all()
