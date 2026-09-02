import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score
import xgboost as xgb
import lightgbm as lgb

class CreditRiskModels:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        
    def train_baseline_logistic(self, X_train, y_train):
        clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=self.random_state)
        clf.fit(X_train, y_train)
        self.models['logistic_regression_pipeline'] = clf
        return clf
        
    def train_xgboost(self, X_train, y_train):
        # Handle the 1:11 imbalance mathematically
        pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        clf = xgb.XGBClassifier(
            scale_pos_weight=pos_weight,
            max_depth=4,
            learning_rate=0.05,
            n_estimators=200,
            random_state=self.random_state,
            eval_metric='auc'
        )
        clf.fit(X_train, y_train)
        self.models['xgboost_model'] = clf
        return clf
        
    def train_lightgbm_champion(self, X_train, y_train):
        pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        clf = lgb.LGBMClassifier(
            scale_pos_weight=pos_weight,
            max_depth=6,
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=300,
            random_state=self.random_state
        )
        clf.fit(X_train, y_train)
        self.models['lightgbm_model'] = clf
        return clf
        
    def evaluate(self, model_name, X_test, y_test):
        clf = self.models[model_name]
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]
        
        metrics = {
            "Model Name": model_name,
            "ROC-AUC": roc_auc_score(y_test, probs),
            "PR-AUC": average_precision_score(y_test, probs),
            "Brier Score": brier_score_loss(y_test, probs),
            "F1-Score": f1_score(y_test, preds)
        }
        return metrics
        
    def save_models(self, path="models/"):
        os.makedirs(path, exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(path, f"{name}.joblib"))
