import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

def evaluate_model(y_true, y_pred_proba, threshold=0.5):
    """Evaluates a model and returns ROC AUC and F1 Score."""
    y_pred = (np.array(y_pred_proba) >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_pred_proba)
    f1 = f1_score(y_true, y_pred)
    
    return {
        "ROC_AUC": auc,
        "F1_Score": f1
    }
