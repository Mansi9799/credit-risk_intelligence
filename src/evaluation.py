import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, 
    f1_score, 
    precision_recall_curve, 
    auc, 
    brier_score_loss,
    confusion_matrix
)
from scipy.stats import ks_2samp

def evaluate_model(y_true, y_pred_proba, threshold=0.5):
    """
    Comprehensive institutional credit risk model evaluation.
    Computes ROC-AUC, PR-AUC, Gini, KS-Statistic, Brier Score, and F1.
    """
    y_pred = (np.array(y_pred_proba) >= threshold).astype(int)
    
    # 1. ROC AUC & Gini
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    gini = (2 * roc_auc) - 1
    
    # 2. PR AUC
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # 3. Brier Score
    brier = brier_score_loss(y_true, y_pred_proba)
    
    # 4. KS-Statistic
    mask_default = (y_true == 1)
    mask_non_default = (y_true == 0)
    
    if sum(mask_default) > 0 and sum(mask_non_default) > 0:
        ks_stat, _ = ks_2samp(y_pred_proba[mask_default], y_pred_proba[mask_non_default])
    else:
        ks_stat = 0.0
        
    # 5. F1 & Confusion Matrix
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        recall_sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        recall_sens, specificity = 0.0, 0.0
    
    return {
        "ROC_AUC": float(roc_auc),
        "PR_AUC": float(pr_auc),
        "Gini": float(gini),
        "KS_Statistic": float(ks_stat),
        "Brier_Score": float(brier),
        "F1_Score": float(f1),
        "Recall": float(recall_sens),
        "Specificity": float(specificity)
    }
