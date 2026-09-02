"""
src/evaluation.py
-----------------
Evaluation utilities for binary congestion spillover classification.

Calculates:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix (TN, FP, FN, TP)
- Classification Report
"""

import json
import logging
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("evaluation")


def evaluate_classifier(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, model_name: str = "Model") -> dict:
    """
    Compute comprehensive classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        roc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc = 0.5
        
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    metrics = {
        "model_name": model_name,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc), 4),
        "confusion_matrix": cm,
    }
    
    log.info(
        "[%s] Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f | ROC-AUC: %.4f",
        model_name, acc, prec, rec, f1, roc
    )
    return metrics
