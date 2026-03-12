import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def tpr_variance(y, yp, a):
    groups = np.unique(a)
    tprs = []
    for g in groups:
        mask = (a == g)
        if mask.sum() == 0:
            continue
        pos_mask = (y[mask] == 1)
        if pos_mask.sum() == 0:
            continue
        tpr = np.mean(yp[mask][pos_mask] == 1)
        tprs.append(tpr)
    return float(np.var(tprs)) if len(tprs) > 0 else 0.0

def fpr_variance(y, yp, a):
    groups = np.unique(a)
    fprs = []
    for g in groups:
        mask = (a == g)
        if mask.sum() == 0:
            continue
        neg_mask = (y[mask] == 0)
        if neg_mask.sum() == 0:
            continue
        fpr = np.mean(yp[mask][neg_mask] == 1)
        fprs.append(fpr)
    return float(np.var(fprs)) if len(fprs) > 0 else 0.0


metrics_dict = {
    "accuracy": lambda y, yp, a: accuracy_score(y, yp),
    "precision": lambda y, yp, a: precision_score(y, yp),
    "recall": lambda y, yp, a: recall_score(y, yp),
    "f1_score": lambda y, yp, a: f1_score(y, yp),
    "tpr_variance": tpr_variance,
    "fpr_variance": fpr_variance
}