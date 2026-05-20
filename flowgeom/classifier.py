from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve


def compute_joint_score(df, w1=1.0, w2=1.0, w3=1.0):
    return w1 * df["T1_curl_score"] + w2 * df["T2_straightness_score"] + w3 * df["T12_coupling_score"]


class FlowGeomClassifier:
    def __init__(self, mode="logistic_regression"):
        self.mode = mode
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


def tpr_at_fpr(y_true, y_score, target_fpr=0.01):
    thresholds = np.unique(y_score)[::-1]
    best = 0.0
    for th in thresholds:
        pred = (y_score >= th).astype(int)
        fp = ((pred == 1) & (y_true == 0)).sum()
        tp = ((pred == 1) & (y_true == 1)).sum()
        tn = ((pred == 0) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        fpr = fp / max(fp + tn, 1)
        tpr = tp / max(tp + fn, 1)
        if fpr <= target_fpr:
            best = max(best, tpr)
    return float(best)


def compute_eer(y_true, y_score) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def p_value_under_real_null(scores_real: np.ndarray, score_x: float) -> float:
    scores_real = np.asarray(scores_real, dtype=float)
    return float((np.sum(scores_real >= score_x) + 1.0) / (len(scores_real) + 1.0))
