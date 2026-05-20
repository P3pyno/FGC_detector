from __future__ import annotations
import pickle
import numpy as np
from .utils import robust_stats


class RobustCalibrator:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.params = {}

    def fit(self, X, feature_names):
        for i, f in enumerate(feature_names):
            c, s = robust_stats(np.asarray(X)[:, i], eps=self.eps)
            self.params[f] = (c, s)
        return self

    def transform(self, X, feature_names):
        Z = np.asarray(X, dtype=float).copy()
        for i, f in enumerate(feature_names):
            c, s = self.params[f]
            Z[:, i] = (Z[:, i] - c) / s
        return Z

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
