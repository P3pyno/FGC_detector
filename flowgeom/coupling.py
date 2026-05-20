from __future__ import annotations
import numpy as np


def compute_coupling_features(per_time_curl: dict, per_time_curv: dict):
    """Couple local random curl probes with global trajectory curvature over time."""
    c = np.asarray(per_time_curl["curl"], dtype=float)
    k = np.asarray(per_time_curv["curvature"], dtype=float)
    t = np.asarray(per_time_curv["t"], dtype=float)
    n = min(len(c), len(k))
    if n == 0:
        c = k = t = np.zeros(1)
        n = 1
    c, k, t = c[:n], k[:n], t[:n]
    prod = c * k
    corr = float(np.corrcoef(c, k)[0, 1]) if n > 1 and c.std() > 0 and k.std() > 0 else 0.0
    topk = max(1, n // 5)
    idx_k = np.argsort(-k)[:topk]
    idx_c = np.argsort(-c)[:topk]
    peak_i = int(np.argmax(prod))
    feats = {
        "corr_curl_curvature": corr,
        "mean_curl_times_curvature": float(prod.mean()),
        "max_curl_times_curvature": float(prod.max()),
        "p95_curl_times_curvature": float(np.quantile(prod, 0.95)),
        "curl_at_top_curvature_timesteps": float(c[idx_k].mean()),
        "curvature_at_top_curl_timesteps": float(k[idx_c].mean()),
        "auc_product_over_time": float(np.trapz(prod, t if len(t) == len(prod) else None)),
        "peak_product_t": float(t[peak_i]),
    }
    t12 = feats["mean_curl_times_curvature"] + 0.2 * max(feats["corr_curl_curvature"], 0.0)
    return t12, feats
