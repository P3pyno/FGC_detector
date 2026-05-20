from __future__ import annotations
import numpy as np
import torch


def _norm(x: torch.Tensor) -> float:
    return float(torch.norm(x.flatten(), p=2).item())


def compute_d2_features(inversion: dict, eps: float = 1e-8):
    """Compute D2 trajectory straightness/curvature features from inverted ODE latents."""
    zs = inversion["latents"]
    ts = np.array(inversion["timesteps"])
    deltas = [zs[i + 1] - zs[i] for i in range(len(zs) - 1)]
    path_length = float(sum(_norm(d) for d in deltas))
    end_to_end = _norm(zs[-1] - zs[0])
    path_length_ratio = path_length / (end_to_end + eps)

    curvatures = []
    curv_t = []
    for k in range(1, len(zs) - 1):
        a = zs[k + 1] - 2 * zs[k] + zs[k - 1]
        curvatures.append(_norm(a))
        curv_t.append(ts[k])
    curv_arr = np.asarray(curvatures, dtype=float) if curvatures else np.zeros(1)

    angle_changes = []
    for k in range(len(deltas) - 1):
        a, b = deltas[k].flatten(), deltas[k + 1].flatten()
        c = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        angle_changes.append(1.0 - c)
    angle_arr = np.asarray(angle_changes) if angle_changes else np.zeros(1)

    def band_mean(lo, hi, include_lo=True):
        mask = (np.array(curv_t) >= lo if include_lo else np.array(curv_t) > lo) & (np.array(curv_t) <= hi)
        return float(curv_arr[mask].mean()) if mask.any() else 0.0

    peak_i = int(np.argmax(curv_arr)) if len(curv_arr) else 0
    d2 = {
        "path_length_ratio": path_length_ratio,
        "mean_curvature": float(curv_arr.mean()),
        "std_curvature": float(curv_arr.std()),
        "max_curvature": float(curv_arr.max()),
        "p95_curvature": float(np.quantile(curv_arr, 0.95)),
        "curvature_auc": float(np.trapz(curv_arr, dx=1.0 / max(len(curv_arr), 1))),
        "mean_angle_change": float(angle_arr.mean()),
        "max_angle_change": float(angle_arr.max()),
        "early_curvature_mean": band_mean(0.0, 0.33, include_lo=True),
        "middle_curvature_mean": band_mean(0.33, 0.66, include_lo=False),
        "late_curvature_mean": band_mean(0.66, 1.0, include_lo=False),
        "peak_curvature_t": float(np.array(curv_t)[peak_i]) if len(curv_t) else 0.0,
        "peak_curvature_value": float(curv_arr[peak_i]),
    }
    t2 = d2["mean_curvature"] + 0.5 * (d2["path_length_ratio"] - 1.0)
    return t2, d2, {"t": curv_t, "curvature": curv_arr.tolist()}
