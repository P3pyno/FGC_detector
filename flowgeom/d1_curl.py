from __future__ import annotations
import numpy as np
import torch


def _random_unit_like(z):
    u = torch.randn_like(z)
    n = torch.norm(u.flatten(), p=2) + 1e-12
    return u / n


def plaquette_circulation(z, t, velocity_model, eps=0.01):
    """Closed-loop circulation estimate via finite-difference plaquette integral."""
    u = _random_unit_like(z)
    w = _random_unit_like(z)
    p0 = z
    p1 = z + eps * u
    p2 = z + eps * u + eps * w
    p3 = z + eps * w
    e0, e1, e2, e3 = eps * u, eps * w, -eps * u, -eps * w
    c = (
        torch.dot(velocity_model.velocity(p0, t).flatten(), e0.flatten())
        + torch.dot(velocity_model.velocity(p1, t).flatten(), e1.flatten())
        + torch.dot(velocity_model.velocity(p2, t).flatten(), e2.flatten())
        + torch.dot(velocity_model.velocity(p3, t).flatten(), e3.flatten())
    ) / (eps * eps + 1e-12)
    return float(c.item())


def compute_d1_features(latents, timesteps, velocity_model, num_probes=16, eps=0.01):
    """Compute D1 local non-conservativity using random curl probes over time."""
    per_time = []
    for z, t in zip(latents, timesteps):
        vals = [abs(plaquette_circulation(z, t, velocity_model, eps=eps)) for _ in range(num_probes)]
        arr = np.asarray(vals)
        per_time.append({"t": float(t), "mean": float(arr.mean()), "std": float(arr.std()), "max": float(arr.max()), "p95": float(np.quantile(arr, 0.95))})

    mean_series = np.array([x["mean"] for x in per_time])
    t_series = np.array([x["t"] for x in per_time])
    peak = int(np.argmax(mean_series))
    d1 = {
        "curl_mean": float(mean_series.mean()),
        "curl_std": float(mean_series.std()),
        "curl_max": float(mean_series.max()),
        "curl_p95": float(np.quantile(mean_series, 0.95)),
        "curl_auc": float(np.trapz(mean_series, t_series)),
        "early_curl_mean": float(mean_series[t_series <= 0.33].mean()) if (t_series <= 0.33).any() else 0.0,
        "middle_curl_mean": float(mean_series[(t_series > 0.33) & (t_series <= 0.66)].mean()) if ((t_series > 0.33) & (t_series <= 0.66)).any() else 0.0,
        "late_curl_mean": float(mean_series[t_series > 0.66].mean()) if (t_series > 0.66).any() else 0.0,
        "peak_curl_t": float(t_series[peak]),
        "peak_curl_value": float(mean_series[peak]),
    }
    t1 = d1["curl_mean"]
    return t1, d1, {"t": t_series.tolist(), "curl": mean_series.tolist()}
