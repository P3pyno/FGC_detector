from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from .coupling import compute_coupling_features
from .d1_curl import compute_d1_features
from .d2_trajectory import compute_d2_features
from .ode_inversion import invert_trajectory
from .utils import save_json


@dataclass
class ExtractSettings:
    num_steps: int = 30
    solver: str = "euler"
    num_curl_probes: int = 16
    eps: float = 0.01
    probe_model_name: str = "flux_surrogate"
    max_images: int | None = None


def extract_features(dataset, vae, velocity_model, settings: ExtractSettings) -> pd.DataFrame:
    """Extract D1/D2/coupling features with per-time series for time-resolved analysis."""
    rows = []
    n = len(dataset) if settings.max_images is None else min(len(dataset), settings.max_images)
    for i in tqdm(range(n), desc="extract_flowgeom_features"):
        x, meta = dataset[i]
        z1 = vae.encode_image(x)
        inv = invert_trajectory(z1, velocity_model, num_steps=settings.num_steps, solver=settings.solver)
        t2, d2, curv = compute_d2_features(inv)
        t1, d1, curl = compute_d1_features(inv["latents"], inv["timesteps"], velocity_model, num_probes=settings.num_curl_probes, eps=settings.eps)
        t12, coup = compute_coupling_features(curl, curv)
        row = {
            **meta,
            "probe_model_name": settings.probe_model_name,
            "T1_curl_score": t1,
            "T2_straightness_score": t2,
            "T12_coupling_score": t12,
            "curl_series": json.dumps(curl),
            "curvature_series": json.dumps(curv),
            **d1,
            **d2,
            **coup,
        }
        row["joint_score"] = t1 + t2 + t12
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_time_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for t, g in df.groupby("t"):
        real = g[g.label == 0]["value"]
        fake = g[g.label == 1]["value"]
        auc = roc_auc_score(g["label"], g["value"]) if g["label"].nunique() > 1 else np.nan
        out.append(
            {
                "t": t,
                "real_mean": real.mean() if len(real) else np.nan,
                "fake_mean": fake.mean() if len(fake) else np.nan,
                "separation": (fake.mean() - real.mean()) if len(real) and len(fake) else np.nan,
                "auc": auc,
            }
        )
    return pd.DataFrame(out).sort_values("t")


def compute_time_resolved_table(features_df: pd.DataFrame, output_dir: str):
    """Compute per-time real/fake separation for curl and curvature plus best t* windows."""
    curl_rows, curv_rows = [], []
    for _, r in features_df.iterrows():
        cs, ks = json.loads(r["curl_series"]), json.loads(r["curvature_series"])
        curl_rows.extend({"t": t, "value": v, "label": r["label"]} for t, v in zip(cs["t"], cs["curl"]))
        curv_rows.extend({"t": t, "value": v, "label": r["label"]} for t, v in zip(ks["t"], ks["curvature"]))
    ctab = _aggregate_time_rows(pd.DataFrame(curl_rows))
    ktab = _aggregate_time_rows(pd.DataFrame(curv_rows))
    ctab.to_csv(f"{output_dir}/time_resolved_curl.csv", index=False)
    ktab.to_csv(f"{output_dir}/time_resolved_curvature.csv", index=False)
    best = {
        "best_curl_t_star": float(ctab.loc[ctab["auc"].fillna(-1).idxmax(), "t"]) if len(ctab) else 0.0,
        "best_curvature_t_star": float(ktab.loc[ktab["auc"].fillna(-1).idxmax(), "t"]) if len(ktab) else 0.0,
    }
    save_json(best, f"{output_dir}/best_time_windows.json")
    return ctab, ktab, best
