#!/usr/bin/env python
from __future__ import annotations

import argparse

from flowgeom.config import load_config
from flowgeom.data import load_dataset
from flowgeom.feature_extraction import ExtractSettings, compute_time_resolved_table, extract_features
from flowgeom.plots import plot_time_resolved, save_score_histograms, scatter_curvature_curl
from flowgeom.utils import ensure_dir, set_seed
from flowgeom.vae_adapter import VAEAdapter
from flowgeom.velocity_adapter import DiffusersFlowVelocityAdapter, DummyCurvedVelocityAdapter, DummyLinearVelocityAdapter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract Flow-Geometric Consistency features")
    p.add_argument("--config", default="configs/flowgeom_default.yaml")
    p.add_argument("--data_csv", default=None)
    p.add_argument("--image_root", default=None)
    p.add_argument("--output", default="outputs/features.csv")
    p.add_argument("--probe_model", default="flux_surrogate")
    p.add_argument("--vae_mode", default="identity", choices=["identity", "diffusers_vae"])
    p.add_argument("--num_steps", type=int, default=30)
    p.add_argument("--num_curl_probes", type=int, default=16)
    p.add_argument("--eps", type=float, default=0.01)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)
    ensure_dir("outputs")
    ensure_dir("outputs/plots")

    ds = load_dataset(args.data_csv, args.image_root, image_size=cfg["data"]["image_size"])
    vae = VAEAdapter(mode=args.vae_mode, device=args.device)
    va = cfg["probe"]["velocity_adapter"]
    vel = DummyLinearVelocityAdapter(args.probe_model) if va == "dummy_linear" else (DiffusersFlowVelocityAdapter(args.probe_model) if va == "diffusers_flow" else DummyCurvedVelocityAdapter(args.probe_model))

    settings = ExtractSettings(args.num_steps, cfg["ode"]["solver"], args.num_curl_probes, args.eps, args.probe_model, args.max_images)
    df = extract_features(ds, vae, vel, settings)
    df.to_csv(args.output, index=False)
    ctab, ktab, best = compute_time_resolved_table(df, "outputs")
    save_score_histograms(df, "outputs/plots")
    plot_time_resolved(ctab, ktab, "outputs/plots", best)
    scatter_curvature_curl(df, "outputs/plots")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
