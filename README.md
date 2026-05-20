# FGC_detector

## Flow-Geometric Consistency Detector D1 + D2

This prototype implements a **Flow-Geometric Consistency Detector** using a frozen surrogate flow-matching velocity field `v_theta`.

- **D1 (local curl / non-conservativity):** random curl probes with closed-loop circulation (plaquette integral), plus optional antisymmetric Jacobian probe language in docs.
- **D2 (global trajectory straightness):** ODE inversion from `t=1` to `t=0` and curvature/straightness diagnostics.
- **Time-resolved analysis:** computes separation and AUC vs time to discover best windows `t*`.
- **Closed-model surrogate testing:** supports probe model metadata independent from image generator (`midjourney`, `nano_banana`, `gpt_image`, etc.).

### Run feature extraction

```bash
python scripts/extract_flowgeom_features.py \
  --config configs/flowgeom_default.yaml \
  --output outputs/features.csv \
  --probe_model flux_surrogate
```

### Train detector

```bash
python scripts/train_flowgeom_detector.py \
  --features outputs/features.csv \
  --output_dir outputs/
```

### Evaluate detector

```bash
python scripts/evaluate_flowgeom_detector.py \
  --features outputs/features.csv \
  --model outputs/detector.pkl \
  --calibrator outputs/calibrator.pkl \
  --output_dir outputs/
```

### Run ablation

```bash
python scripts/run_ablation.py \
  --features outputs/features.csv \
  --output_dir outputs/
```

Outputs are saved under `outputs/`, including `features.csv`, `ablation_results.csv`, `metrics_by_generator.csv`, time-resolved CSVs, and plots under `outputs/plots/`.
