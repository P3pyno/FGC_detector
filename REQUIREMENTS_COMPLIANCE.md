# FlowGeom Prompt Compliance Matrix

This matrix maps the original 20-point prompt to current implementation status.

Legend: ✅ complete, ⚠️ partial, ❌ missing.

1. Data loading (folder/CSV + metadata): ✅ (`flowgeom/data.py`)
2. VAE adapter identity + diffusers fallback: ✅ (`flowgeom/vae_adapter.py`)
3. Velocity adapters incl dummy + diffusers skeleton: ✅ (`flowgeom/velocity_adapter.py`)
4. D2 inversion and curvature/straightness features: ✅ (`flowgeom/ode_inversion.py`, `flowgeom/d2_trajectory.py`)
5. D1 local curl via plaquette loop (optional antisymmetric probe noted): ⚠️ (plaquette implemented; JVP/VJP experimental path not implemented)
6. Time-resolved analysis tables and best t*: ✅ (`flowgeom/feature_extraction.py`)
7. D1-D2 coupling features: ✅ (`flowgeom/coupling.py`)
8. Real-null calibration: ✅ (`flowgeom/calibration.py`, `scripts/train_flowgeom_detector.py`)
9. Joint score + manual/learned weights + p-value: ⚠️ (learned/logistic + p-value implemented; manual weighted scoring path not fully wired in CLI)
10. Closed-model surrogate grouped metrics: ✅ (`scripts/evaluate_flowgeom_detector.py`)
11. Ablation study variants: ✅ (`scripts/run_ablation.py`)
12. Plot set A-H: ⚠️ (core set now present; some formatting/detail differences)
13. Scripts and args: ⚠️ (scripts exist; a few config/arg interactions still simplified)
14. Config default structure: ✅ (`configs/flowgeom_default.yaml`)
15. Tests A-E: ✅ (`tests/`)
16. Coding style/libs/docstrings: ⚠️ (libs/style okay; docstrings still not exhaustive in every module)
17. `features.csv` required columns: ⚠️ (most present; `probability_generated` appears post-eval not extraction)
18. First-run behavior fallback/warnings: ✅ (synthetic dataset + dummy adapter warning)
19. README section requested: ✅ (`README.md`)
20. Conceptual correction language (no Hutchinson trace misuse): ✅ (D1 descriptions use curl/probe/circulation language)

## Remaining work for strict 100%
- Implement experimental antisymmetric Jacobian JVP/VJP probe path in `d1_curl.py`.
- Wire explicit manual weighted scoring mode from config alongside learned mode in train/eval CLI.
- Enforce/export every required `features.csv` column contract in extraction stage and add schema checks.
- Extend docstrings in all modules to full research-note level.
- Add stricter split-aware evaluation controls and richer per-time AUC plot annotations.
