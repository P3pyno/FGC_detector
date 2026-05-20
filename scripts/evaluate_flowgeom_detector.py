#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from flowgeom.calibration import RobustCalibrator
from flowgeom.classifier import compute_eer, p_value_under_real_null, tpr_at_fpr
from flowgeom.plots import grouped_bar
from flowgeom.utils import ensure_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="outputs/features.csv")
    p.add_argument("--model", default="outputs/detector.pkl")
    p.add_argument("--calibrator", default="outputs/calibrator.pkl")
    p.add_argument("--output_dir", default="outputs/")
    args = p.parse_args()

    ensure_dir(args.output_dir)
    ensure_dir(f"{args.output_dir}/plots")

    df = pd.read_csv(args.features)
    with open(f"{args.output_dir}/detector_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    threshold = float(meta.get("threshold", 0.5))

    cal = RobustCalibrator.load(args.calibrator)
    with open(args.model, "rb") as f:
        clf = pickle.load(f)

    x = cal.transform(df[feature_cols].values, feature_cols)
    proba = clf.predict_proba(x)
    df["probability_generated"] = proba
    df["decision"] = (df["probability_generated"] >= threshold).astype(int)

    null_path = f"{args.output_dir}/real_null_scores.csv"
    null_scores = pd.read_csv(null_path)["real_null_scores"].values if pd.io.common.file_exists(null_path) else proba[df["label"] == 0]
    df["p_value_under_real_null"] = [p_value_under_real_null(null_scores, s) for s in df["probability_generated"].values]

    rows = []
    for g, grp in df.groupby("generator_name"):
        yg = grp.label.values
        sg = grp.probability_generated.values
        pg = grp.decision.values
        auc = roc_auc_score(yg, sg) if len(np.unique(yg)) > 1 else np.nan
        rows.append(
            {
                "generator_name": g,
                "auc": auc,
                "accuracy": accuracy_score(yg, pg),
                "f1": f1_score(yg, pg, zero_division=0),
                "precision": precision_score(yg, pg, zero_division=0),
                "recall": recall_score(yg, pg, zero_division=0),
                "tpr_fpr_1": tpr_at_fpr(yg, sg, 0.01),
                "tpr_fpr_5": tpr_at_fpr(yg, sg, 0.05),
                "eer": compute_eer(yg, sg),
            }
        )

    mdf = pd.DataFrame(rows)
    mdf.to_csv(f"{args.output_dir}/metrics_by_generator.csv", index=False)
    df.to_csv(f"{args.output_dir}/predictions.csv", index=False)
    grouped_bar(mdf, f"{args.output_dir}/plots")
    print("saved metrics_by_generator.csv and predictions.csv")


if __name__ == "__main__":
    main()
