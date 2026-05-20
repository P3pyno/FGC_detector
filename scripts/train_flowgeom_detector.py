#!/usr/bin/env python
from __future__ import annotations

import argparse
import pickle

import pandas as pd
from sklearn.metrics import roc_curve

from flowgeom.calibration import RobustCalibrator
from flowgeom.classifier import FlowGeomClassifier, compute_joint_score
from flowgeom.utils import ensure_dir, save_json


FEATURE_COLS = ["T1_curl_score", "T2_straightness_score", "T12_coupling_score"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="outputs/features.csv")
    p.add_argument("--output_dir", default="outputs/")
    p.add_argument("--label_col", default="label")
    p.add_argument("--split_col", default="split")
    args = p.parse_args()

    ensure_dir(args.output_dir)
    df = pd.read_csv(args.features)
    df["joint_score"] = compute_joint_score(df)

    val_real = df[(df[args.split_col] == "val") & (df[args.label_col] == 0)]
    real_for_null = val_real if len(val_real) else df[df[args.label_col] == 0]
    cal = RobustCalibrator().fit(real_for_null[FEATURE_COLS].values, FEATURE_COLS)
    cal.save(f"{args.output_dir}/calibrator.pkl")

    train = df[df[args.split_col].isin(["train", "val"])]
    clf = FlowGeomClassifier()
    x_train = cal.transform(train[FEATURE_COLS].values, FEATURE_COLS)
    y_train = train[args.label_col].values
    clf.fit(x_train, y_train)

    with open(f"{args.output_dir}/detector.pkl", "wb") as f:
        pickle.dump(clf, f)

    val = df[df[args.split_col] == "val"]
    if len(val) and val[args.label_col].nunique() > 1:
        val_scores = clf.predict_proba(cal.transform(val[FEATURE_COLS].values, FEATURE_COLS))
        fpr, tpr, thr = roc_curve(val[args.label_col].values, val_scores)
        youden = tpr - fpr
        best_idx = int(youden.argmax())
        threshold = float(thr[best_idx])
    else:
        threshold = 0.5

    save_json({"threshold": threshold, "feature_cols": FEATURE_COLS}, f"{args.output_dir}/detector_meta.json")
    real_scores = clf.predict_proba(cal.transform(real_for_null[FEATURE_COLS].values, FEATURE_COLS))
    pd.DataFrame({"real_null_scores": real_scores}).to_csv(f"{args.output_dir}/real_null_scores.csv", index=False)
    print("saved detector.pkl calibrator.pkl detector_meta.json real_null_scores.csv")


if __name__ == "__main__":
    main()
