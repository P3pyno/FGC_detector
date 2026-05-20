from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def save_score_histograms(df: pd.DataFrame, outdir: str):
    mapping = [
        ("T1_curl_score", "hist_t1_curl_real_vs_fake.png"),
        ("T2_straightness_score", "hist_t2_straightness_real_vs_fake.png"),
        ("joint_score", "hist_joint_score_real_vs_fake.png"),
    ]
    for col, name in mapping:
        plt.figure()
        for lbl, grp in df.groupby("label"):
            plt.hist(grp[col], bins=20, alpha=0.5, label=f"label={lbl}")
        plt.legend()
        plt.title(col)
        plt.tight_layout()
        plt.savefig(f"{outdir}/{name}")
        plt.close()


def plot_time_resolved(ctab: pd.DataFrame, ktab: pd.DataFrame, outdir: str, best: dict):
    plt.figure()
    plt.plot(ctab.t, ctab.real_mean, label="real")
    plt.plot(ctab.t, ctab.fake_mean, label="fake")
    plt.axvline(best.get("best_curl_t_star", 0.0), color="k", linestyle="--")
    plt.title("curl vs t")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/curl_vs_t_real_vs_fake.png")
    plt.close()

    plt.figure()
    plt.plot(ktab.t, ktab.real_mean, label="real")
    plt.plot(ktab.t, ktab.fake_mean, label="fake")
    plt.axvline(best.get("best_curvature_t_star", 0.0), color="k", linestyle="--")
    plt.title("curvature vs t")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/curvature_vs_t_real_vs_fake.png")
    plt.close()

    plt.figure()
    plt.plot(ctab.t, ctab.auc, label="curl AUC")
    plt.plot(ktab.t, ktab.auc, label="curvature AUC")
    plt.title("AUC vs t")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/auc_vs_t.png")
    plt.close()


def scatter_curvature_curl(df: pd.DataFrame, outdir: str):
    plt.figure()
    plt.scatter(df["mean_curvature"], df["curl_mean"], c=df["label"], alpha=0.7)
    plt.xlabel("mean_curvature")
    plt.ylabel("curl_mean")
    plt.tight_layout()
    plt.savefig(f"{outdir}/scatter_curvature_vs_curl.png")
    plt.close()


def grouped_bar(metrics_df: pd.DataFrame, outdir: str):
    plot_cols = [c for c in ["auc", "accuracy", "f1", "precision", "recall"] if c in metrics_df.columns]
    for col in plot_cols:
        plt.figure(figsize=(8, 4))
        plt.bar(metrics_df["generator_name"], metrics_df[col].fillna(0))
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{col} by generator")
        plt.tight_layout()
        plt.savefig(f"{outdir}/grouped_{col}_by_generator.png")
        plt.close()
