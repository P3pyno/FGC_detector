#!/usr/bin/env python
from __future__ import annotations
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

p=argparse.ArgumentParser();p.add_argument("--features",default="outputs/features.csv");p.add_argument("--output_dir",default="outputs/");args=p.parse_args()
df=pd.read_csv(args.features)
configs={
"D1 only":["T1_curl_score"],
"D2 only":["T2_straightness_score"],
"D1 + D2":["T1_curl_score","T2_straightness_score"],
"D1 + D2 + coupling":["T1_curl_score","T2_straightness_score","T12_coupling_score"],
"D1 time-best only":["peak_curl_value"],
"D2 time-best only":["peak_curvature_value"],
"all features":["T1_curl_score","T2_straightness_score","T12_coupling_score","curl_auc","curvature_auc","corr_curl_curvature"]}
train=df[df.split.isin(["train","val"])]; test=df[df.split=="test"] if (df.split=="test").any() else df
rows=[]
for name,cols in configs.items():
    clf=LogisticRegression(max_iter=1000).fit(train[cols].values,train.label.values)
    score=clf.predict_proba(test[cols].values)[:,1]
    auc=roc_auc_score(test.label.values,score) if test.label.nunique()>1 else float("nan")
    rows.append({"ablation":name,"auc":auc,"n_features":len(cols)})
pd.DataFrame(rows).to_csv(f"{args.output_dir}/ablation_results.csv",index=False)
print("saved ablation_results.csv")
