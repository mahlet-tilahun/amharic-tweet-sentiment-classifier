"""
Step 7 (extra experiment): does correcting for LABEL SHIFT explain the drop from dev to test?

Problem: the model learned from train that ~52% of tweets are neutral. The test set is ~67% negative.
A classifier's probabilities contain this train prior: p(y|x) is proportional to p(x|y) * p_train(y).


"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from utils import LABELS, RESULTS_DIR, compute_metrics, load_split

parser = argparse.ArgumentParser()
parser.add_argument("--pred", nargs="+", required=True, help="prediction csv files with p_<label> columns")
parser.add_argument("--prior", choices=["uniform", "train"], required=True,
                    help="uniform for class-weighted models, train for unweighted models")
args = parser.parse_args()

if args.prior == "uniform":
    pi_train = np.ones(len(LABELS)) / len(LABELS)
else:
    pi_train = load_split("train")["label"].value_counts(normalize=True).reindex(LABELS).values


def em_prior_adjustment(probs, pi_train, n_iter=100):
    pi_test = pi_train.copy()
    for _ in range(n_iter):
        adjusted = probs * (pi_test / pi_train)
        adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
        new_pi = adjusted.mean(axis=0)
        if np.abs(new_pi - pi_test).max() < 1e-6:
            break
        pi_test = new_pi
    return adjusted, pi_test


out_path = os.path.join(RESULTS_DIR, "label_shift.json")
out = json.load(open(out_path)) if os.path.exists(out_path) else {}
for path in args.pred:
    df = pd.read_csv(path)
    probs = df[[f"p_{label}" for label in LABELS]].values
    adjusted, pi_est = em_prior_adjustment(probs, pi_train)
    new_pred = [LABELS[i] for i in adjusted.argmax(axis=1)]
    name = os.path.basename(path).replace("predictions_", "").replace(".csv", "")
    out[name] = {
        "source_prior": dict(zip(LABELS, pi_train.round(3))),
        "estimated_test_prior": dict(zip(LABELS, pi_est.round(3))),
        "true_test_prior": df["label"].value_counts(normalize=True).reindex(LABELS).round(3).to_dict(),
        "test_before": compute_metrics(df["label"], df["pred"]),
        "test_after": compute_metrics(df["label"], new_pred),
    }
    print(name, json.dumps(out[name], indent=2, default=float))

with open(out_path, "w") as f:
    json.dump(out, f, indent=2, default=float)
