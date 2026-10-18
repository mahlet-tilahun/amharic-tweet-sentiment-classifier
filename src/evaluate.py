"""
Step 6: evaluation and error analysis of the final model (and comparison with the baseline).

Uses the prediction files written by the training scripts, so nothing is retrained here.

Outputs (in results/):
  figures/experiments.png          dev vs test macro-F1 of every experiment
  figures/confusion_<name>.png      confusion matrices (final model and best baseline)
  figures/training_curve.png        loss / dev F1 per epoch of the final model
  error_analysis.json               accuracy broken down by tweet properties
  errors_<name>.csv                 misclassified test tweets, most confident mistakes first

Run:  python src/evaluate.py --final T2_bert_medium_amharic
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import LABELS, RESULTS_DIR, clean_text, full_report, load_split

parser = argparse.ArgumentParser()
parser.add_argument("--final", required=True, help="experiment name of the final model")
parser.add_argument("--baseline", default="baseline")
args = parser.parse_args()
FIG = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG, exist_ok=True)

# ---- 1. experiment comparison chart --------------------------------------------------
exp = pd.read_csv(os.path.join(RESULTS_DIR, "experiments.csv"))
exp = exp.sort_values("experiment")
ax = exp.set_index("experiment")[["dev_macro_f1", "test_macro_f1"]].plot.barh(
    figsize=(8, 0.45 * len(exp) + 1.5), color=["#5b8def", "#f0a04b"])
ax.set_xlabel("macro-F1")
ax.set_title("All experiments: dev vs test macro-F1")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "experiments.png"), dpi=150)
plt.close()


# ---- 2. confusion matrices ---------------------------------------------------------------
def confusion_plot(pred_file, title, out_name):
    df = pd.read_csv(os.path.join(RESULTS_DIR, pred_file))
    report, cm = full_report(df["label"], df["pred"])
    plt.figure(figsize=(4.2, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, out_name), dpi=150)
    plt.close()
    return df, report, cm


final_df, final_report, final_cm = confusion_plot(
    f"predictions_{args.final}.csv", f"Final model ({args.final}) - test", "confusion_final.png")
_, base_report, base_cm = confusion_plot(
    f"predictions_{args.baseline}.csv", "Best baseline - test", "confusion_baseline.png")

# ---- 3. training curve ---------------------------------------------------------------------
hist_path = os.path.join(RESULTS_DIR, f"{args.final}_history.csv")
if os.path.exists(hist_path):
    hist = pd.read_csv(hist_path)
    fig, ax1 = plt.subplots(figsize=(5, 3.2))
    ax1.plot(hist["epoch"], hist["train_loss"], "o-", color="#c0392b", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss")
    ax2 = ax1.twinx()
    ax2.plot(hist["epoch"], hist["dev_macro_f1"], "s-", color="#2980b9", label="dev macro-F1")
    ax2.set_ylabel("dev macro-F1")
    fig.legend(loc="upper center", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "training_curve.png"), dpi=150)
    plt.close()

# ---- 4. error analysis: where do the models fail? ---------------------------------------------
train_vocab = {w for t in load_split("train")["tweet"] for w in clean_text(t).split()}
prob_cols = [f"p_{label}" for label in LABELS]


def add_features(pred_df):
    """Add columns describing each test tweet, so accuracy can be broken down by them."""
    d = pred_df.copy()
    d["clean"] = d["tweet"].apply(clean_text)
    d["correct"] = d["label"] == d["pred"]
    d["n_words"] = d["clean"].str.split().str.len()
    d["oov_ratio"] = d["clean"].apply(
        lambda t: sum(w not in train_vocab for w in t.split()) / max(1, len(t.split())))
    d["has_latin"] = d["clean"].str.contains(r"[a-z]{2,}")
    if all(c in d for c in prob_cols):
        d["confidence"] = d[prob_cols].max(axis=1)
    return d


def acc_by(d, column, bins=None, labels=None):
    groups = pd.cut(d[column], bins=bins, labels=labels) if bins else d[column]
    out = d.groupby(groups, observed=True)["correct"].agg(["mean", "count"])
    return {str(k): {"accuracy": round(v["mean"], 3), "n": int(v["count"])} for k, v in out.iterrows()}


def breakdown(d):
    result = {
        "most_common_errors (true -> pred)": {
            f"{t} -> {p}": int(n) for (t, p), n in
            d[~d["correct"]].groupby(["label", "pred"]).size().sort_values(ascending=False).items()},
        "accuracy_by_length": acc_by(d, "n_words", [0, 8, 15, 25, 100], ["1-8", "9-15", "16-25", "26+"]),
        "accuracy_by_oov_ratio": acc_by(d, "oov_ratio", [-0.01, 0.2, 0.4, 1.0], ["0-20%", "20-40%", "40%+"]),
        "accuracy_by_latin_words": acc_by(d, "has_latin"),
        "prediction_distribution_test": d["pred"].value_counts().to_dict(),
    }
    if "confidence" in d:
        result["accuracy_by_confidence"] = acc_by(d, "confidence", [0, 0.5, 0.7, 0.9, 1.0],
                                                  ["<0.5", "0.5-0.7", "0.7-0.9", ">0.9"])
    return result


df = add_features(final_df)
base_df = add_features(pd.read_csv(os.path.join(RESULTS_DIR, f"predictions_{args.baseline}.csv")))
per_class = lambda rep: {k: {m: round(v[m], 3) for m in ["precision", "recall", "f1-score"]}
                         for k, v in rep.items() if k in LABELS}
analysis = {
    "final_model": args.final,
    "true_distribution_test": df["label"].value_counts().to_dict(),
    "final": {"per_class": per_class(final_report),
              "confusion_matrix (rows=true neg/neu/pos, cols=pred)": final_cm, **breakdown(df)},
    "baseline": {"per_class": per_class(base_report),
                 "confusion_matrix (rows=true neg/neu/pos, cols=pred)": base_cm, **breakdown(base_df)},
}

with open(os.path.join(RESULTS_DIR, "error_analysis.json"), "w", encoding="utf-8") as f:
    json.dump(analysis, f, indent=2, ensure_ascii=False)
print(json.dumps(analysis, indent=2, ensure_ascii=False))

# misclassified tweets, most confident mistakes first (these are the most interesting to read)
errors = df[~df["correct"]]
if "confidence" in errors:
    errors = errors.sort_values("confidence", ascending=False)
errors[["tweet", "label", "pred"] + [c for c in prob_cols + ["confidence"] if c in errors]].to_csv(
    os.path.join(RESULTS_DIR, f"errors_{args.final}.csv"), index=False)
print(f"\n{len(errors)} misclassified test tweets saved to results/errors_{args.final}.csv")
