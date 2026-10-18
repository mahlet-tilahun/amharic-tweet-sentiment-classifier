"""
Step 2: explore the dataset before training anything.

Prints and saves:
- number of tweets and class distribution for each split
- tweet length (in words)
- duplicates and train/test overlap
- how many tweets contain Latin letters or emojis
- vocabulary size and out-of-vocabulary (OOV) rate, with and without my Amharic normalisation

Run:  python src/explore_data.py
"""
import json
import os
import re
from collections import Counter

import matplotlib.pyplot as plt

from utils import EMOJI_RE, LABELS, RESULTS_DIR, clean_text, load_split

os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
splits = {name: load_split(name) for name in ["train", "dev", "test"]}
stats = {}

# ---- 1. size and class distribution -----------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=False)
for ax, (name, df) in zip(axes, splits.items()):
    counts = df["label"].value_counts().reindex(LABELS)
    stats[name] = {
        "n_tweets": len(df),
        "class_counts": counts.to_dict(),
        "class_percent": (counts / len(df) * 100).round(1).to_dict(),
    }
    ax.bar(LABELS, counts.values, color=["#c0392b", "#7f8c8d", "#27ae60"])
    ax.set_title(f"{name} (n={len(df)})")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
fig.suptitle("Class distribution of the Amharic AfriSenti splits")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "figures", "class_distribution.png"), dpi=150)

# ---- 2. tweet length ----------------------------------------------------------
for name, df in splits.items():
    lengths = df["tweet"].apply(lambda t: len(clean_text(t).split()))
    stats[name]["words_mean"] = round(lengths.mean(), 1)
    stats[name]["words_median"] = int(lengths.median())
    stats[name]["words_max"] = int(lengths.max())

# ---- 3. duplicates and overlap -----------------------------------------------
train_texts = set(splits["train"]["tweet"].apply(clean_text))
for name, df in splits.items():
    cleaned = df["tweet"].apply(clean_text)
    stats[name]["duplicates_within_split"] = int(cleaned.duplicated().sum())
    if name != "train":
        stats[name]["also_in_train"] = int(cleaned.isin(train_texts).sum())

# duplicated tweets in train that have different labels (label noise)
tr = splits["train"].assign(clean=splits["train"]["tweet"].apply(clean_text))
conflicts = tr.groupby("clean")["label"].nunique()
stats["train"]["duplicate_texts_with_conflicting_labels"] = int((conflicts > 1).sum())

# ---- 4. script mix: Latin letters and emojis -----------------------------------
for name, df in splits.items():
    cleaned = df["tweet"].apply(clean_text)  # cleaned, so "@user" is not counted as Latin text
    stats[name]["percent_with_latin_words"] = round(cleaned.str.contains(r"[a-z]{2,}").mean() * 100, 1)
    stats[name]["percent_with_emoji"] = round(df["tweet"].apply(lambda t: bool(EMOJI_RE.search(t))).mean() * 100, 1)
    stats[name]["percent_with_@user"] = round(df["tweet"].str.contains("@user").mean() * 100, 1)

# ---- 5. vocabulary and OOV rate (with / without normalisation) --------------
for normalise in [False, True]:
    key = "normalised" if normalise else "raw"
    train_tokens = [w for t in splits["train"]["tweet"] for w in clean_text(t, normalise).split()]
    vocab = set(train_tokens)
    stats[f"vocab_size_train_{key}"] = len(vocab)
    stats[f"singleton_words_train_{key}"] = sum(1 for c in Counter(train_tokens).values() if c == 1)
    for name in ["dev", "test"]:
        tokens = [w for t in splits[name]["tweet"] for w in clean_text(t, normalise).split()]
        oov = sum(1 for w in tokens if w not in vocab) / len(tokens)
        stats[f"oov_rate_{name}_{key}"] = round(oov * 100, 1)

with open(os.path.join(RESULTS_DIR, "data_stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)
print(json.dumps(stats, indent=2, ensure_ascii=False))

# ---- 6. a few example tweets per class, for the report -------------------------
print("\nExamples:")
for label in LABELS:
    for t in splits["train"][splits["train"]["label"] == label]["tweet"].head(2):
        print(f"  [{label}] {t}")
