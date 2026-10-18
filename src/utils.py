"""
Shared helper functions used by all the training scripts.

- load_split():   reads one of the AfriSenti Amharic files (train / dev / test)
- clean_text():   my preprocessing for Amharic tweets
- compute_metrics(): accuracy, precision, recall and F1 in one place
- save_results(): writes the metrics to results/ so the report tables come from real runs
"""
import json
import os
import random
import re

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_score, recall_score)

# Folders (relative to the project root)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "raw")
RESULTS_DIR = os.path.join(ROOT, "results")
MODELS_DIR = os.path.join(ROOT, "models")

# The three sentiment classes. The order is fixed so that label ids are always the same.
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

SEED = 42


def set_seed(seed=SEED):
    """Make runs repeatable (same random numbers every time)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def load_split(split):
    """Load 'train', 'dev' or 'test' as a pandas DataFrame with columns tweet, label."""
    path = os.path.join(DATA_DIR, f"{split}.tsv")
    df = pd.read_csv(path, sep="\t", quoting=3)  # quoting=3 -> don't treat " as special
    df = df.dropna()
    df["label"] = df["label"].str.strip()
    return df


# ---------------------------------------------------------------------------
# Amharic text normalisation
# ---------------------------------------------------------------------------
# Amharic (Ge'ez / Fidel script) has several letters that sound the same but are
# written differently, e.g. ሀ / ሐ / ኀ are all pronounced "ha". People on Twitter use
# them interchangeably, so the same word can appear with different spellings.
# Mapping them to one form reduces the number of different spellings the model sees.
# Each Fidel "family" occupies 8 Unicode code points (one per vowel order), so I
# map the first 7 orders of each family to the matching order of the main family.
HOMOPHONE_FAMILIES = [
    (0x1210, 0x1200),  # ሐ ሑ ሒ ... -> ሀ ሁ ሂ ...
    (0x1280, 0x1200),  # ኀ ኁ ኂ ... -> ሀ ሁ ሂ ...
    (0x1220, 0x1230),  # ሠ ሡ ሢ ... -> ሰ ሱ ሲ ...
    (0x12D0, 0x12A0),  # ዐ ዑ ዒ ... -> አ ኡ ኢ ...
    (0x1340, 0x1338),  # ፀ ፁ ፂ ... -> ጸ ጹ ጺ ...
]
NORMALISE_MAP = {}
for src_start, dst_start in HOMOPHONE_FAMILIES:
    for order in range(7):
        NORMALISE_MAP[chr(src_start + order)] = chr(dst_start + order)
NORMALISE_TABLE = str.maketrans(NORMALISE_MAP)


def normalise_amharic(text):
    """Replace homophone Fidel characters with a single standard form."""
    return text.translate(NORMALISE_TABLE)


# Emojis and other pictographic symbols. The AfriSenti Amharic TEST file has almost no
# emojis (they seem to have been stripped), while ~30% of train tweets contain them.
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF←-⯿️‍]")


def clean_text(text, normalise=True, remove_emoji=False):
    """Preprocessing applied to every tweet (train, dev, test and in the web app)."""
    text = str(text).strip()
    text = text.strip("'\"")                        # test file wraps tweets in quotes
    text = re.sub(r"@\w+", " ", text)               # remove @user mentions (anonymised anyway)
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # remove links
    text = re.sub(r"#", " ", text)                  # keep hashtag words, drop the # sign
    text = re.sub(r"[።፣፤፥፦፧፨]", " ", text)          # Ethiopic punctuation -> space
    text = re.sub(r"([!?.,])\1+", r"\1", text)      # "!!!!" -> "!"
    text = text.lower()                             # only affects Latin letters
    if normalise:
        text = normalise_amharic(text)
    if remove_emoji:
        text = EMOJI_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()        # collapse spaces
    return text


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    """Return the metrics I report for every experiment."""
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "macro_recall": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }


def full_report(y_true, y_pred):
    """Per-class precision/recall/F1 and the confusion matrix (rows = true, cols = predicted)."""
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    return report, cm.tolist()


def save_results(name, config, dev_metrics, test_metrics):
    """Append one row to results/experiments.csv and save a json with details."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    row = {"experiment": name, **config}
    row.update({f"dev_{k}": v for k, v in dev_metrics.items()})
    row.update({f"test_{k}": v for k, v in test_metrics.items()})

    csv_path = os.path.join(RESULTS_DIR, "experiments.csv")
    new = pd.DataFrame([row])
    if os.path.exists(csv_path):
        old = pd.read_csv(csv_path)
        old = old[old["experiment"] != name]  # re-running an experiment replaces its row
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(csv_path, index=False)

    with open(os.path.join(RESULTS_DIR, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2, ensure_ascii=False)
    print(f"[{name}] dev macro-F1={dev_metrics['macro_f1']}  test macro-F1={test_metrics['macro_f1']}")
