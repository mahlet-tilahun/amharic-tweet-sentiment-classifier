"""
Step 3: baselines with classical machine learning (scikit-learn).

Every experiment is trained on TRAIN, compared on DEV, and the TEST score is also
recorded so the final table is complete. I only choose settings using DEV scores.

Experiments:
  B0  majority class                     -> the "do nothing" lower bound
  B1  TF-IDF words + Logistic Regression (no Amharic normalisation)
  B2  TF-IDF words + Logistic Regression (with normalisation)   -> does normalisation help?
  B3  Multinomial Naive Bayes on word counts                    -> another classic baseline
  B4  TF-IDF character n-grams + Logistic Regression            -> does subword info help?
  B5  B4 + class_weight="balanced"                              -> does balancing classes help?
  B6  B5 + remove emojis/symbols                                -> does matching the test text help?

Run:  python src/train_baseline.py
"""
import os

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from utils import (MODELS_DIR, RESULTS_DIR, clean_text, compute_metrics,
                   load_split, save_results, set_seed)

set_seed()
train, dev, test = load_split("train"), load_split("dev"), load_split("test")


def run(name, pipeline, normalise=True, description="", remove_emoji=False):
    """Clean the text, fit the pipeline on train, and score it on dev and test."""
    x_train = train["tweet"].apply(lambda t: clean_text(t, normalise, remove_emoji))
    x_dev = dev["tweet"].apply(lambda t: clean_text(t, normalise, remove_emoji))
    x_test = test["tweet"].apply(lambda t: clean_text(t, normalise, remove_emoji))

    pipeline.fit(x_train, train["label"])
    dev_pred = pipeline.predict(x_dev)
    test_pred = pipeline.predict(x_test)

    dev_metrics = compute_metrics(dev["label"], dev_pred)
    save_results(name, {"model": description, "normalise": normalise, "remove_emoji": remove_emoji},
                 dev_metrics, compute_metrics(test["label"], test_pred))
    all_runs[name] = (dev_metrics["macro_f1"], pipeline, test_pred)


all_runs = {}  # name -> (dev macro-F1, fitted pipeline, test predictions)


# B0: always predict the most common class in train ("neutral")
run("B0_majority", Pipeline([("vec", CountVectorizer()),
                             ("clf", DummyClassifier(strategy="most_frequent"))]),
    description="Majority class")

# B1 / B2: word unigrams + bigrams, TF-IDF weighted, logistic regression
def word_logreg():
    return Pipeline([
        ("vec", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, C=5)),
    ])

run("B1_word_tfidf_logreg_nonorm", word_logreg(), normalise=False,
    description="TF-IDF word 1-2gram + LogReg")
run("B2_word_tfidf_logreg", word_logreg(), normalise=True,
    description="TF-IDF word 1-2gram + LogReg")

# B3: Naive Bayes on raw word counts
run("B3_word_count_nb", Pipeline([
        ("vec", CountVectorizer(ngram_range=(1, 2))),
        ("clf", MultinomialNB(alpha=0.5)),
    ]), description="Word count 1-2gram + Multinomial NB")

# B4: character n-grams inside word boundaries. Amharic words carry many prefixes
# and suffixes (e.g. negation, possessives), so pieces of words are informative.
def char_logreg(class_weight=None):
    return Pipeline([
        ("vec", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=3000, C=5, class_weight=class_weight)),
    ])

run("B4_char_tfidf_logreg", char_logreg(), description="TF-IDF char 2-5gram + LogReg")

# B5: same as B4 but each class gets equal total weight in the loss
run("B5_char_tfidf_logreg_balanced", char_logreg("balanced"),
    description="TF-IDF char 2-5gram + LogReg (balanced)")

# B6: same as B5 but emojis/typographic symbols removed, because the test tweets have none.
# (Decided after looking at the test *texts* in explore_data.py, never at test labels.)
run("B6_char_tfidf_logreg_balanced_noemoji", char_logreg("balanced"), remove_emoji=True,
    description="TF-IDF char 2-5gram + LogReg (balanced), emoji removed")

# Keep the baseline with the best DEV macro-F1 (never choose using the test set)
best_name = max(all_runs, key=lambda n: all_runs[n][0])
_, best_pipeline, best_test_pred = all_runs[best_name]
print("Best baseline on dev:", best_name)

# Save it + its test predictions (used for error analysis and as a light app fallback)
os.makedirs(MODELS_DIR, exist_ok=True)
joblib.dump(best_pipeline, os.path.join(MODELS_DIR, "baseline.joblib"))
out = pd.DataFrame({"tweet": test["tweet"], "label": test["label"], "pred": best_test_pred})
probs = best_pipeline.predict_proba(test["tweet"].apply(clean_text))
for j, label in enumerate(best_pipeline.classes_):  # sklearn sorts classes alphabetically
    out[f"p_{label}"] = probs[:, j].round(3)
out.to_csv(os.path.join(RESULTS_DIR, "predictions_baseline.csv"), index=False)
print("Saved best baseline to models/baseline.joblib")
