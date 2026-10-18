# Amharic Tweet Sentiment Classification

**NLP Summative Project, Option 3: Text Classification for an African Language**

Classifying Amharic (አማርኛ) tweets as **negative**, **neutral** or **positive** using real, human-annotated
social-media data from the AfriSenti benchmark. The project compares classical baselines, a BiLSTM trained from
scratch, and fine-tuned pretrained Transformer models.

| Resource      | Link                                                                                                     |
| ------------- | -------------------------------------------------------------------------------------------------------- |
| Web deployment| https://amharic-tweet-sentiment-classifier.streamlit.app/                                                |
| Hosted model  | [Amharic sentiment model on Hugging Face](https://huggingface.co/mahlet-tilahun/amharic-sentiment-model) |
| Demo video    | Add after publishing the video                                                                           |
| Report (PDF)  | [report/report.pdf](report/report.pdf)                                                                   |

---

## 1. Problem

Sentiment analysis tools are common for English, but Amharic is under-resourced and more difficult because it is
written in the Ge'ez script, is morphologically rich, and is frequently mixed with English or spelling variants on
social media. A working Amharic sentiment classifier would help researchers, NGOs and media teams understand
public opinion in Ethiopian online discussions.

**Target users:** media monitors, researchers and customer-feedback teams working with Amharic content.

## 2. Dataset

- **AfriSenti, Amharic subset** (Muhammad et al., 2023a; 2023b), licence CC BY 4.0.
- Official train/dev/test splits, with no overlap across splits.

| split | tweets |      negative |       neutral |      positive |
| ----- | -----: | ------------: | ------------: | ------------: |
| train |  5,984 | 1,548 (25.9%) | 3,104 (51.9%) | 1,332 (22.3%) |
| dev   |  1,497 |   388 (25.9%) |   776 (51.8%) |   333 (22.2%) |
| test  |  1,999 | 1,337 (66.9%) |   224 (11.2%) |   438 (21.9%) |

Key findings from `src/explore_data.py` and `results/data_stats.json`:

- **Label shift:** train/dev are mostly neutral, while the test set is dominated by negative tweets.
- **Many unseen words:** 30%+ of dev/test tokens do not appear in training.
- **Test text differs from train/dev:** more `@user` placeholders and almost no emojis, suggesting a shift in formatting.

**Preprocessing** (`src/utils.py → clean_text`): remove `@user`, URLs and hashtags, replace Ethiopic punctuation with
spaces, lowercase Latin letters, and normalise homophone Amharic letters such as ሐ/ኀ→ሀ and ዐ→አ.

## 3. Methods

| Group        | Model                                                                        | Course concept                                         |
| ------------ | ---------------------------------------------------------------------------- | ------------------------------------------------------ |
| Baselines    | majority class; TF-IDF + Logistic Regression; Naive Bayes; character n-grams | bag-of-words, TF-IDF                                   |
| RNN          | BiLSTM with learned embeddings and word/subword tokens                       | RNN/LSTM                                               |
| Transformers | fine-tuned **AfroXLMR-mini** and **BERT-medium-Amharic**                     | contextual embeddings, Transformers, transfer learning |

All models are chosen on the **dev** set and evaluated once on the **test** set. The main metric is **macro-F1**
because the label distribution changes between dev and test.

## 4. Results

The strongest baseline was the balanced character TF-IDF + logistic regression model (macro-F1 0.4608 and
weighted F1 0.5397 on test),
while the best deep model was the Amharic BERT fine-tune (macro-F1 0.6249 on test). The larger improvement on the
shifted test split shows that pretrained contextual embeddings generalise better than shallow baselines.

| Model                                         | Dev macro-F1 | Test macro-F1 | Test weighted F1 |
| --------------------------------------------- | -----------: | ------------: | ---------------: |
| Majority class                                |       0.2276 |        0.0672 |           0.0226 |
| Best baseline (char TF-IDF + balanced logreg) |       0.5312 |        0.4608 |           0.5397 |
| AfroXLMR-mini                                 |       0.5371 |        0.5795 |           0.7057 |
| Amharic BERT (final)                          |       0.6038 |        0.6249 |           0.7249 |

## 5. How to run

```bash
# 1. set up (Python 3.10+)
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt

# 2. data
python src/download_data.py
python src/explore_data.py

# 3. experiments
python src/train_baseline.py
python src/train_bilstm.py --tokens word
python src/train_bilstm.py --tokens subword
python src/train_transformer.py --model Davlan/afro-xlmr-mini --name T1_afroxlmr_mini --save
python src/train_transformer.py --model rasyosef/bert-medium-amharic --name T2_bert_medium_amharic --save

# 4. evaluation + error analysis
python src/evaluate.py --final T2_bert_medium_amharic

# 5. local web app
streamlit run streamlit_app.py    # http://localhost:8501
```

Transformer fine-tuning can take 1–2 hours on a CPU for one model; GPU training is much faster.

### Streamlit Community Cloud deployment

The project’s model weights are excluded from Git to avoid pushing a large file. They are available from the [Hugging Face model repository](https://huggingface.co/mahlet-tilahun/amharic-sentiment-model).

## 6. Repository structure

```text
├── streamlit_app.py            final web app
├── data/raw/                  AfriSenti Amharic train/dev/test
├── models/                    local model files (gitignored; hosted model linked above)
├── report/                    final report
├── results/                   experiments.csv, metrics, predictions, figures
├── src/
│   ├── utils.py               preprocessing and evaluation helpers
│   ├── download_data.py
│   ├── explore_data.py
│   ├── train_baseline.py      baseline experiments
│   ├── train_bilstm.py        sequence model training
│   ├── train_transformer.py   fine-tuning loop
│   ├── evaluate.py            evaluation and error analysis
│   └── label_shift.py         label-shift investigation
├── requirements.txt
├── README.md
└── .gitignore
```
