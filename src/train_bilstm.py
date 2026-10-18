"""
Step 4: a Bidirectional LSTM classifier written in PyTorch (course topic: RNN / LSTM).

How it works:
  tweet -> tokens -> token ids -> Embedding layer (learned from scratch)
        -> BiLSTM (reads the tweet left-to-right AND right-to-left)
        -> max-pooling over all time steps (keeps the strongest signal of each feature)
        -> dropout -> Linear layer -> 3 scores (negative / neutral / positive)

Two tokenisation options (this is the experiment):
  --tokens word     split on spaces (every inflected Amharic word is its own token)
  --tokens subword  use the SentencePiece tokenizer of AfroXLMR, so unknown words are
                    broken into known pieces (should reduce the out-of-vocabulary problem)

Run:  python src/train_bilstm.py --tokens word
      python src/train_bilstm.py --tokens subword
"""
import argparse
import copy
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight

from utils import (ID2LABEL, LABEL2ID, LABELS, RESULTS_DIR, clean_text,
                   compute_metrics, load_split, save_results, set_seed)

parser = argparse.ArgumentParser()
parser.add_argument("--tokens", choices=["word", "subword"], default="word")
parser.add_argument("--emb_dim", type=int, default=128)
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--max_len", type=int, default=64)
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--dropout", type=float, default=0.3)
args = parser.parse_args()

set_seed()
torch.set_num_threads(os.cpu_count())
train, dev, test = load_split("train"), load_split("dev"), load_split("test")

# ---- 1. Tokenisation and vocabulary -------------------------------------------
if args.tokens == "word":
    tokenize = lambda text: clean_text(text).split()
else:
    # only the tokenizer is reused here, NOT the pretrained model weights
    from transformers import AutoTokenizer
    hf_tok = AutoTokenizer.from_pretrained("Davlan/afro-xlmr-mini")
    tokenize = lambda text: hf_tok.tokenize(clean_text(text))

# keep tokens seen at least twice in train; everything else becomes <unk>
counts = Counter(tok for t in train["tweet"] for tok in tokenize(t))
vocab = {"<pad>": 0, "<unk>": 1}
for token, c in counts.items():
    if c >= 2:
        vocab[token] = len(vocab)
encode = lambda text: [vocab.get(tok, 1) for tok in tokenize(text)]
vocab_size = len(vocab)
print(f"Tokens: {args.tokens}, vocabulary size: {vocab_size}")
print("Example:", tokenize(train['tweet'][0])[:12])


def to_tensor(df):
    """Turn a DataFrame into padded id tensor X and label tensor y."""
    X = np.zeros((len(df), args.max_len), dtype=np.int64)  # 0 = <pad>
    for i, text in enumerate(df["tweet"]):
        ids = encode(text)[: args.max_len] or [1]
        X[i, : len(ids)] = ids
    y = df["label"].map(LABEL2ID).values
    return torch.tensor(X), torch.tensor(y)


X_train, y_train = to_tensor(train)
X_dev, y_dev = to_tensor(dev)
X_test, y_test = to_tensor(test)


# ---- 2. The model ---------------------------------------------------------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden, n_classes, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, n_classes)  # 2x because forward + backward

    def forward(self, x):
        emb = self.dropout(self.embedding(x))          # (batch, seq_len, emb_dim)
        outputs, _ = self.lstm(emb)                    # (batch, seq_len, 2*hidden)
        mask = (x != 0).unsqueeze(-1)                  # ignore padding positions
        outputs = outputs.masked_fill(~mask, -1e9)
        pooled, _ = outputs.max(dim=1)                 # max-pool over time
        return self.out(self.dropout(pooled))          # (batch, 3) raw scores (logits)


model = BiLSTMClassifier(vocab_size, args.emb_dim, args.hidden, len(LABELS), args.dropout)
print("Trainable parameters:", sum(p.numel() for p in model.parameters()))

# class weights: rarer classes count more in the loss (this helped the baseline)
weights = compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=y_train.numpy())
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float))
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def predict(X):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            preds.append(model(X[i:i + 256]).argmax(dim=1))
    return [ID2LABEL[int(p)] for p in torch.cat(preds)]


# ---- 3. Training loop with early stopping on dev macro-F1 --------------------------
best_f1, best_state, patience, history = -1, None, 0, []
for epoch in range(1, args.epochs + 1):
    model.train()
    perm = torch.randperm(len(X_train))
    total_loss = 0
    for i in range(0, len(X_train), args.batch_size):
        idx = perm[i:i + args.batch_size]
        optimizer.zero_grad()
        loss = loss_fn(model(X_train[idx]), y_train[idx])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # avoid exploding gradients
        optimizer.step()
        total_loss += loss.item() * len(idx)

    dev_f1 = compute_metrics(dev["label"], predict(X_dev))["macro_f1"]
    history.append({"epoch": epoch, "train_loss": total_loss / len(X_train), "dev_macro_f1": dev_f1})
    print(f"epoch {epoch:2d}  train loss {total_loss / len(X_train):.4f}  dev macro-F1 {dev_f1:.4f}")

    if dev_f1 > best_f1:
        best_f1, best_state, patience = dev_f1, copy.deepcopy(model.state_dict()), 0
    else:
        patience += 1
        if patience == 3:
            print("Dev score stopped improving -> early stopping")
            break

# ---- 4. Evaluate the best epoch -------------------------------------------------------
model.load_state_dict(best_state)
name = f"L_bilstm_{args.tokens}"
dev_pred, test_pred = predict(X_dev), predict(X_test)
save_results(name,
             {"model": f"BiLSTM ({args.tokens} tokens, random init embeddings)", "normalise": True},
             compute_metrics(dev["label"], dev_pred), compute_metrics(test["label"], test_pred))
pd.DataFrame(history).to_csv(os.path.join(RESULTS_DIR, f"{name}_history.csv"), index=False)
pd.DataFrame({"tweet": test["tweet"], "label": test["label"], "pred": test_pred}).to_csv(
    os.path.join(RESULTS_DIR, f"predictions_{name}.csv"), index=False)
