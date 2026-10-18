"""
Step 5: fine-tune a pretrained Transformer from Hugging Face (course topic: Transformers /
contextual embeddings / transfer learning).

How it works:
  tweet -> SentencePiece/WordPiece tokenizer -> input ids + attention mask
        -> pretrained Transformer encoder (self-attention layers, contextual embeddings)
        -> the vector of the first token ([CLS] / <s>) summarises the tweet
        -> a new, randomly initialised Linear layer -> 3 scores
  All weights (encoder + new layer) are updated with AdamW for a few epochs.

I wrote the training loop by hand (instead of using the HF Trainer) so every step
(forward pass, loss, backward pass, optimizer step) is visible.

Run examples:
  python src/train_transformer.py --model Davlan/afro-xlmr-mini --name T1_afroxlmr_mini
  python src/train_transformer.py --model rasyosef/bert-medium-amharic --name T2_bert_medium_amharic
  python src/train_transformer.py --model Davlan/afro-xlmr-mini --no_class_weights --name T3_afroxlmr_mini_noweights
Add --save to keep the fine-tuned model in models/<name> (needed for the web app).
"""
import argparse
import copy
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from utils import (ID2LABEL, LABEL2ID, LABELS, MODELS_DIR, RESULTS_DIR,
                   clean_text, compute_metrics, full_report, load_split,
                   save_results, set_seed)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Davlan/afro-xlmr-mini")
parser.add_argument("--name", default="T1_afroxlmr_mini")
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_len", type=int, default=64)   # tweets are ~15 words, 64 tokens is enough
parser.add_argument("--no_class_weights", action="store_true")
parser.add_argument("--save", action="store_true")
args = parser.parse_args()

set_seed()
torch.set_num_threads(os.cpu_count())
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

train, dev, test = load_split("train"), load_split("dev"), load_split("test")

# ---- 1. Tokenizer and pretrained model ----------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSequenceClassification.from_pretrained(
    args.model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)
model.to(device)
print("Parameters:", sum(p.numel() for p in model.parameters()))


def encode(df):
    texts = [clean_text(t) for t in df["tweet"]]
    enc = tokenizer(texts, truncation=True, max_length=args.max_len, padding="max_length",
                    return_tensors="pt")
    labels = torch.tensor(df["label"].map(LABEL2ID).values)
    return enc["input_ids"], enc["attention_mask"], labels


train_ids, train_mask, train_y = encode(train)
dev_ids, dev_mask, _ = encode(dev)
test_ids, test_mask, _ = encode(test)

# ---- 2. Loss, optimizer and learning-rate schedule ------------------------------------
if args.no_class_weights:
    loss_fn = torch.nn.CrossEntropyLoss()
else:
    w = compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=train_y.numpy())
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float, device=device))

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
steps_per_epoch = (len(train_ids) + args.batch_size - 1) // args.batch_size
total_steps = steps_per_epoch * args.epochs
# warm up for the first 10% of steps, then decrease the learning rate linearly to 0
scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)


def predict(ids, mask):
    """Return predicted labels and class probabilities."""
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(ids), 64):
            logits = model(input_ids=ids[i:i + 64].to(device),
                           attention_mask=mask[i:i + 64].to(device)).logits
            probs.append(torch.softmax(logits, dim=-1).cpu())
    probs = torch.cat(probs)
    return [ID2LABEL[int(p)] for p in probs.argmax(dim=1)], probs.numpy()


# ---- 3. Fine-tuning loop -------------------------------------------------------------
best_f1, best_state, history = -1, None, []
start = time.time()
for epoch in range(1, args.epochs + 1):
    epoch_start = time.time()
    model.train()
    perm = torch.randperm(len(train_ids))
    total_loss = 0
    for step, i in enumerate(range(0, len(train_ids), args.batch_size)):
        idx = perm[i:i + args.batch_size]
        logits = model(input_ids=train_ids[idx].to(device),
                       attention_mask=train_mask[idx].to(device)).logits   # forward pass
        loss = loss_fn(logits, train_y[idx].to(device))                     # how wrong are we?
        loss.backward()                                                     # gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()                                                    # update weights
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item() * len(idx)
        if step % 50 == 0:
            print(f"  epoch {epoch} step {step}/{steps_per_epoch} loss {loss.item():.4f} "
                  f"({(time.time() - start) / 60:.1f} min)")

    dev_pred, _ = predict(dev_ids, dev_mask)
    dev_f1 = compute_metrics(dev["label"], dev_pred)["macro_f1"]
    history.append({"epoch": epoch, "train_loss": total_loss / len(train_ids), "dev_macro_f1": dev_f1,
                    "minutes": round((time.time() - epoch_start) / 60, 1)})
    print(f"epoch {epoch}: train loss {total_loss / len(train_ids):.4f}  dev macro-F1 {dev_f1:.4f}")
    if dev_f1 > best_f1:  # keep the epoch with the best dev score
        best_f1, best_state = dev_f1, copy.deepcopy(model.state_dict())

# ---- 4. Final evaluation with the best epoch ---------------------------------------------
model.load_state_dict(best_state)
dev_pred, _ = predict(dev_ids, dev_mask)
test_pred, test_probs = predict(test_ids, test_mask)
config = {"model": args.model, "normalise": True, "epochs": args.epochs, "lr": args.lr,
          "batch_size": args.batch_size, "class_weights": not args.no_class_weights,
          "minutes": round((time.time() - start) / 60, 1)}
save_results(args.name, config, compute_metrics(dev["label"], dev_pred),
             compute_metrics(test["label"], test_pred))

report, cm = full_report(test["label"], test_pred)
pd.DataFrame(report).T.to_csv(os.path.join(RESULTS_DIR, f"{args.name}_test_report.csv"))
pd.DataFrame(history).to_csv(os.path.join(RESULTS_DIR, f"{args.name}_history.csv"), index=False)
out = pd.DataFrame({"tweet": test["tweet"], "label": test["label"], "pred": test_pred})
for j, label in enumerate(LABELS):
    out[f"p_{label}"] = test_probs[:, j].round(3)
out.to_csv(os.path.join(RESULTS_DIR, f"predictions_{args.name}.csv"), index=False)

if args.save:
    save_dir = os.path.join(MODELS_DIR, args.name)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Saved fine-tuned model to", save_dir)
