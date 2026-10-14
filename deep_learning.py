"""
Stage 5: Deep learning sentiment models (PyTorch + Hugging Face).

Usage (Colab: Runtime > Change runtime type > GPU, then upload the artifacts folder):
    pip install torch transformers scikit-learn pandas plotly scipy
    python deep_learning.py --artifacts artifacts --model all

    --model {all, distilbert, lstm, cnn, attention}   run one model or all of them
    --glove path/to/glove.twitter.27B.100d.txt        optional pretrained embeddings

Models
    distilbert  fine-tuned DistilBERT (raw tweet text, URLs/mentions normalised)
    lstm        embedding -> BiLSTM -> masked max+mean pooling
    cnn         embedding -> parallel Conv1d (kernel sizes 2,3,4,5) -> max-over-time
    attention   embedding -> BiLSTM -> additive attention (weights are returned)

Everything uses the SAME held-out test split as the classical models
(the `split` column from preprocess.py). A 10% validation slice is carved out of
the training set for early stopping / model selection, so the test set is touched once.
Class imbalance is handled with class-weighted cross-entropy.

Outputs (in --artifacts):
    dl_results.csv, dl_predictions.csv, model_comparison.csv, model_comparison.html,
    mcnemar.txt, attention_examples.html,
    distilbert_sentiment/ (model + tokenizer), <name>.pt, vocab.json
"""
import argparse
import copy
import html
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn as nn
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, TensorDataset

LABELS = ["negative", "neutral", "positive"]
L2I = {l: i for i, l in enumerate(LABELS)}
PAD, UNK = 0, 1
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
MENTION_RE = re.compile(r"@\w+")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------ data
def load_data(art, seed):
    df = pd.read_csv(art / "clean_tweets.csv")
    df["clean_text"] = df["clean_text"].fillna("")
    train_full = df[df["split"] == "train"]
    test = df[df["split"] == "test"].reset_index(drop=True)
    train, val = train_test_split(train_full, test_size=0.1, random_state=seed,
                                  stratify=train_full["airline_sentiment"])
    return train.reset_index(drop=True), val.reset_index(drop=True), test


def class_weights(y, device):
    counts = np.bincount(y, minlength=len(LABELS))
    return torch.tensor(len(y) / (len(LABELS) * counts), dtype=torch.float, device=device)


def build_vocab(texts, min_freq=2):
    cnt = Counter(w for t in texts for w in t.split())
    itos = ["<pad>", "<unk>"] + [w for w, c in cnt.most_common() if c >= min_freq]
    return {w: i for i, w in enumerate(itos)}


def encode(texts, vocab, max_len):
    X = np.zeros((len(texts), max_len), dtype=np.int64)
    L = np.ones(len(texts), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [vocab.get(w, UNK) for w in t.split()][:max_len]
        if ids:
            X[i, :len(ids)] = ids
            L[i] = len(ids)
    return torch.from_numpy(X), torch.from_numpy(L)


def load_glove(path, vocab):
    vecs, dim = {}, None
    with open(path, encoding="utf8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if parts[0] in vocab:
                vecs[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
                dim = len(parts) - 1
    W = np.random.normal(0, 0.1, (len(vocab), dim)).astype(np.float32)
    W[PAD] = 0
    for w, v in vecs.items():
        W[vocab[w]] = v
    print(f"GloVe covers {len(vecs)}/{len(vocab)} vocab words")
    return torch.from_numpy(W)


# ------------------------------------------------------------------ models
class _Base(nn.Module):
    def __init__(self, vocab_size, emb_dim, emb_weights):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        if emb_weights is not None:
            self.emb.weight.data.copy_(emb_weights)
        self.drop = nn.Dropout(0.4)


def length_mask(lengths, T):
    return torch.arange(T, device=lengths.device)[None, :] < lengths[:, None]


class BiLSTMClassifier(_Base):
    def __init__(self, vocab_size, emb_dim=128, hidden=128, emb_weights=None):
        super().__init__(vocab_size, emb_dim, emb_weights)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 4, len(LABELS))

    def run_lstm(self, x, lengths):
        packed = pack_padded_sequence(self.drop(self.emb(x)), lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        return out

    def forward(self, x, lengths):
        out = self.run_lstm(x, lengths)
        mask = length_mask(lengths, x.size(1)).unsqueeze(-1)
        mx = out.masked_fill(~mask, -1e9).max(1).values
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(self.drop(torch.cat([mx, mean], dim=1)))


class AttentionBiLSTM(BiLSTMClassifier):
    """BiLSTM + additive attention. forward(..., return_attn=True) exposes the weights."""
    def __init__(self, vocab_size, emb_dim=128, hidden=128, emb_weights=None):
        super().__init__(vocab_size, emb_dim, hidden, emb_weights)
        self.fc = nn.Linear(hidden * 2, len(LABELS))
        self.attn_proj = nn.Linear(hidden * 2, hidden * 2)
        self.attn_vec = nn.Linear(hidden * 2, 1, bias=False)

    def forward(self, x, lengths, return_attn=False):
        out = self.run_lstm(x, lengths)
        scores = self.attn_vec(torch.tanh(self.attn_proj(out))).squeeze(-1)
        scores = scores.masked_fill(~length_mask(lengths, x.size(1)), -1e9)
        weights = torch.softmax(scores, dim=1)
        context = (weights.unsqueeze(-1) * out).sum(1)
        logits = self.fc(self.drop(context))
        return (logits, weights) if return_attn else logits


class TextCNN(_Base):
    def __init__(self, vocab_size, emb_dim=128, n_filters=100, kernels=(2, 3, 4, 5),
                 emb_weights=None):
        super().__init__(vocab_size, emb_dim, emb_weights)
        self.convs = nn.ModuleList(nn.Conv1d(emb_dim, n_filters, k) for k in kernels)
        self.fc = nn.Linear(n_filters * len(kernels), len(LABELS))

    def forward(self, x, lengths):
        e = self.drop(self.emb(x)).transpose(1, 2)           # (B, emb, T)
        feats = [torch.relu(c(e)).max(dim=2).values for c in self.convs]
        return self.fc(self.drop(torch.cat(feats, dim=1)))


# ------------------------------------------------------------------ training utils
def metrics(y_true, y_pred, name, kind):
    row = {"model": name, "type": kind,
           "accuracy": accuracy_score(y_true, y_pred),
           "macro_f1": f1_score(y_true, y_pred, average="macro"),
           "weighted_f1": f1_score(y_true, y_pred, average="weighted")}
    for l in LABELS:
        row[f"f1_{l}"] = f1_score(y_true, y_pred, labels=[l], average="macro")
    return row


@torch.no_grad()
def predict_scratch(model, loader, device):
    model.eval()
    preds = []
    for x, l, _ in loader:
        preds.append(model(x.to(device), l.to(device)).argmax(1).cpu())
    return torch.cat(preds).numpy()


def train_scratch(model, train_dl, val_dl, y_val, w, device, epochs, lr=1e-3, patience=3):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    best_f1, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, l, y in train_dl:
            x, l, y = x.to(device), l.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x, l), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(y)
        f1 = f1_score(y_val, predict_scratch(model, val_dl, device), average="macro")
        print(f"  epoch {ep:2d}  loss {total / len(train_dl.dataset):.4f}  val macro-F1 {f1:.3f}")
        if f1 > best_f1:
            best_f1, best_state, bad = f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                print("  early stopping")
                break
    model.load_state_dict(best_state)
    return model


# ------------------------------------------------------------------ attention visualisation
def attention_html(examples, path):
    """examples: dicts with model, tokens, weights, true, pred."""
    rows = []
    for ex in examples:
        w = np.asarray(ex["weights"], dtype=float)
        w = w / (w.max() + 1e-9)
        spans = "".join(
            f'<span style="background:rgba(255,80,0,{a:.2f});padding:2px 3px;margin:1px;'
            f'border-radius:3px;display:inline-block">{html.escape(t)}</span>'
            for t, a in zip(ex["tokens"], w))
        ok = "&#10003;" if ex["true"] == ex["pred"] else "&#10007;"
        rows.append(f'<div style="margin:14px 0"><b>{ex["model"]}</b> '
                    f'true=<i>{ex["true"]}</i> pred=<i>{ex["pred"]}</i> {ok}<br>{spans}</div>')
    note = ("<p style='color:#666'>Darker = more attention. Attention shows where the model "
            "looked, which is suggestive but not a guaranteed explanation of its decision.</p>")
    Path(path).write_text("<html><body style='font-family:sans-serif;max-width:900px;margin:auto'>"
                          "<h2>Attention examples</h2>" + note + "".join(rows) + "</body></html>")


def pick_examples(test, preds, n_per_class=3):
    idx = []
    for l in LABELS:
        idx += list(test.index[test["airline_sentiment"] == l][:n_per_class])
    return idx


# ------------------------------------------------------------------ runners
def run_scratch(name, args, data, device, art, glove_cache):
    train, val, test = data
    vocab = build_vocab(train["clean_text"])
    (art / "vocab.json").write_text(json.dumps(vocab))
    enc = lambda d: encode(d["clean_text"].tolist(), vocab, args.max_len)
    y = {k: torch.tensor(d["airline_sentiment"].map(L2I).values)
         for k, d in [("train", train), ("val", val), ("test", test)]}
    ds = {k: TensorDataset(*enc(d), y[k]) for k, d in [("train", train), ("val", val), ("test", test)]}
    train_dl = DataLoader(ds["train"], batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(ds["val"], batch_size=256)
    test_dl = DataLoader(ds["test"], batch_size=256)

    emb_w = glove_cache.get("w")
    if args.glove and emb_w is None:
        emb_w = glove_cache["w"] = load_glove(args.glove, vocab)
    emb_dim = emb_w.shape[1] if emb_w is not None else 128
    cls = {"lstm": BiLSTMClassifier, "cnn": TextCNN, "attention": AttentionBiLSTM}[name]
    model = cls(len(vocab), emb_dim=emb_dim, emb_weights=emb_w)

    print(f"\n=== {name} ({sum(p.numel() for p in model.parameters()):,} params) ===")
    t0 = time.time()
    model = train_scratch(model, train_dl, val_dl, y["val"].numpy(),
                          class_weights(y["train"].numpy(), device), device, args.epochs)
    pred_ids = predict_scratch(model, test_dl, device)
    preds = [LABELS[i] for i in pred_ids]
    print(classification_report(test["airline_sentiment"], preds, labels=LABELS, digits=3))
    print(f"  trained in {time.time() - t0:.0f}s")
    torch.save(model.state_dict(), art / f"{name}.pt")

    examples = []
    if name == "attention":
        model.eval()
        for i in pick_examples(test, preds):
            x, l, _ = ds["test"][i]
            with torch.no_grad():
                _, w = model(x[None].to(device), l[None].to(device), return_attn=True)
            n = int(l)
            toks = test.loc[i, "clean_text"].split()[:n] or ["<empty>"]
            examples.append({"model": "attention-BiLSTM", "tokens": toks,
                             "weights": w[0, :len(toks)].cpu().numpy(),
                             "true": test.loc[i, "airline_sentiment"], "pred": preds[i]})
    return preds, examples


def normalise_for_bert(t):
    t = html.unescape(str(t))
    t = URL_RE.sub("http", t)
    return MENTION_RE.sub("@user", t)


def run_distilbert(args, data, device, art):
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)
    train, val, test = data
    tok = AutoTokenizer.from_pretrained(args.bert_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.bert_name, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id=L2I, attn_implementation="eager").to(device)

    def make_ds(d):
        enc = tok([normalise_for_bert(t) for t in d["text"]], truncation=True,
                  max_length=args.bert_max_len, padding="max_length", return_tensors="pt")
        y = torch.tensor(d["airline_sentiment"].map(L2I).values)
        return TensorDataset(enc["input_ids"], enc["attention_mask"], y)

    ds_train, ds_val, ds_test = make_ds(train), make_ds(val), make_ds(test)
    train_dl = DataLoader(ds_train, batch_size=args.bert_batch, shuffle=True)
    y_train = ds_train.tensors[2].numpy()

    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_dl) * args.bert_epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_train, device))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    @torch.no_grad()
    def predict(ds):
        model.eval()
        out = []
        for ids, mask, _ in DataLoader(ds, batch_size=128):
            with torch.autocast(device.type, enabled=use_amp):
                out.append(model(input_ids=ids.to(device),
                                 attention_mask=mask.to(device)).logits.argmax(1).cpu())
        return torch.cat(out).numpy()

    print(f"\n=== distilbert ({args.bert_name}) on {device} ===")
    best_f1, best_state = -1, None
    t0 = time.time()
    for ep in range(1, args.bert_epochs + 1):
        model.train()
        total = 0.0
        for ids, mask, y in train_dl:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device.type, enabled=use_amp):
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = loss_fn(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(y)
        f1 = f1_score(ds_val.tensors[2].numpy(), predict(ds_val), average="macro")
        print(f"  epoch {ep}  loss {total / len(ds_train):.4f}  val macro-F1 {f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    preds = [LABELS[i] for i in predict(ds_test)]
    print(classification_report(test["airline_sentiment"], preds, labels=LABELS, digits=3))
    print(f"  trained in {time.time() - t0:.0f}s")
    out_dir = art / "distilbert_sentiment"
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    # attention of the [CLS] token in the last layer, averaged over heads
    examples, model_eval = [], model.eval()
    for i in pick_examples(test, preds):
        ids, mask, _ = ds_test[i]
        with torch.no_grad():
            att = model_eval(input_ids=ids[None].to(device), attention_mask=mask[None].to(device),
                             output_attentions=True).attentions[-1][0].mean(0)[0].cpu().numpy()
        toks = tok.convert_ids_to_tokens(ids)
        keep = [j for j in range(len(toks)) if mask[j] == 1 and toks[j] not in ("[CLS]", "[SEP]")]
        examples.append({"model": "DistilBERT (CLS attention, last layer)",
                         "tokens": [toks[j] for j in keep], "weights": att[keep],
                         "true": test.loc[i, "airline_sentiment"], "pred": preds[i]})
    return preds, examples


# ------------------------------------------------------------------ comparison
def compare(art, test_y):
    dl_res = pd.read_csv(art / "dl_results.csv")
    dl_pred = pd.read_csv(art / "dl_predictions.csv")
    frames = [dl_res]
    classical_path = art / "final_test_results.csv"
    if classical_path.exists():
        cl = pd.read_csv(classical_path)
        cl["type"] = "classical"
        frames.append(cl.drop(columns=["strategy"], errors="ignore"))
    comp = pd.concat(frames, ignore_index=True).sort_values("macro_f1", ascending=False)
    comp.round(4).to_csv(art / "model_comparison.csv", index=False)
    print("\n=== Model comparison (test set) ===")
    print(comp.round(3).to_string(index=False))

    long = comp.melt(id_vars=["model", "type"], value_vars=["macro_f1", "f1_negative",
                                                            "f1_neutral", "f1_positive"],
                     var_name="metric", value_name="score")
    px.bar(long, x="model", y="score", color="metric", barmode="group", pattern_shape="type",
           title="Test performance: classical vs deep learning").write_html(
        art / "model_comparison.html")

    pred_path = art / "test_predictions.csv"
    if pred_path.exists() and classical_path.exists():
        cp = pd.read_csv(pred_path)
        if len(cp) == len(test_y) and (cp["y_true"].values == test_y.values).all():
            best_cl = comp[comp["type"] == "classical"].iloc[0]["model"]
            best_dl = comp[comp["type"] != "classical"].iloc[0]["model"]
            a = cp[best_cl].values == cp["y_true"].values
            b = dl_pred[best_dl].values == test_y.values
            only_cl, only_dl = int((a & ~b).sum()), int((~a & b).sum())
            p = binomtest(only_dl, only_dl + only_cl, 0.5).pvalue if only_cl + only_dl else 1.0
            msg = (f"Exact McNemar test: {best_dl} vs {best_cl}\n"
                   f"  only {best_dl} correct: {only_dl}\n  only {best_cl} correct: {only_cl}\n"
                   f"  p-value = {p:.4f}\n")
            print("\n" + msg)
            (art / "mcnemar.txt").write_text(msg)
        else:
            print("Skipped McNemar: classical and DL test sets are not aligned.")


def main(args):
    art = Path(args.artifacts)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + ("  (DistilBERT will be slow on CPU; use a GPU)"
                                  if device.type == "cpu" else ""))
    data = load_data(art, args.seed)
    test = data[2]
    names = ["distilbert", "lstm", "cnn", "attention"] if args.model == "all" else [args.model]

    res_path, pred_path = art / "dl_results.csv", art / "dl_predictions.csv"
    results = pd.read_csv(res_path) if res_path.exists() else pd.DataFrame()
    preds_df = pd.read_csv(pred_path) if pred_path.exists() else pd.DataFrame({"y_true": test["airline_sentiment"]})
    all_examples, glove_cache = [], {}

    for name in names:
        if name == "distilbert":
            preds, ex = run_distilbert(args, data, device, art)
        else:
            preds, ex = run_scratch(name, args, data, device, art, glove_cache)
        label = {"distilbert": "distilbert", "lstm": "bilstm", "cnn": "textcnn",
                 "attention": "attention_bilstm"}[name]
        kind = "transformer" if name == "distilbert" else "deep"
        row = pd.DataFrame([metrics(test["airline_sentiment"], preds, label, kind)])
        results = pd.concat([results[results["model"] != label], row], ignore_index=True) \
            if len(results) else row
        preds_df[label] = preds
        all_examples += ex

    results.to_csv(res_path, index=False)
    preds_df.to_csv(pred_path, index=False)
    if all_examples:
        attention_html(all_examples, art / "attention_examples.html")
    compare(art, test["airline_sentiment"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--model", default="all",
                    choices=["all", "distilbert", "lstm", "cnn", "attention"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=40)
    ap.add_argument("--glove", default=None)
    ap.add_argument("--bert-name", default="distilbert-base-uncased")
    ap.add_argument("--bert-epochs", type=int, default=3)
    ap.add_argument("--bert-batch", type=int, default=32)
    ap.add_argument("--bert-max-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
