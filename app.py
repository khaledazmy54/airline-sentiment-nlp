"""
Stage 6: Streamlit sentiment dashboard.

Run (from the folder containing app.py, preprocess.py and the artifacts folder):
    pip install -r requirements.txt
    streamlit run app.py
    # custom artifacts location:  ARTIFACTS_DIR=/path/to/artifacts streamlit run app.py

Tabs
    1. Live analysis      type a tweet -> sentiment, topic, similar tweets, hashtags/mentions
    2. Trends & alerts    Plotly sentiment trends + negative-spike alerts (rolling z-score)
    3. Batch & export     classify thousands of tweets (upload CSV or demo data), export CSV / HTML report

The app uses whatever earlier stages produced: DistilBERT if available, otherwise the best
classical model; LDA topics if gensim + the LDA files exist; human topic labels from
review_sheet.csv when filled in.
"""
import html
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import clean_text, extract_entities, tokenize_and_lemmatize  # noqa: E402

LABELS = ["negative", "neutral", "positive"]
COLORS = {"negative": "#d62728", "neutral": "#7f7f7f", "positive": "#2ca02c"}
EMOJI = {"negative": "😠", "neutral": "😐", "positive": "😊"}
# Keep in sync with DOMAIN_STOPWORDS in unsupervised.py
DOMAIN_STOPWORDS = {"flight", "get", "go", "amp", "u", "would", "one", "say", "make",
                    "take", "also", "im", "us", "need", "know", "want"}
MODEL_FILES = {"Linear SVM": "best_linear_svm.joblib",
               "Logistic Regression": "best_logreg.joblib",
               "Random Forest": "best_random_forest.joblib"}
NAME_MAP = {"linear_svm": "Linear SVM", "logreg": "Logistic Regression",
            "random_forest": "Random Forest", "distilbert": "DistilBERT"}
EXAMPLE = "@United my bag was lost AGAIN and nobody at the gate will help me. 4 hours of waiting. #terrible"

st.set_page_config(page_title="Airline Sentiment Monitor", page_icon="📊", layout="wide")


# ======================================================================== text + models
def preprocess_text(raw: str) -> str:
    _, lemmas = tokenize_and_lemmatize(clean_text(str(raw)))
    return " ".join(lemmas)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class ClassicalModel:
    """Wraps an sklearn/imblearn pipeline trained on lemma_text."""

    def __init__(self, est):
        self.est = est
        self.calibrated = hasattr(est, "predict_proba")

    def predict_proba(self, raw_texts, lemma_texts):
        if self.calibrated:
            p = self.est.predict_proba(lemma_texts)
        else:                                   # LinearSVC: softmax of margins (uncalibrated)
            p = softmax(np.atleast_2d(self.est.decision_function(lemma_texts)))
        order = [list(self.est.classes_).index(l) for l in LABELS]
        return p[:, order]


class BertModel:
    calibrated = True

    def __init__(self, path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path)).eval().to(self.device)
        label2id = {v: k for k, v in self.model.config.id2label.items()}
        self.order = [int(label2id[l]) for l in LABELS]

    @staticmethod
    def _norm(t):
        import re
        t = html.unescape(str(t))
        t = re.sub(r"(https?://\S+|www\.\S+)", "http", t)
        return re.sub(r"@\w+", "@user", t)

    def predict_proba(self, raw_texts, lemma_texts=None):
        out = []
        texts = [self._norm(t) for t in raw_texts]
        for i in range(0, len(texts), 64):
            enc = self.tok(texts[i:i + 64], truncation=True, max_length=64, padding=True,
                           return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                logits = self.model(**enc).logits
            out.append(self.torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(out)[:, self.order]


@st.cache_resource(show_spinner="Loading tweets…")
def load_data(art_dir: str) -> pd.DataFrame:
    df = pd.read_csv(Path(art_dir) / "clean_tweets.csv")
    df["lemma_text"] = df["lemma_text"].fillna("")
    if "tweet_created" in df:
        df["tweet_created"] = pd.to_datetime(df["tweet_created"], errors="coerce", utc=True)
    return df


def discover_models(art: Path):
    found = {n: ("classical", art / f) for n, f in MODEL_FILES.items() if (art / f).exists()}
    if (art / "distilbert_sentiment").exists():
        found["DistilBERT"] = ("bert", art / "distilbert_sentiment")
    default_order = []
    for f in ["model_comparison.csv", "final_test_results.csv"]:
        if (art / f).exists():
            res = pd.read_csv(art / f).sort_values("macro_f1", ascending=False)
            default_order = [NAME_MAP.get(m) for m in res["model"]]
            break
    ranked = [m for m in default_order if m in found] + [m for m in found if m not in default_order]
    return found, ranked


@st.cache_resource(show_spinner="Loading model…")
def load_model(kind: str, path: str):
    return ClassicalModel(joblib.load(path)) if kind == "classical" else BertModel(path)


@st.cache_resource(show_spinner="Loading topic model…")
def load_topics(art_dir: str):
    art = Path(art_dir)
    info = {"lda": None, "dictionary": None, "words": {}, "labels": {}, "error": None}
    try:
        from gensim.corpora import Dictionary
        from gensim.models import LdaModel
        if (art / "lda_model").exists() and (art / "dictionary.gensim").exists():
            info["lda"] = LdaModel.load(str(art / "lda_model"))
            info["dictionary"] = Dictionary.load(str(art / "dictionary.gensim"))
    except Exception as e:  # gensim missing or incompatible
        info["error"] = str(e)
    if (art / "topic_summary.csv").exists():
        ts = pd.read_csv(art / "topic_summary.csv")
        info["words"] = dict(ts[ts["method"] == "lda"][["topic", "top_words"]].values)
    if (art / "review_sheet.csv").exists():
        rs = pd.read_csv(art / "review_sheet.csv")
        rs = rs[(rs["method"] == "lda") & rs["human_label"].notna()]
        info["labels"] = {int(t): str(l).strip() for t, l in rs[["topic", "human_label"]].values
                          if str(l).strip()}
    return info


def topic_name(info, t: int) -> str:
    if t in info["labels"]:
        return f"{info['labels'][t]} (topic {t})"
    words = ", ".join(str(info["words"].get(t, "")).split(", ")[:3])
    return f"Topic {t}: {words}" if words else f"Topic {t}"


def assign_topics(info, lemma_text: str):
    """Returns [(topic_id, prob), ...] sorted by prob, or None."""
    if info["lda"] is None:
        return None
    toks = [w for w in lemma_text.split() if w not in DOMAIN_STOPWORDS]
    bow = info["dictionary"].doc2bow(toks)
    if not bow:
        return None
    return sorted(info["lda"].get_document_topics(bow, minimum_probability=0.0),
                  key=lambda x: -x[1])


@st.cache_resource(show_spinner="Indexing tweets for similarity search…")
def load_similarity(art_dir: str):
    art = Path(art_dir)
    if not (art / "tfidf_topics.joblib").exists():
        return None
    tfidf = joblib.load(art / "tfidf_topics.joblib")
    return tfidf, tfidf.transform(load_data(art_dir)["lemma_text"])


def similar_tweets(art_dir, lemma_text, k=5):
    sim = load_similarity(art_dir)
    if sim is None or not lemma_text.strip():
        return None
    tfidf, X = sim
    q = tfidf.transform([lemma_text])
    scores = (X @ q.T).toarray().ravel()
    top = [i for i in np.argsort(scores)[::-1][:k] if scores[i] > 0]
    df = load_data(art_dir)
    cols = [c for c in ["text", "airline", "airline_sentiment"] if c in df]
    out = df.iloc[top][cols].copy()
    out.insert(0, "similarity", scores[top].round(3))
    return out


# ======================================================================== trends + alerts
def detect_spikes(df, time_col, sent_col, freq, window, z_thr, min_n):
    s = df.dropna(subset=[time_col]).copy()
    s["_neg"] = (s[sent_col] == "negative").astype(int)
    s = s.set_index(time_col).sort_index()
    agg = s.resample(freq).agg(neg_share=("_neg", "mean"), n=("_neg", "size"))
    agg = agg[agg["n"] >= min_n].copy()
    minp = max(3, window // 2)
    mean = agg["neg_share"].rolling(window, min_periods=minp).mean().shift(1)   # past only
    std = agg["neg_share"].rolling(window, min_periods=minp).std().shift(1)
    agg["expected"] = mean
    agg["z"] = (agg["neg_share"] - mean) / std.replace(0, np.nan)
    agg["spike"] = agg["z"] > z_thr                                              # upward only
    return agg


def trend_figures(df, agg, time_col, sent_col, freq):
    fig1 = go.Figure()
    fig1.add_scatter(x=agg.index, y=agg["neg_share"], mode="lines+markers",
                     name="Negative share", line_color=COLORS["negative"])
    fig1.add_scatter(x=agg.index, y=agg["expected"], mode="lines", name="Expected (rolling)",
                     line=dict(color="#888", dash="dot"))
    sp = agg[agg["spike"]]
    fig1.add_scatter(x=sp.index, y=sp["neg_share"], mode="markers", name="Spike",
                     marker=dict(size=14, symbol="x", color="black"))
    fig1.update_layout(title="Negative-tweet share with spike detection",
                       yaxis_tickformat=".0%", margin=dict(t=50, b=10))

    s = df.dropna(subset=[time_col])
    counts = (s.groupby([pd.Grouper(key=time_col, freq=freq), sent_col]).size()
                .unstack(fill_value=0).reindex(columns=LABELS, fill_value=0))
    counts = counts[counts.sum(axis=1) >= 1]
    fig2 = px.bar(counts.reset_index().melt(id_vars=time_col, var_name="sentiment", value_name="tweets"),
                  x=time_col, y="tweets", color="sentiment", color_discrete_map=COLORS,
                  category_orders={"sentiment": LABELS}, title="Tweet volume by sentiment")
    fig2.update_layout(margin=dict(t=50, b=10))
    return fig1, fig2


def render_alert(agg, z_thr):
    if agg.empty:
        st.info("Not enough tweets per time bucket for alerting. Lower the minimum or use wider buckets.")
        return
    last = agg.iloc[-1]
    if bool(last["spike"]):
        st.error(f"🚨 **Negative spike** in the latest bucket ({agg.index[-1]:%Y-%m-%d %H:%M}): "
                 f"{last['neg_share']:.0%} negative vs ~{last['expected']:.0%} expected "
                 f"(z = {last['z']:.1f}, {int(last['n'])} tweets).")
    else:
        z = last["z"]
        st.success(f"✅ No spike in the latest bucket ({agg.index[-1]:%Y-%m-%d %H:%M}): "
                   f"{last['neg_share']:.0%} negative"
                   + ("" if pd.isna(z) else f" (z = {z:.1f}, alert threshold {z_thr})") + ".")
    n_sp = int(agg["spike"].sum())
    st.caption(f"{n_sp} spike bucket(s) in the selected period out of {len(agg)}.")


# ======================================================================== export
def build_report(title, summary: pd.DataFrame, figs, spikes: pd.DataFrame):
    parts = [f"<h1>{html.escape(title)}</h1>",
             "<h2>Summary</h2>", summary.to_html(index=False, border=0)]
    for i, f in enumerate(figs):
        parts.append(f.to_html(full_html=False, include_plotlyjs=True if i == 0 else False))
    if spikes is not None and len(spikes):
        parts += ["<h2>Negative spikes</h2>", spikes.round(3).to_html(border=0)]
    return ("<html><head><meta charset='utf-8'><style>body{font-family:sans-serif;max-width:1000px;"
            "margin:auto;padding:20px}table{border-collapse:collapse}td,th{padding:4px 10px;"
            "border-bottom:1px solid #ddd}</style></head><body>" + "".join(parts) + "</body></html>")


def classify_df(df, text_col, model, topics, progress):
    texts = df[text_col].fillna("").astype(str).tolist()
    lemmas = []
    for i, t in enumerate(texts):
        lemmas.append(preprocess_text(t))
        if i % 50 == 0:
            progress.progress(min(i / max(len(texts), 1), 0.9), text=f"Preprocessing {i:,}/{len(texts):,}")
    progress.progress(0.92, text="Running model…")
    proba = model.predict_proba(texts, lemmas)
    out = df.copy()
    out["predicted_sentiment"] = [LABELS[i] for i in proba.argmax(1)]
    out["confidence"] = proba.max(1).round(3)
    for j, l in enumerate(LABELS):
        out[f"p_{l}"] = proba[:, j].round(3)
    if topics["lda"] is not None:
        tp = [assign_topics(topics, l) for l in lemmas]
        out["topic"] = [topic_name(topics, t[0][0]) if t else "n/a" for t in tp]
    progress.progress(1.0, text="Done")
    return out


# ======================================================================== UI
st.title("📊 Airline Sentiment Monitor")

art_dir = st.sidebar.text_input("Artifacts folder", value=os.environ.get("ARTIFACTS_DIR", "artifacts"))
art = Path(art_dir)
if not (art / "clean_tweets.csv").exists():
    st.error(f"`{art / 'clean_tweets.csv'}` not found. Run preprocess.py first, or fix the folder path.")
    st.stop()

df = load_data(art_dir)
models, ranked = discover_models(art)
if not models:
    st.error("No trained model found (best_*.joblib or distilbert_sentiment/). Run supervised.py first.")
    st.stop()

choice = st.sidebar.selectbox("Sentiment model", ranked, index=0,
                              help="Defaults to the best model by test macro-F1.")
try:
    model = load_model(*models[choice][:1], str(models[choice][1]))
except Exception as e:
    fallback = next((m for m in ranked if models[m][0] == "classical"), None)
    if fallback is None:
        st.error(f"Could not load {choice}: {e}")
        st.stop()
    st.sidebar.warning(f"{choice} could not be loaded ({type(e).__name__}); using {fallback}.")
    choice = fallback
    model = load_model(*models[choice][:1], str(models[choice][1]))
topics = load_topics(art_dir)

st.sidebar.subheader("Alert settings")
freq_label = st.sidebar.selectbox("Time bucket", ["1 hour", "3 hours", "6 hours", "1 day"], index=1)
freq = {"1 hour": "1h", "3 hours": "3h", "6 hours": "6h", "1 day": "1D"}[freq_label]
window = st.sidebar.slider("Baseline window (buckets)", 4, 24, 8)
z_thr = st.sidebar.slider("Spike threshold (z-score)", 1.0, 4.0, 2.0, 0.1)
min_n = st.sidebar.slider("Min tweets per bucket", 5, 100, 30)
st.sidebar.caption(f"Model: **{choice}** · {len(df):,} tweets loaded")

tab_live, tab_trend, tab_batch = st.tabs(["Live analysis", "Trends & alerts", "Batch & export"])

# ---------------------------------------------------------------- live
with tab_live:
    text = st.text_area("Tweet text", value=EXAMPLE, height=100,
                        help="Press Ctrl+Enter to analyse.")
    if text.strip():
        lemma = preprocess_text(text)
        proba = model.predict_proba([text], [lemma])[0]
        pred = LABELS[int(proba.argmax())]
        c1, c2, c3 = st.columns([1, 1.4, 1.6])
        c1.metric("Predicted sentiment", f"{EMOJI[pred]} {pred.title()}")
        c1.metric("Confidence" if model.calibrated else "Score (uncalibrated)", f"{proba.max():.0%}")
        if not model.calibrated:
            c1.caption("Linear SVM has no probabilities; this is a softmax of its margins.")
        fig = px.bar(x=proba, y=[l.title() for l in LABELS], orientation="h",
                     color=LABELS, color_discrete_map=COLORS)
        fig.update_layout(showlegend=False, height=190, margin=dict(t=10, b=10, l=0, r=0),
                          xaxis=dict(range=[0, 1], tickformat=".0%", title=None), yaxis_title=None)
        c2.plotly_chart(fig, width="stretch")

        with c3:
            st.markdown("**Topic**")
            tp = assign_topics(topics, lemma)
            if tp:
                st.write(f"{topic_name(topics, tp[0][0])}  —  {tp[0][1]:.0%}")
                st.caption("Next best: " + " · ".join(f"{topic_name(topics, t)} ({p:.0%})" for t, p in tp[1:3]))
            elif topics["lda"] is None:
                st.caption("LDA files not found" + (f" ({topics['error']})" if topics["error"] else "")
                           + ". Run unsupervised.py and install gensim.")
            else:
                st.caption("No topic words recognised in this text.")
            hashtags, mentions = extract_entities(text)
            st.markdown("**Hashtags / mentions**")
            st.write(" ".join(hashtags) or "—", " | ", " ".join(mentions) or "—")

        if not lemma:
            st.warning("No informative words remain after cleaning, so this prediction is unreliable.")
        with st.expander("Preprocessed text"):
            st.code(lemma or "(empty)")

        st.subheader("Similar tweets")
        sim = similar_tweets(art_dir, lemma)
        if sim is None or sim.empty:
            st.caption("No similar tweets found (or tfidf_topics.joblib is missing).")
        else:
            st.dataframe(sim, width="stretch", hide_index=True)

# ---------------------------------------------------------------- trends
with tab_trend:
    if "tweet_created" not in df or df["tweet_created"].notna().sum() == 0:
        st.info("The dataset has no usable timestamps, so trends are unavailable here. "
                "You can still use the Batch tab with your own time column.")
    else:
        d = df
        if "airline" in df:
            picks = st.multiselect("Airlines", sorted(df["airline"].dropna().unique()),
                                   default=sorted(df["airline"].dropna().unique()))
            d = df[df["airline"].isin(picks)] if picks else df
        agg = detect_spikes(d, "tweet_created", "airline_sentiment", freq, window, z_thr, min_n)
        render_alert(agg, z_thr)
        if not agg.empty:
            f1, f2 = trend_figures(d, agg, "tweet_created", "airline_sentiment", freq)
            st.plotly_chart(f1, width="stretch")
            st.plotly_chart(f2, width="stretch")
            if agg["spike"].any():
                st.subheader("Spike log")
                st.dataframe(agg[agg["spike"]].round(3), width="stretch")
        st.caption("Trends here use the dataset's human labels (historical). "
                   "The Batch tab applies the same logic to model predictions on your own data.")

# ---------------------------------------------------------------- batch
with tab_batch:
    src = st.radio("Data source", ["Demo: sample of project tweets", "Upload CSV"], horizontal=True)
    work, text_col, time_col = None, None, None
    if src.startswith("Demo"):
        n = st.slider("Sample size", 100, min(len(df), 5000), min(len(df), 1000), step=100) \
            if len(df) > 100 else len(df)
        work = df.sample(n, random_state=0)[[c for c in ["text", "tweet_created", "airline"] if c in df]]
        text_col, time_col = "text", ("tweet_created" if "tweet_created" in work else None)
    else:
        up = st.file_uploader("CSV with a tweet-text column", type="csv")
        if up is not None:
            work = pd.read_csv(up)
            guess = next((c for c in work.columns if c.lower() in ("text", "tweet", "content")), work.columns[0])
            text_col = st.selectbox("Text column", work.columns, index=list(work.columns).index(guess))
            opt = ["(none)"] + list(work.columns)
            tc = st.selectbox("Timestamp column (optional, enables trends/alerts)", opt)
            time_col = None if tc == "(none)" else tc
            if time_col:
                work[time_col] = pd.to_datetime(work[time_col], errors="coerce", utc=True)

    if work is not None and st.button("Classify", type="primary"):
        bar = st.progress(0.0, text="Starting…")
        st.session_state["batch"] = {"df": classify_df(work, text_col, model, topics, bar),
                                     "time_col": time_col, "model": choice}
        bar.empty()

    b = st.session_state.get("batch")
    if b is not None:
        res, tcol = b["df"], b["time_col"]
        dist = res["predicted_sentiment"].value_counts().reindex(LABELS, fill_value=0)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tweets", f"{len(res):,}")
        for col, l in zip([m2, m3, m4], LABELS):
            col.metric(f"{EMOJI[l]} {l.title()}", f"{dist[l] / len(res):.0%}")

        figs = [px.bar(x=dist.index, y=dist.values, color=dist.index, color_discrete_map=COLORS,
                       labels={"x": "sentiment", "y": "tweets"}, title="Predicted sentiment")]
        if "topic" in res:
            tt = (pd.crosstab(res["topic"], res["predicted_sentiment"], normalize="index")
                    .reindex(columns=LABELS, fill_value=0).reset_index()
                    .melt(id_vars="topic", var_name="sentiment", value_name="share"))
            figs.append(px.bar(tt, y="topic", x="share", color="sentiment", orientation="h",
                               color_discrete_map=COLORS, title="Sentiment mix by topic"))
        spikes = None
        if tcol and res[tcol].notna().any():
            agg = detect_spikes(res, tcol, "predicted_sentiment", freq, window, z_thr, min_n)
            render_alert(agg, z_thr)
            if not agg.empty:
                f1, f2 = trend_figures(res, agg, tcol, "predicted_sentiment", freq)
                figs += [f1, f2]
                spikes = agg[agg["spike"]]
        for f in figs:
            st.plotly_chart(f, width="stretch")
        st.dataframe(res.head(200), width="stretch")

        summary = pd.DataFrame({"Metric": ["Model", "Tweets"] + [f"% {l}" for l in LABELS],
                                "Value": [b["model"], f"{len(res):,}"] + [f"{dist[l] / len(res):.1%}" for l in LABELS]})
        e1, e2 = st.columns(2)
        e1.download_button("⬇ Download results (CSV)", res.to_csv(index=False).encode("utf-8"),
                           "sentiment_results.csv", "text/csv")
        e2.download_button("⬇ Download report (HTML)",
                           build_report("Sentiment report", summary, figs, spikes).encode("utf-8"),
                           "sentiment_report.html", "text/html")
