"""
Stage 4: Unsupervised topic discovery.

Run modelling:
    python unsupervised.py --artifacts artifacts

After humans fill in review_sheet.csv and intrusion_task.csv:
    python unsupervised.py --artifacts artifacts --score-review

Methods
    LDA   (gensim) -- sweep 15..20 topics, pick best c_v coherence, pyLDAvis export
    NMF   (sklearn) -- additive decomposition of TF-IDF
    KMeans (sklearn) -- k=10 on TF-IDF vectors
All three share one vocabulary so their topic words can be compared with the same
c_v coherence metric.

Outputs (in --artifacts):
    lda_vis.html                interactive pyLDAvis
    topic_summary.csv           top words + coherence per topic, per method
    method_comparison.csv       coherence / silhouette per method
    doc_topics.csv              topic assignment per tweet (used by the Streamlit app)
    topic_sentiment.html/.csv   which topics drive negative sentiment
    review_sheet.csv            human rating sheet (label, coherence 1-5, distinctness 1-5)
    intrusion_task.csv/_key.csv word-intrusion test for human validation
    lda_model*, dictionary.gensim, nmf.joblib, kmeans.joblib, tfidf_topics.joblib
"""
import argparse
import random
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import pyLDAvis
import pyLDAvis.gensim_models
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
SEED = 42
TOPN = 10
# Words that appear in nearly every airline tweet and swamp topics. Tune freely.
DOMAIN_STOPWORDS = {"flight", "get", "go", "amp", "u", "would", "one", "say", "make",
                    "take", "also", "im", "us", "need", "know", "want"}


# --------------------------------------------------------------------------- data
def prepare(art: Path):
    df = pd.read_csv(art / "clean_tweets.csv")
    df["lemma_text"] = df["lemma_text"].fillna("")
    toks = [[w for w in t.split() if w not in DOMAIN_STOPWORDS] for t in df["lemma_text"]]

    dictionary = Dictionary(toks)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    vocab = set(dictionary.token2id)
    toks = [[w for w in t if w in vocab] for t in toks]

    keep = [i for i, t in enumerate(toks) if len(t) >= 3]     # drop near-empty docs
    df = df.iloc[keep].reset_index(drop=True)
    toks = [toks[i] for i in keep]
    corpus = [dictionary.doc2bow(t) for t in toks]
    print(f"{len(df):,} tweets, vocabulary {len(vocab):,}")
    return df, toks, dictionary, corpus


def coherence(topic_words, toks, dictionary):
    cm = CoherenceModel(topics=topic_words, texts=toks, dictionary=dictionary,
                        coherence="c_v", processes=1)
    return cm.get_coherence(), cm.get_coherence_per_topic()


# --------------------------------------------------------------------------- models
def fit_lda(corpus, dictionary, toks, k_min, k_max):
    best = None
    for k in range(k_min, k_max + 1):
        lda = LdaModel(corpus, num_topics=k, id2word=dictionary, passes=10,
                       iterations=100, alpha="auto", eta="auto",
                       random_state=SEED, eval_every=None)
        words = [[w for w, _ in lda.show_topic(i, TOPN)] for i in range(k)]
        score, per_topic = coherence(words, toks, dictionary)
        print(f"LDA k={k}: c_v={score:.3f}")
        if best is None or score > best[0]:
            best = (score, k, lda, words, per_topic)
    print(f"-> chosen LDA k={best[1]}")
    return best


def top_words_from_matrix(components, terms, n=TOPN):
    return [[terms[j] for j in np.argsort(row)[-n:][::-1]] for row in components]


# --------------------------------------------------------------------------- human review
def representative_tweets(scores, texts, n=5):
    """scores: (n_docs, n_topics) affinity matrix -> top-n raw tweets per topic."""
    return [" || ".join(texts[i].replace("\n", " ") for i in np.argsort(scores[:, t])[-n:][::-1])
            for t in range(scores.shape[1])]


def build_review_files(art, summaries):
    """summaries: {method: (top_words, per_topic_coh, rep_tweets)}"""
    rows, intr_rows, key_rows = [], [], []
    rng = random.Random(SEED)
    for method, (words, coh, reps) in summaries.items():
        for t, w in enumerate(words):
            rows.append({"method": method, "topic": t, "top_words": ", ".join(w),
                         "auto_c_v": round(coh[t], 3), "example_tweets": reps[t],
                         "human_label": "", "coherence_1_to_5": "",
                         "distinctness_1_to_5": "", "notes": ""})
            # word-intrusion: 5 own top words + 1 high-ranked word from another topic
            others = [o for o in range(len(words)) if o != t]
            pool = [x for o in others for x in words[o][:3] if x not in w]
            if pool:
                intruder = rng.choice(pool)
                options = w[:5] + [intruder]
                rng.shuffle(options)
                intr_rows.append({"method": method, "topic": t,
                                  "options": " | ".join(options), "guess_intruder": ""})
                key_rows.append({"method": method, "topic": t, "intruder": intruder})
    pd.DataFrame(rows).to_csv(art / "review_sheet.csv", index=False)
    pd.DataFrame(intr_rows).to_csv(art / "intrusion_task.csv", index=False)
    pd.DataFrame(key_rows).to_csv(art / "intrusion_task_key.csv", index=False)


def score_review(art):
    sheet = pd.read_csv(art / "review_sheet.csv")
    for c in ["coherence_1_to_5", "distinctness_1_to_5"]:
        sheet[c] = pd.to_numeric(sheet[c], errors="coerce")
    print("Human ratings (mean, 1-5):")
    print(sheet.groupby("method")[["coherence_1_to_5", "distinctness_1_to_5"]]
               .mean().round(2))
    print("\nCorrelation between automatic c_v and human coherence rating:")
    print(sheet[["auto_c_v", "coherence_1_to_5"]].corr().iloc[0, 1].round(3))

    task = pd.read_csv(art / "intrusion_task.csv")
    key = pd.read_csv(art / "intrusion_task_key.csv")
    m = task.merge(key, on=["method", "topic"])
    m["correct"] = m["guess_intruder"].astype(str).str.strip().str.lower() == \
        m["intruder"].astype(str).str.lower()
    print("\nWord-intrusion accuracy (chance is about 17%):")
    print(m.groupby("method")["correct"].mean().round(2))
    bad = sheet[(sheet["coherence_1_to_5"] <= 2)][["method", "topic", "top_words", "notes"]]
    print("\nLow-rated topics to revisit (merge, drop or re-tune):")
    print(bad.to_string(index=False))


# --------------------------------------------------------------------------- main
def main(args):
    art = Path(args.artifacts)
    if args.score_review:
        return score_review(art)

    df, toks, dictionary, corpus = prepare(art)
    texts = df["text"].tolist() if "text" in df else df["lemma_text"].tolist()

    # ---- LDA ----
    lda_score, k_lda, lda, lda_words, lda_per = fit_lda(
        corpus, dictionary, toks, args.lda_min, args.lda_max)
    lda_theta = np.zeros((len(corpus), k_lda))
    for i, doc in enumerate(corpus):
        for t, p in lda.get_document_topics(doc, minimum_probability=0):
            lda_theta[i, t] = p
    lda.save(str(art / "lda_model"))
    dictionary.save(str(art / "dictionary.gensim"))
    vis = pyLDAvis.gensim_models.prepare(lda, corpus, dictionary, sort_topics=False)
    pyLDAvis.save_html(vis, str(art / "lda_vis.html"))

    # ---- shared TF-IDF space for NMF and KMeans ----
    tfidf = TfidfVectorizer(vocabulary=sorted(dictionary.token2id), sublinear_tf=True,
                            token_pattern=r"\S+")
    X = tfidf.fit_transform(df["lemma_text"].apply(
        lambda s: " ".join(w for w in s.split() if w in dictionary.token2id)))
    terms = tfidf.get_feature_names_out()

    # ---- NMF ----
    nmf = NMF(n_components=args.nmf_k, init="nndsvd", max_iter=400, random_state=SEED)
    W = nmf.fit_transform(X)
    nmf_words = top_words_from_matrix(nmf.components_, terms)
    nmf_score, nmf_per = coherence(nmf_words, toks, dictionary)

    # ---- KMeans ----
    km = KMeans(n_clusters=args.kmeans_k, n_init=10, random_state=SEED).fit(X)
    km_words = top_words_from_matrix(km.cluster_centers_, terms)
    km_score, km_per = coherence(km_words, toks, dictionary)
    sample = np.random.RandomState(SEED).choice(X.shape[0], min(3000, X.shape[0]), replace=False)
    sil = silhouette_score(X[sample], km.labels_[sample], metric="cosine")
    km_affinity = -km.transform(X)             # closer to centroid = higher

    joblib.dump(nmf, art / "nmf.joblib")
    joblib.dump(km, art / "kmeans.joblib")
    joblib.dump(tfidf, art / "tfidf_topics.joblib")

    # ---- comparison + summaries ----
    pd.DataFrame([
        {"method": "lda", "n_topics": k_lda, "c_v": lda_score, "silhouette": np.nan},
        {"method": "nmf", "n_topics": args.nmf_k, "c_v": nmf_score, "silhouette": np.nan},
        {"method": "kmeans", "n_topics": args.kmeans_k, "c_v": km_score, "silhouette": sil},
    ]).to_csv(art / "method_comparison.csv", index=False)

    summaries = {
        "lda": (lda_words, lda_per, representative_tweets(lda_theta, texts)),
        "nmf": (nmf_words, nmf_per, representative_tweets(W, texts)),
        "kmeans": (km_words, km_per, representative_tweets(km_affinity, texts)),
    }
    pd.DataFrame([{"method": m, "topic": t, "c_v": round(c[t], 3), "top_words": ", ".join(w[t])}
                  for m, (w, c, _) in summaries.items() for t in range(len(w))]
                 ).to_csv(art / "topic_summary.csv", index=False)
    build_review_files(art, summaries)

    # ---- assignments for downstream use ----
    id_col = "tweet_id" if "tweet_id" in df else None
    out = pd.DataFrame({
        "row": df.index,
        "lda_topic": lda_theta.argmax(1), "lda_prob": lda_theta.max(1),
        "nmf_topic": W.argmax(1), "kmeans_cluster": km.labels_,
    })
    if id_col:
        out.insert(0, "tweet_id", df[id_col].values)
    out.to_csv(art / "doc_topics.csv", index=False)

    # ---- which topics drive negativity? ----
    ts = (pd.crosstab(out["lda_topic"], df["airline_sentiment"], normalize="index")
            .join(out["lda_topic"].value_counts().rename("n_tweets")))
    ts["top_words"] = [", ".join(lda_words[t][:6]) for t in ts.index]
    ts.sort_values("negative", ascending=False).round(3).to_csv(art / "topic_sentiment.csv")
    long = ts.reset_index().melt(id_vars=["lda_topic", "n_tweets", "top_words"],
                                 value_vars=[c for c in ["negative", "neutral", "positive"] if c in ts],
                                 var_name="sentiment", value_name="share")
    fig = px.bar(long, x="lda_topic", y="share", color="sentiment", hover_data=["top_words"],
                 color_discrete_map={"negative": "#d62728", "neutral": "#7f7f7f",
                                     "positive": "#2ca02c"},
                 title="Sentiment mix by LDA topic")
    fig.write_html(art / "topic_sentiment.html")

    print("\nMethod comparison:")
    print(pd.read_csv(art / "method_comparison.csv").round(3).to_string(index=False))
    print("\nNext: have 2+ people fill review_sheet.csv and intrusion_task.csv, then run "
          "with --score-review")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--lda-min", type=int, default=15)
    ap.add_argument("--lda-max", type=int, default=20)
    ap.add_argument("--nmf-k", type=int, default=15)
    ap.add_argument("--kmeans-k", type=int, default=10)
    ap.add_argument("--score-review", action="store_true")
    main(ap.parse_args())
