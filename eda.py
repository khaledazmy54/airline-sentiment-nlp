# %% [markdown]
# # Stage 2: Exploratory Data Analysis
# Input: `artifacts/clean_tweets.csv` produced by `preprocess.py`.
# Figures are saved as interactive HTML in `figures/`.

# %%
import ast
import numpy as np
from itertools import combinations
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.feature_extraction.text import CountVectorizer

ART = Path("artifacts")
FIG = Path("figures")
FIG.mkdir(exist_ok=True)
SENT_ORDER = ["negative", "neutral", "positive"]
COLORS = {"negative": "#d62728", "neutral": "#7f7f7f", "positive": "#2ca02c"}


def save(fig, name):
    fig.write_html(FIG / f"{name}.html")
    fig.show()


df = pd.read_csv(ART / "clean_tweets.csv", parse_dates=["tweet_created"])
for col in ["hashtags", "mentions", "tokens", "lemmas"]:
    df[col] = df[col].apply(ast.literal_eval)
df["lemma_text"] = df["lemma_text"].fillna("")
print(df.shape)
print("Date range:", df["tweet_created"].min(), "->", df["tweet_created"].max())

# %% [markdown]
# ## 1. Sentiment distribution: overall and by entity (airline)

# %%
overall = df["airline_sentiment"].value_counts().reindex(SENT_ORDER).reset_index()
overall.columns = ["sentiment", "count"]
save(px.bar(overall, x="sentiment", y="count", color="sentiment",
            color_discrete_map=COLORS, title="Overall sentiment distribution"),
     "01_overall")

by_airline = (df.groupby(["airline", "airline_sentiment"]).size()
                .groupby(level=0).apply(lambda s: s / s.sum())
                .rename("share").reset_index())
save(px.bar(by_airline, x="airline", y="share", color="airline_sentiment",
            category_orders={"airline_sentiment": SENT_ORDER},
            color_discrete_map=COLORS, barmode="stack",
            title="Sentiment share by airline"), "02_by_airline")

if "negativereason" in df.columns:
    reasons = (df[df["airline_sentiment"] == "negative"]
               .groupby(["airline", "negativereason"]).size().reset_index(name="n"))
    save(px.bar(reasons, x="airline", y="n", color="negativereason",
                title="Reasons for negative tweets, by airline"), "03_neg_reasons")

# %% [markdown]
# ## 2. Word frequency and n-grams per sentiment class
# Raw frequency is dominated by shared words ("flight", "get"), so we also
# compute a smoothed log-odds score to surface words that are *distinctive* per class.

# %%
def top_ngrams(texts, ngram, k=15):
    cv = CountVectorizer(ngram_range=(ngram, ngram), min_df=3)
    X = cv.fit_transform(texts)
    freqs = pd.Series(X.sum(axis=0).A1, index=cv.get_feature_names_out())
    return freqs.nlargest(k)


rows = []
for sent in SENT_ORDER:
    texts = df.loc[df["airline_sentiment"] == sent, "lemma_text"]
    for n, label in [(1, "unigram"), (2, "bigram")]:
        for term, cnt in top_ngrams(texts, n).items():
            rows.append((sent, label, term, cnt))
ng = pd.DataFrame(rows, columns=["sentiment", "type", "term", "count"])

for typ in ["unigram", "bigram"]:
    sub = ng[ng["type"] == typ]
    fig = px.bar(sub, x="count", y="term", color="sentiment", facet_col="sentiment",
                 color_discrete_map=COLORS, orientation="h",
                 category_orders={"sentiment": SENT_ORDER},
                 title=f"Top {typ}s per sentiment class")
    fig.update_yaxes(matches=None, showticklabels=True, autorange="reversed")
    fig.update_xaxes(matches=None)
    save(fig, f"04_top_{typ}s")

# Distinctive words: log-odds of word in class vs. all other classes (Dirichlet-smoothed)
cv = CountVectorizer(min_df=5)
X = cv.fit_transform(df["lemma_text"])
vocab = cv.get_feature_names_out()
distinct = {}
for sent in SENT_ORDER:
    mask = (df["airline_sentiment"] == sent).values
    a = pd.Series(X[mask].sum(axis=0).A1, index=vocab) + 0.5
    b = pd.Series(X[~mask].sum(axis=0).A1, index=vocab) + 0.5
    logodds = np.log(a / (a.sum() - a)) - np.log(b / (b.sum() - b))
    distinct[sent] = logodds.nlargest(10).index.tolist()
print(pd.DataFrame(distinct))

# %% [markdown]
# ## 3. Network analysis
# **Mention network:** nodes are handles, edges connect handles co-mentioned in a tweet.
# **Retweet network:** directed edges author -> retweeted account (tweets starting with `RT @`).

# %%
G = nx.Graph()
for ms in df["mentions"]:
    for u, v in combinations(sorted(set(ms)), 2):
        w = G.get_edge_data(u, v, {}).get("weight", 0)
        G.add_edge(u, v, weight=w + 1)
print(f"Mention graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

if G.number_of_nodes():
    deg = pd.Series(dict(G.degree(weight="weight"))).nlargest(15)
    print("Most connected handles:\n", deg)

    # Plot the core only (top 60 handles by weighted degree) to keep it readable
    core = G.subgraph(deg.index.union(
        pd.Series(dict(G.degree(weight="weight"))).nlargest(60).index))
    pos = nx.spring_layout(core, seed=42, k=0.6)
    comms = greedy_modularity_communities(core, weight="weight")
    node_comm = {n: i for i, c in enumerate(comms) for n in c}

    edge_x, edge_y = [], []
    for u, v in core.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
    fig = go.Figure([
        go.Scatter(x=edge_x, y=edge_y, mode="lines",
                   line=dict(width=0.5, color="#aaa"), hoverinfo="none"),
        go.Scatter(x=[pos[n][0] for n in core], y=[pos[n][1] for n in core],
                   mode="markers+text", text=list(core), textposition="top center",
                   marker=dict(size=[6 + 2 * core.degree(n) ** 0.5 for n in core],
                               color=[node_comm[n] for n in core],
                               colorscale="Turbo"))])
    fig.update_layout(title="Mention co-occurrence network (core, coloured by community)",
                      showlegend=False, xaxis_visible=False, yaxis_visible=False)
    save(fig, "05_mention_network")

rt = df[df["is_retweet"] == 1]
R = nx.DiGraph()
if "name" in df.columns:
    for author, ms in zip(rt["name"], rt["mentions"]):
        if ms:
            R.add_edge(author, ms[0])
print(f"Retweet graph: {R.number_of_nodes()} nodes, {R.number_of_edges()} edges "
      f"({len(rt)} retweets of {len(df)} tweets)")
if R.number_of_edges():
    print("Most retweeted accounts:\n",
          pd.Series(dict(R.in_degree())).nlargest(10))

# %% [markdown]
# ## 4. Temporal sentiment trends with anomaly detection
# The data covers only about a week, so we aggregate in 3-hour buckets and flag buckets
# whose negative share deviates >2 sigma from the *previous* 24 hours (no look-ahead).

# %%
ts = df.dropna(subset=["tweet_created"]).set_index("tweet_created").sort_index()
ts["is_neg"] = (ts["airline_sentiment"] == "negative").astype(int)

agg = ts.resample("3h").agg(neg_share=("is_neg", "mean"), n=("is_neg", "size"))
agg = agg[agg["n"] >= 30].copy()                      # ignore near-empty buckets
roll_mean = agg["neg_share"].rolling(8, min_periods=4).mean().shift(1)
roll_std = agg["neg_share"].rolling(8, min_periods=4).std().shift(1)
agg["z"] = (agg["neg_share"] - roll_mean) / roll_std
agg["anomaly"] = agg["z"].abs() > 2
print(agg[agg["anomaly"]])

fig = go.Figure()
fig.add_scatter(x=agg.index, y=agg["neg_share"], mode="lines+markers",
                name="Negative share", line_color=COLORS["negative"])
a = agg[agg["anomaly"]]
fig.add_scatter(x=a.index, y=a["neg_share"], mode="markers", name="Anomaly",
                marker=dict(size=14, symbol="x", color="black"))
fig.update_layout(title="Negative-tweet share (3h buckets) with anomalies",
                  yaxis_tickformat=".0%")
save(fig, "06_temporal_anomalies")

daily = (ts.groupby([pd.Grouper(freq="D"), "airline_sentiment"]).size()
           .reset_index(name="n"))
save(px.line(daily, x="tweet_created", y="n", color="airline_sentiment",
             color_discrete_map=COLORS, markers=True,
             title="Daily tweet volume by sentiment"), "07_daily")

# Per-airline daily negative share
air_daily = (ts.groupby([pd.Grouper(freq="D"), "airline"])["is_neg"].mean()
               .reset_index())
save(px.line(air_daily, x="tweet_created", y="is_neg", color="airline", markers=True,
             title="Daily negative share by airline"), "08_airline_daily")

# %% [markdown]
# ## 5. Geographic sentiment
# Only a small share of tweets carry coordinates. We map those, and fall back to
# `user_timezone` (much better coverage, coarser) for a regional view.

# %%
geo = df.dropna(subset=["lat", "lon"])
print(f"{len(geo):,} tweets with coordinates ({len(geo) / len(df):.1%})")
if len(geo):
    save(px.scatter_geo(geo, lat="lat", lon="lon", color="airline_sentiment",
                        color_discrete_map=COLORS, scope="north america",
                        hover_data=["airline"], title="Geotagged tweets by sentiment"),
         "09_geo_points")

tz = df.dropna(subset=["user_timezone"])
top_tz = tz["user_timezone"].value_counts().head(15).index
tz_share = (tz[tz["user_timezone"].isin(top_tz)]
            .groupby("user_timezone")["airline_sentiment"]
            .value_counts(normalize=True).rename("share").reset_index())
save(px.bar(tz_share, x="user_timezone", y="share", color="airline_sentiment",
            color_discrete_map=COLORS,
            category_orders={"airline_sentiment": SENT_ORDER},
            title="Sentiment share by user timezone (top 15)"), "10_timezone")

# %% [markdown]
# ## 6. Quick feature sanity checks

# %%
save(px.box(df, x="airline_sentiment", y="tweet_len", color="airline_sentiment",
            color_discrete_map=COLORS,
            category_orders={"airline_sentiment": SENT_ORDER},
            title="Tweet length by sentiment"), "11_length")
print(df.groupby("airline_sentiment")[["tweet_len", "n_hashtags", "n_mentions",
                                       "has_url", "is_retweet"]].mean().round(2))
