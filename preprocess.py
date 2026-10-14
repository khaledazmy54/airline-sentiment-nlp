"""
Stage 1: Text preprocessing pipeline for Twitter US Airline Sentiment.

Usage:
    python preprocess.py --input Tweets.csv --output-dir artifacts

Outputs (in --output-dir):
    clean_tweets.csv       full dataframe with cleaned text, tokens, lemmas, features
    X_train.npz / X_test.npz   TF-IDF matrices (sublinear_tf=True)
    y_train.csv / y_test.csv   labels
    tfidf_vectorizer.joblib    fitted vectorizer (train data only, no leakage)
"""
import argparse
import html
import re
from pathlib import Path

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

for pkg in ["wordnet", "omw-1.4", "stopwords",
            "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    nltk.download(pkg, quiet=True)

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")

# Keep negations: they carry sentiment ("not good", "never again").
KEEP = {"no", "not", "nor", "never", "nothing", "n't", "against", "off", "down"}
STOPWORDS = set(stopwords.words("english")) - KEEP | {"rt", "amp"}

tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
lemmatizer = WordNetLemmatizer()


def extract_entities(text: str):
    """Hashtags and mentions are extracted from the RAW text, before cleaning."""
    hashtags = [h.lower() for h in HASHTAG_RE.findall(text)]
    mentions = [m.lower() for m in MENTION_RE.findall(text)]
    return hashtags, mentions


def clean_text(text: str) -> str:
    text = html.unescape(text).lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    text = text.replace("n't", " not")          # "didn't" -> "did not"
    text = NON_ALPHA_RE.sub(" ", text)
    return MULTI_SPACE_RE.sub(" ", text).strip()


def to_wordnet_pos(treebank_tag: str):
    return {"J": wordnet.ADJ, "V": wordnet.VERB,
            "N": wordnet.NOUN, "R": wordnet.ADV}.get(treebank_tag[0], wordnet.NOUN)


def tokenize_and_lemmatize(clean: str):
    tokens = [t for t in tokenizer.tokenize(clean) if len(t) > 1 or t in {"i"}]
    tagged = nltk.pos_tag(tokens)               # POS-aware lemmatization
    lemmas = [lemmatizer.lemmatize(tok, to_wordnet_pos(tag)) for tok, tag in tagged]
    lemmas = [l for l in lemmas if l not in STOPWORDS]
    return tokens, lemmas


def parse_coord(val):
    """tweet_coord looks like '[40.64, -73.77]' or NaN."""
    if not isinstance(val, str):
        return (None, None)
    nums = re.findall(r"-?\d+\.\d+", val)
    return (float(nums[0]), float(nums[1])) if len(nums) == 2 else (None, None)


def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df = df.dropna(subset=["text", "airline_sentiment"]).drop_duplicates(subset="text")
    df = df.reset_index(drop=True)
    print(f"Loaded {len(df):,} tweets after dropping nulls/duplicates")

    # --- Entity extraction (raw text) ---
    ents = df["text"].apply(extract_entities)
    df["hashtags"] = ents.apply(lambda e: e[0])
    df["mentions"] = ents.apply(lambda e: e[1])
    df["n_hashtags"] = df["hashtags"].str.len()
    df["n_mentions"] = df["mentions"].str.len()

    # --- Cleaning, tokenization, lemmatization ---
    df["clean_text"] = df["text"].apply(clean_text)
    toks = df["clean_text"].apply(tokenize_and_lemmatize)
    df["tokens"] = toks.apply(lambda t: t[0])
    df["lemmas"] = toks.apply(lambda t: t[1])
    df["lemma_text"] = df["lemmas"].apply(" ".join)

    # --- Extra features for EDA ---
    df["tweet_len"] = df["text"].str.len()
    df["has_url"] = df["text"].str.contains(URL_RE).astype(int)
    df["is_retweet"] = df["text"].str.startswith("RT @").astype(int)
    df["tweet_created"] = pd.to_datetime(df["tweet_created"], errors="coerce", utc=True)
    coords = df["tweet_coord"].apply(parse_coord)
    df["lat"] = coords.apply(lambda c: c[0])
    df["lon"] = coords.apply(lambda c: c[1])

    df = df[df["lemma_text"].str.len() > 0].reset_index(drop=True)

    # --- Stratified split BEFORE fitting TF-IDF (avoids leakage) ---
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, stratify=df["airline_sentiment"], random_state=42
    )
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"

    vectorizer = TfidfVectorizer(
        sublinear_tf=True,      # tf -> 1 + log(tf)
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        max_features=50_000,
    )
    X_train = vectorizer.fit_transform(df.loc[train_idx, "lemma_text"])
    X_test = vectorizer.transform(df.loc[test_idx, "lemma_text"])

    sparse.save_npz(out / "X_train.npz", X_train)
    sparse.save_npz(out / "X_test.npz", X_test)
    df.loc[train_idx, "airline_sentiment"].to_csv(out / "y_train.csv", index=False)
    df.loc[test_idx, "airline_sentiment"].to_csv(out / "y_test.csv", index=False)
    joblib.dump(vectorizer, out / "tfidf_vectorizer.joblib")
    df.to_csv(out / "clean_tweets.csv", index=False)

    print(f"TF-IDF shape: train {X_train.shape}, test {X_test.shape}")
    print(df["airline_sentiment"].value_counts(normalize=True).round(3))
    print(f"Tweets with coordinates: {df['lat'].notna().sum():,}")
    print(f"Saved artifacts to {out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", default="artifacts")
    main(ap.parse_args())
