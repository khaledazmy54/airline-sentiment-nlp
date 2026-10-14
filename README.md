# ✈️ Airline Sentiment & Topic Analysis

An end-to-end NLP project that turns raw airline tweets into **sentiment predictions, discovered topics, and an interactive monitoring dashboard** with negative-spike alerts.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-DL-red)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-ff4b4b)

<!-- Add a screenshot or GIF of the dashboard here:
![Dashboard](docs/dashboard.png)
-->

## Overview

Customer feedback on social media is noisy, fast, and unstructured. This project builds the full pipeline a brand-monitoring team would need:

- clean and normalise tweets,
- classify them as **negative / neutral / positive**,
- discover what people are actually complaining about (topics),
- watch sentiment over time and **alert on sudden negative spikes**.

**Dataset:** [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) (Kaggle), about 14.6k tweets about six US airlines, February 2015. It is not included in this repo; download `Tweets.csv` from Kaggle.

## Features

| Stage | File | What it does |
|---|---|---|
| 1. Preprocessing | `preprocess.py` | Cleaning, `TweetTokenizer`, POS-aware lemmatization (negations kept), hashtag/mention extraction, TF-IDF with sublinear tf |
| 2. EDA | `eda.ipynb` / `eda.py` | Sentiment by airline and over time, n-grams per class, mention and retweet networks, anomaly detection, geographic view |
| 3. Supervised ML | `supervised.py` | Logistic Regression, Random Forest (1,2)-grams, Linear SVM; stratified CV; none / class-weight / SMOTE compared |
| 4. Unsupervised ML | `unsupervised.py` | LDA (15-20 topics) + pyLDAvis, KMeans (k=10), NMF, human topic review and word-intrusion test |
| 5. Deep learning | `deep_learning.py` | Fine-tuned DistilBERT, BiLSTM, TextCNN, attention BiLSTM; McNemar test vs classical models |
| 6. Dashboard | `app.py` | Live prediction, topic assignment, similar tweets, Plotly trends, spike alerts, CSV/HTML export |

## Results

<!-- Replace with your own numbers from artifacts/model_comparison.csv -->

| Model | Type | Accuracy | Macro-F1 |
|---|---|---|---|
| Logistic Regression | classical | … | … |
| Linear SVM | classical | … | … |
| Random Forest | classical | … | … |
| BiLSTM / TextCNN / Attention | deep | … | … |
| DistilBERT | transformer | … | … |

Macro-F1 is the main metric because the classes are imbalanced (negative is the majority). All models are scored on the same held-out test split.

## Quick start

```bash
git clone https://github.com/khaledazmy54/airline-sentiment-nlp.git
cd airline-sentiment-nlp

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download `Tweets.csv` from Kaggle into the project folder, then run the stages in order:

```bash
python preprocess.py --input Tweets.csv --output-dir artifacts
python supervised.py --artifacts artifacts
python unsupervised.py --artifacts artifacts
python deep_learning.py --artifacts artifacts --model all    # GPU recommended (e.g. Colab)
streamlit run app.py
```

`eda.ipynb` can be run any time after preprocessing. The dashboard works with whatever stages you have run: it uses DistilBERT if it was trained, otherwise the best classical model.

## Project structure

```
├── preprocess.py        # stage 1
├── eda.ipynb / eda.py   # stage 2
├── supervised.py        # stage 3
├── unsupervised.py      # stage 4
├── deep_learning.py     # stage 5
├── app.py               # stage 6 (Streamlit)
├── requirements.txt
└── artifacts/           # generated, git-ignored
```

## Design decisions

- **No data leakage:** the train/test split happens before TF-IDF is fitted, and TF-IDF and SMOTE live inside each CV fold.
- **Negations kept:** words like "not" and "never" are not removed as stopwords, since they flip sentiment.
- **Fair comparison:** classical and deep models share one test split; a validation slice of the training data is used for early stopping.
- **Spike detection uses only past data:** a time bucket is flagged when its negative share is more than *z* standard deviations above the previous window (no look-ahead).
- **Human validation for topics:** automatic coherence (c_v) is checked against human ratings and a word-intrusion test.

## Limitations

- The data covers about one week, so "trends" are short-term patterns.
- Few tweets carry coordinates, and there is no true retweet-of field.
- Neutral is the hardest class (smallest and most ambiguous).
- Attention weights show where a model looked, not a guaranteed explanation of its decision.

## Built by

Khaled · GitHub: [@khaledazmy54](https://github.com/khaledazmy54)
