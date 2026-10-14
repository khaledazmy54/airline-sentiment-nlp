# Airline Sentiment & Topic Analysis

End-to-end NLP pipeline on the [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
dataset (about 14.6k tweets, 3 classes: negative / neutral / positive).

## Pipeline

| Stage | File | What it does |
|---|---|---|
| 1. Preprocessing | `preprocess.py` | Cleaning, TweetTokenizer, POS-aware lemmatization, hashtag/mention extraction, TF-IDF (sublinear tf) |
| 2. EDA | `eda.ipynb` / `eda.py` | Sentiment by airline and over time, n-grams, mention/retweet networks, anomaly detection, geography |
| 3. Supervised ML | `supervised.py` | Logistic Regression, Random Forest (1,2)-grams, Linear SVM; stratified CV; class-imbalance strategies |
| 4. Unsupervised ML | `unsupervised.py` | LDA + pyLDAvis, KMeans (k=10), NMF, human topic review |
| 5. Deep learning | `deep_learning.py` | DistilBERT, BiLSTM, TextCNN, attention BiLSTM; comparison with classical models |
| 6. Dashboard | `app.py` | Streamlit app: live prediction, topics, similar tweets, trends, spike alerts, export |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Download Tweets.csv from the Kaggle link above, then:
python preprocess.py --input Tweets.csv --output-dir artifacts
python supervised.py --artifacts artifacts
python unsupervised.py --artifacts artifacts
python deep_learning.py --artifacts artifacts --model all   # GPU recommended
streamlit run app.py
```

`artifacts/` and `figures/` are generated and git-ignored.

## Notes

- The dataset is not included in this repo; download it from Kaggle.
- Macro-F1 is the main metric because the classes are imbalanced (negative is the majority).
- Dataset caveats: about a week of data (Feb 2015), sparse geotags, and no true retweet-of field.
