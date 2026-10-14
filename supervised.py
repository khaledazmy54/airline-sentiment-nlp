"""
Stage 3: Supervised sentiment classification.

Usage:
    python supervised.py --artifacts artifacts --cv 5

Models: Logistic Regression (baseline), Random Forest (1,2)-grams, Linear SVM.
Imbalance strategies compared: none | class_weight | SMOTE (applied inside each CV fold).

TF-IDF is part of every pipeline, so it is re-fitted on each CV training fold.
That avoids the vocabulary/IDF leakage you'd get from cross-validating on a
matrix that was fitted on the whole training set.

Outputs (in --artifacts):
    cv_results.csv, final_test_results.csv, test_predictions.csv,
    confusion_matrices.html, top_features.csv, best_<model>.joblib
"""
import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate)
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
SEED = 42
LABELS = ["negative", "neutral", "positive"]
STRATEGIES = ["none", "class_weight", "smote"]
SCORING = {"macro_f1": "f1_macro", "weighted_f1": "f1_weighted", "accuracy": "accuracy"}


def build_pipeline(model: str, strategy: str) -> Pipeline:
    cw = "balanced" if strategy == "class_weight" else None

    if model == "logreg":      # baseline: unigram TF-IDF
        vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 1), min_df=2)
        clf = LogisticRegression(max_iter=2000, class_weight=cw, random_state=SEED)
    elif model == "random_forest":   # (1,2)-grams as required
        vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=3,
                              max_features=10_000)
        clf = RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=SEED,
            class_weight="balanced_subsample" if cw else None)
    elif model == "linear_svm":
        vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2)
        clf = LinearSVC(class_weight=cw, random_state=SEED)
    else:
        raise ValueError(model)

    steps = [("tfidf", vec)]
    if strategy == "smote":
        steps.append(("smote", SMOTE(random_state=SEED)))
    steps.append(("clf", clf))
    return Pipeline(steps)


def main(args):
    art = Path(args.artifacts)
    df = pd.read_csv(art / "clean_tweets.csv")
    df["lemma_text"] = df["lemma_text"].fillna("")
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    X_train, y_train = train["lemma_text"], train["airline_sentiment"]
    X_test, y_test = test["lemma_text"], test["airline_sentiment"]
    print(f"Train {len(train):,} | Test {len(test):,}")
    print(y_train.value_counts(normalize=True).round(3).to_dict())

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=SEED)

    # ---- 1. Compare models x imbalance strategies with stratified CV ----
    rows = []
    for model in ["logreg", "random_forest", "linear_svm"]:
        for strategy in STRATEGIES:
            res = cross_validate(build_pipeline(model, strategy), X_train, y_train,
                                 cv=cv, scoring=SCORING, n_jobs=1)
            row = {"model": model, "strategy": strategy}
            for k in SCORING:
                row[f"{k}_mean"] = res[f"test_{k}"].mean()
                row[f"{k}_std"] = res[f"test_{k}"].std()
            rows.append(row)
            print(f"{model:14s} {strategy:13s} macroF1 "
                  f"{row['macro_f1_mean']:.3f} +/- {row['macro_f1_std']:.3f}")
    cv_df = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)
    cv_df.to_csv(art / "cv_results.csv", index=False)

    # ---- 2. Tune C for LR / SVM using each model's best strategy ----
    best_strategy = (cv_df.sort_values("macro_f1_mean", ascending=False)
                          .groupby("model").first()["strategy"].to_dict())
    final_models = {}
    for model, strategy in best_strategy.items():
        pipe = build_pipeline(model, strategy)
        if model in ("logreg", "linear_svm"):
            gs = GridSearchCV(pipe, {"clf__C": [0.1, 0.3, 1, 3, 10]},
                              scoring="f1_macro", cv=cv, n_jobs=-1)
            gs.fit(X_train, y_train)
            print(f"{model}: best C={gs.best_params_['clf__C']} "
                  f"(CV macro-F1 {gs.best_score_:.3f})")
            final_models[model] = gs.best_estimator_
        else:
            final_models[model] = pipe.fit(X_train, y_train)

    # ---- 3. One-time evaluation on the held-out test set ----
    results, preds = [], pd.DataFrame({"y_true": y_test.values})
    fig = make_subplots(rows=1, cols=len(final_models),
                        subplot_titles=list(final_models), horizontal_spacing=0.08)
    for i, (model, est) in enumerate(final_models.items(), start=1):
        y_pred = est.predict(X_test)
        preds[model] = y_pred
        results.append({
            "model": model, "strategy": best_strategy[model],
            "accuracy": accuracy_score(y_test, y_pred),
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
            "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
            **{f"f1_{l}": f1_score(y_test, y_pred, labels=[l], average="macro")
               for l in LABELS}})
        print(f"\n=== {model} ({best_strategy[model]}) ===")
        print(classification_report(y_test, y_pred, labels=LABELS, digits=3))
        cm = confusion_matrix(y_test, y_pred, labels=LABELS, normalize="true")
        fig.add_trace(go.Heatmap(z=cm, x=LABELS, y=LABELS, zmin=0, zmax=1,
                                 text=np.round(cm, 2), texttemplate="%{text}",
                                 colorscale="Blues", showscale=False), row=1, col=i)
        joblib.dump(est, art / f"best_{model}.joblib")
    fig.update_layout(title="Normalised confusion matrices (test set)", width=1100, height=400)
    fig.write_html(art / "confusion_matrices.html")

    pd.DataFrame(results).sort_values("macro_f1", ascending=False).to_csv(
        art / "final_test_results.csv", index=False)
    preds.to_csv(art / "test_predictions.csv", index=False)
    print(pd.DataFrame(results).round(3).to_string(index=False))

    # ---- 4. Interpretability: top weighted terms per class (linear SVM) ----
    svm = final_models["linear_svm"]
    terms = svm.named_steps["tfidf"].get_feature_names_out()
    coefs = svm.named_steps["clf"].coef_
    classes = svm.named_steps["clf"].classes_
    top = [{"class": c, "term": terms[j], "weight": coefs[k][j]}
           for k, c in enumerate(classes) for j in np.argsort(coefs[k])[-15:][::-1]]
    pd.DataFrame(top).to_csv(art / "top_features.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--cv", type=int, default=5)
    main(ap.parse_args())
