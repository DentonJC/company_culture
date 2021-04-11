import os
import pickle

import numpy as np
import optuna
import pandas as pd
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from utils import process_text


def objective(trial: Trial, X_train, X_test, y_train, y_test) -> float:
    params = {
        "booster": "gbtree",
        "n_estimators": trial.suggest_int("n_estimators", 0, 1000),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "reg_alpha": trial.suggest_int("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_int("reg_lambda", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 0, 5),
        "gamma": trial.suggest_int("gamma", 0, 5),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.5),
        "colsample_bytree": trial.suggest_discrete_uniform(
            "colsample_bytree", 0.1, 1, 0.01
        ),
        "nthread": -1,
        "use_label_encoder": False,
        "eval_metric": "logloss",
    }

    model = MultiOutputRegressor(XGBRegressor(**params))
    model.fit(X_train, y_train)

    n_scores = cross_val_score(
        model, X_test, y_test, scoring="neg_mean_absolute_error", cv=3, n_jobs=-1
    )

    return np.mean(abs(n_scores))


df = pd.read_csv("data/culture500_data.csv", encoding="ISO-8859-1")
df = df.set_index(["company", "value"])["sentiment_score"].unstack()

companies_names = list(df.index.values)


infile = open("data/companies.txt", "rb")
companies = pickle.load(infile)
infile.close()

y = df.values
idx = [i for i in range(len(companies)) if "error" not in companies[i]]
companies = np.array(companies)
X = companies[idx]
y = y[idx]

ngram_range = (1, 1)
# min_df=5
max_features = 500
vect = TfidfVectorizer(
    tokenizer=process_text,
    analyzer="word",
    stop_words="english",
    use_idf=True,
    smooth_idf=True,
    # min_df=min_df,
    ngram_range=ngram_range,
    max_features=max_features,
    norm="l2",
)
vect.fit(X)
X = vect.transform(X).toarray()

mask = ~np.any(np.isnan(X), axis=1)
X = X[mask]
y = y[mask]

mask = ~np.any(np.isnan(y), axis=1)
X = X[mask]
y = y[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1337
)


if not os.path.exists("search"):
    os.makedirs("search")

study = optuna.create_study(
    direction="minimize",
    sampler=TPESampler(seed=1337),
    study_name="res",
    storage="sqlite:///search/res.db",
    load_if_exists=True,
)
study.optimize(
    lambda trial: objective(trial, X_train, X_test, y_train, y_test),
    n_trials=1,
    show_progress_bar=True,
)
df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
df.to_csv("search/res.csv", sep="\t")

# open text file in read mode
letter_file = open("data/letter.txt", "r")
letter = letter_file.read().replace("\n", " ")
letter_file.close()

model = MultiOutputRegressor(XGBRegressor(**study.best_params))
model.fit(X, y)
preds = model.predict(vect.transform([letter]))

similarity = cosine_similarity(y, preds).flatten()
idx = sorted(range(len(similarity)), key=lambda i: similarity[i])[-10:]
print(np.array(similarity)[idx])
print(np.array(companies_names)[idx])
