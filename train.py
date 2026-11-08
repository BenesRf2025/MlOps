"""Étape 1 : entraînement (scikit-learn + MLflow). Produit models/model.pkl."""
import json

import joblib
import mlflow
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from common import FEATURES, NUM, TARGET, load

p = yaml.safe_load(open("params.yaml"))["train"]
df = load("data/irrigation_reference.csv").sort_values("timestamp")

# découpage temporel : on teste sur la fin de la période
cut = int(len(df) * (1 - p["test_size"]))
train, test = df.iloc[:cut], df.iloc[cut:]

model = Pipeline(
    [
        (
            "prep",
            ColumnTransformer(
                [("cat", OneHotEncoder(handle_unknown="ignore"), ["culture"])],
                remainder="passthrough",
            ),
        ),
        (
            "clf",
            RandomForestClassifier(
                n_estimators=p["n_estimators"],
                max_depth=p["max_depth"],
                class_weight="balanced",
                random_state=p["seed"],
            ),
        ),
    ]
)

mlflow.set_experiment("mpamboly-irrigation")
with mlflow.start_run():
    mlflow.log_params(p)
    model.fit(train[FEATURES], train[TARGET])
    pred = model.predict(test[FEATURES])
    metrics = {
        "precision": precision_score(test[TARGET], pred, zero_division=0),
        "recall": recall_score(test[TARGET], pred, zero_division=0),
        "f1": f1_score(test[TARGET], pred, zero_division=0),
    }
    mlflow.log_metrics(metrics)
    joblib.dump(model, "models/model.pkl")
    mlflow.log_artifact("models/model.pkl")

json.dump(metrics, open("metrics.json", "w"), indent=2)
print(metrics)
