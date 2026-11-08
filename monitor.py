"""Étape 6 : rapport de data drift (Evidently) : référence vs données récentes."""
import sys

from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from common import FEATURES, load

current_path = sys.argv[1] if len(sys.argv) > 1 else "data/irrigation_drift.csv"
ref = load("data/irrigation_reference.csv")[FEATURES]
cur = load(current_path)[FEATURES]

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref, current_data=cur)
report.save_html("reports/drift_report.html")

res = report.as_dict()["metrics"][0]["result"]
print(f"Colonnes en dérive : {res['number_of_drifted_columns']}/{res['number_of_columns']}")
if res["dataset_drift"]:
    print("DRIFT détecté -> ré-entraîner le modèle (dvc repro)")
