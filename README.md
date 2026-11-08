# MpambolyMivoatry : pipeline MLOps (irrigation intelligente)

Données **simulées** : à remplacer par les mesures réelles du pilote.

## Démarrage rapide
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python generate_data.py        # données simulées (ou dvc repro)
python train.py                # entraîne, log MLflow, écrit models/model.pkl
pytest -q                      # tests de l'API
uvicorn app:app --reload       # API sur http://localhost:8000/docs
mlflow ui                      # suivi des expériences
python monitor.py              # rapport de drift -> reports/drift_report.html
```

## Docker
```bash
docker build -t mpamboly-api .
docker run -p 8000:8000 mpamboly-api
```

## DVC (versionnage des données)
```bash
git init && dvc init
dvc repro                      # rejoue generate + train
dvc add data/irrigation_reference.csv   # si vous utilisez de vraies données
```

## Étapes du schéma
1 Entraînement : train.py, DVC, MLflow · 2 Code : app.py · 3 CI/CD : .github/workflows/ci.yml
4 Build : Dockerfile · 5 Déploiement : à brancher · 6 Monitoring : monitor.py
