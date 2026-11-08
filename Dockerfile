
# --- Étape 1 : entraîner le modèle pendant le build ---
FROM python:3.12-slim AS train
WORKDIR /app
RUN pip install --no-cache-dir pandas numpy scikit-learn joblib pyyaml
COPY common.py train.py params.yaml ./
COPY data/irrigation_reference.csv data/irrigation_reference.csv
RUN mkdir -p models && python train.py
 
# --- Étape 2 : image finale de l'API ---
FROM python:3.12-slim
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY app.py .
COPY --from=train /app/models/model.pkl models/model.pkl
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]