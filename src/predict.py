import joblib
import os

MODEL_PATH = "models/model.joblib"
VECTORIZER_PATH = "models/vectorizer.joblib"

def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}")
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(f"Vetorizador não encontrado em {VECTORIZER_PATH}")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

def predict_sentiment(text: str) -> str:
    model, vectorizer = load_artifacts()
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    return prediction