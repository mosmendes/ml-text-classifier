from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.predict import predict_sentiment

app = FastAPI(
    title="Sentiment Classifier API",
    description="API para classificação de sentimentos em textos",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Texto para análise de sentimento")

class PredictionResponse(BaseModel):
    prediction: str

@app.get("/")
def root():
    return {"message": "API de classificação de sentimentos online"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="O texto enviado está vazio.")

    try:
        prediction = predict_sentiment(text)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao realizar predição: {str(e)}")