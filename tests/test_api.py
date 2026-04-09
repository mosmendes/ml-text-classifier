from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_predict():
    payload = {"text": "eu adorei esse produto"}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "prediction" in response.json()