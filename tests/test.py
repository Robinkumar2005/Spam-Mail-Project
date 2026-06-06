from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the FastAPI application!"}

def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json() == {
        "model": "SVM_MODEL",
        "description": "A NLP model for detecting spam emails.",
        "Version": "V1"
    }

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "timestamp" in response.json()