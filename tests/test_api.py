import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"


def test_predict_success(client: TestClient):
    payload = {
        "sepal_length": 6.2,
        "sepal_width": 2.8,
        "petal_length": 4.8,
        "petal_width": 1.8,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data["predicted_class"], int)
    assert isinstance(data["predicted_label"], str)
    assert data["predicted_label"]  


def test_predict_invalid_input(client: TestClient):

    payload = {
        "sepal_length": "wrong",
        "sepal_width": 2.8,
        "petal_length": 4.8,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422