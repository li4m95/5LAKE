import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mlops import api

HOUSE = {
    "OverallQual": 7,
    "GrLivArea": 1800,
    "GarageCars": 2,
    "TotalBsmtSF": 900,
    "YearBuilt": 2005,
}
HEADERS = {"X-API-Key": "test-key"}


class FakeModel:
    def predict(self, frame):
        return np.full(len(frame), 250_000.0)


class FakeStore:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.predictions: dict[uuid.UUID, float] = {}
        self.feedback: dict[uuid.UUID, float] = {}

    def log_prediction(self, prediction_id, model_version, features, predicted_price):
        if self.fail:
            raise RuntimeError("database is down")
        self.predictions[prediction_id] = predicted_price

    def add_feedback(self, prediction_id, actual_price) -> bool:
        if prediction_id not in self.predictions:
            return False
        self.feedback[prediction_id] = actual_price
        return True


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def client(monkeypatch, store):
    monkeypatch.setenv("API_KEYS", "test-key")
    api.app.dependency_overrides[api.get_model] = lambda: api.LoadedModel(FakeModel(), "7")
    api.app.dependency_overrides[api.get_store] = lambda: store
    # No "with" block: the lifespan (MLflow, Postgres) does not run in unit tests.
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ready_reports_model_version(client):
    assert client.get("/ready").json()["model_version"] == "7"


def test_predict_requires_api_key(client):
    assert client.post("/predict", json=HOUSE).status_code == 401


def test_predict_rejects_wrong_api_key(client):
    response = client.post("/predict", json=HOUSE, headers={"X-API-Key": "nope"})
    assert response.status_code == 401


def test_no_configured_keys_rejects_everything(client, monkeypatch):
    monkeypatch.delenv("API_KEYS")
    assert client.post("/predict", json=HOUSE, headers=HEADERS).status_code == 503


def test_predict_rejects_out_of_range_input(client):
    response = client.post("/predict", json={**HOUSE, "OverallQual": 100}, headers=HEADERS)
    assert response.status_code == 422


def test_predict_rejects_unknown_fields(client):
    response = client.post("/predict", json={**HOUSE, "Pool": 1}, headers=HEADERS)
    assert response.status_code == 422


def test_predict_returns_price_and_stores_it(client, store):
    response = client.post("/predict", json=HOUSE, headers=HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_sale_price"] == 250_000.0
    assert body["model_version"] == "7"
    assert uuid.UUID(body["prediction_id"]) in store.predictions
    assert "X-Request-ID" in response.headers


def test_predict_still_answers_when_storage_fails(client):
    api.app.dependency_overrides[api.get_store] = lambda: FakeStore(fail=True)
    response = client.post("/predict", json=HOUSE, headers=HEADERS)
    assert response.status_code == 200


def test_feedback_for_known_prediction(client, store):
    prediction_id = client.post("/predict", json=HOUSE, headers=HEADERS).json()["prediction_id"]
    body = {"prediction_id": prediction_id, "actual_sale_price": 240_000}
    assert client.post("/feedback", json=body, headers=HEADERS).status_code == 201
    assert store.feedback[uuid.UUID(prediction_id)] == 240_000


def test_feedback_for_unknown_prediction(client):
    body = {"prediction_id": str(uuid.uuid4()), "actual_sale_price": 240_000}
    assert client.post("/feedback", json=body, headers=HEADERS).status_code == 404
