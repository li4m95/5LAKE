"""FastAPI prediction service. Serves the champion model from the MLflow registry."""

import logging
import secrets
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any

import mlflow.sklearn
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from mlflow.tracking import MlflowClient
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field
from starlette.routing import Match

from mlops import settings
from mlops.db import PredictionStore
from mlops.logging_config import configure_logging, request_id_var

logger = logging.getLogger("mlops.api")

HTTP_REQUESTS = Counter("http_requests", "HTTP requests handled", ["method", "route", "status"])
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.5),
)
PREDICTIONS = Counter("predictions", "Predictions served", ["model_version"])
PREDICTED_PRICE = Histogram(
    "predicted_sale_price",
    "Predicted sale prices",
    buckets=(50_000, 100_000, 150_000, 200_000, 250_000, 300_000, 400_000, 600_000),
)
PREDICTION_LOG_FAILURES = Counter(
    "prediction_log_failures", "Predictions that could not be written to the database"
)
MODEL_LOADED = Gauge("model_loaded", "1 when a champion model is loaded, otherwise 0")

# ---------- model loading ----------


@dataclass
class LoadedModel:
    model: Any
    version: str


def load_champion() -> LoadedModel:
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    champion = client.get_model_version_by_alias(settings.MODEL_NAME, settings.CHAMPION_ALIAS)
    # Load by explicit version, so the served version is known exactly.
    model = mlflow.sklearn.load_model(f"models:/{settings.MODEL_NAME}/{champion.version}")
    return LoadedModel(model=model, version=str(champion.version))


class ModelHolder:
    """Holds the loaded model and retries loading at most every `retry_seconds`."""

    def __init__(self, loader: Callable[[], LoadedModel], retry_seconds: float = 30.0) -> None:
        self._loader = loader
        self._retry_seconds = retry_seconds
        self._lock = threading.Lock()
        self._last_attempt = float("-inf")
        self.loaded: LoadedModel | None = None

    def _due(self) -> bool:
        return time.monotonic() - self._last_attempt >= self._retry_seconds

    def get(self) -> LoadedModel | None:
        if self.loaded is None and self._due():
            with self._lock:
                if self.loaded is None and self._due():
                    self._last_attempt = time.monotonic()
                    try:
                        self.loaded = self._loader()
                        logger.info("Champion loaded", extra={"model_version": self.loaded.version})
                    except Exception:
                        logger.exception("Could not load the champion model")
        MODEL_LOADED.set(1 if self.loaded else 0)
        return self.loaded


model_holder = ModelHolder(load_champion)

# ---------- dependencies ----------


def get_model() -> LoadedModel:
    loaded = model_holder.get()
    if loaded is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "No champion model is loaded.")
    return loaded


def get_store(request: Request) -> PredictionStore | None:
    return getattr(request.app.state, "store", None)


def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    valid_keys = settings.api_keys()
    if not valid_keys:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "API keys are not configured.")
    provided = (x_api_key or "").encode()
    if not any(secrets.compare_digest(provided, key.encode()) for key in valid_keys):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Missing or invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


# ---------- app ----------


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.LOG_LEVEL)
    app.state.store = PredictionStore(settings.DATABASE_URL) if settings.DATABASE_URL else None
    if app.state.store is None:
        logger.warning("DATABASE_URL is not set; predictions will not be stored.")
    model_holder.get()
    yield
    if app.state.store is not None:
        app.state.store.close()


app = FastAPI(title="House Price API", version="0.2.0", lifespan=lifespan)


def route_template(request: Request) -> str:
    """Use the route pattern (e.g. /predict), not the raw URL, to keep metric labels bounded."""
    for route in request.app.router.routes:
        match, _ = route.matches(request.scope)
        if match == Match.FULL:
            return getattr(route, "path", "unknown")
    return "unmatched"


@app.middleware("http")
async def observe_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    request_id = request.headers.get("X-Request-ID", "")[:64] or uuid.uuid4().hex
    token = request_id_var.set(request_id)
    route = route_template(request)
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        duration = time.perf_counter() - started
        HTTP_REQUESTS.labels(request.method, route, str(status_code)).inc()
        HTTP_LATENCY.labels(route).observe(duration)
        if route != "/metrics":
            logger.info(
                "request",
                extra={
                    "method": request.method,
                    "route": route,
                    "status_code": status_code,
                    "duration_ms": round(duration * 1000, 1),
                },
            )
        request_id_var.reset(token)


# ---------- schemas ----------


class HouseFeatures(BaseModel):
    """Limits match the training validation rules in mlops/validation.py."""

    model_config = ConfigDict(extra="forbid")

    OverallQual: int = Field(ge=1, le=10, description="Overall quality, 1-10")
    GrLivArea: float = Field(gt=0, le=10_000, description="Above-ground living area (sq ft)")
    GarageCars: int = Field(ge=0, le=10, description="Garage capacity in cars")
    TotalBsmtSF: float = Field(ge=0, le=10_000, description="Basement area (sq ft)")
    YearBuilt: int = Field(ge=1800, le=2100)


class PredictionResponse(BaseModel):
    prediction_id: uuid.UUID
    predicted_sale_price: float
    model_version: str


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: uuid.UUID
    actual_sale_price: float = Field(gt=0, le=10_000_000)


# ---------- endpoints ----------
# Docstrings of the endpoints below appear in the interactive docs at /docs.


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness: the process is running. Used by Docker's healthcheck."""
    return {"status": "ok"}


@app.get("/ready")
def ready(model: Annotated[LoadedModel, Depends(get_model)]) -> dict[str, str]:
    """Readiness: a champion model is loaded, so predictions can be served (503 otherwise)."""
    return {"status": "ready", "model_version": model.version}


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Prometheus metrics in text format. Prometheus reads this page every 15 seconds."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", dependencies=[Depends(require_api_key)])
def predict(
    features: HouseFeatures,
    model: Annotated[LoadedModel, Depends(get_model)],
    store: Annotated[PredictionStore | None, Depends(get_store)],
) -> PredictionResponse:
    """Predict the sale price of one house. Requires the X-API-Key header.

    The response includes a prediction_id: keep it to send the real price to /feedback later.
    """
    values = features.model_dump()
    frame = pd.DataFrame([values], columns=settings.FEATURES).astype(float)
    price = float(model.model.predict(frame)[0])
    prediction_id = uuid.uuid4()
    PREDICTIONS.labels(model.version).inc()
    PREDICTED_PRICE.observe(price)

    # Store the prediction for monitoring. If the database is down, still answer the client
    # but count the failure: the PredictionLogFailures alert fires on this counter.
    if store is None:
        PREDICTION_LOG_FAILURES.inc()
    else:
        try:
            store.log_prediction(prediction_id, model.version, values, price)
        except Exception:
            PREDICTION_LOG_FAILURES.inc()
            logger.exception("Could not store prediction", extra={"prediction_id": prediction_id})

    return PredictionResponse(
        prediction_id=prediction_id, predicted_sale_price=price, model_version=model.version
    )


@app.post("/feedback", status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_api_key)])
def feedback(
    body: FeedbackRequest,
    store: Annotated[PredictionStore | None, Depends(get_store)],
) -> dict[str, str]:
    """Send the real sale price for an earlier prediction. Used to measure live accuracy."""
    if store is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Feedback storage unavailable.")
    try:
        found = store.add_feedback(body.prediction_id, body.actual_sale_price)
    except Exception as exc:
        logger.exception("Could not store feedback")
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Feedback storage unavailable."
        ) from exc
    if not found:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown prediction_id.")
    return {"status": "recorded"}
