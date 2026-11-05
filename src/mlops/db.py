"""Postgres storage for predictions, feedback and monitoring results."""

import threading
import uuid
from datetime import timedelta
from typing import Any

import pandas as pd
from psycopg import errors
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

# Three tables:
#   predictions        one row per /predict call (the inputs, the price, the model version)
#   feedback           the real sale price, sent later, linked to a prediction by its ID
#   monitoring_results one row per monitoring check run (shown in Grafana)
# "IF NOT EXISTS" makes the schema safe to apply again on every start.
SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version TEXT NOT NULL,
    features JSONB NOT NULL,
    predicted_price DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS predictions_created_at_idx ON predictions (created_at);

CREATE TABLE IF NOT EXISTS feedback (
    prediction_id UUID PRIMARY KEY REFERENCES predictions (prediction_id),
    actual_price DOUBLE PRECISION NOT NULL CHECK (actual_price > 0),
    received_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS monitoring_results (
    id BIGSERIAL PRIMARY KEY,
    checked_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    check_name TEXT NOT NULL,
    model_version TEXT,
    value DOUBLE PRECISION,
    threshold DOUBLE PRECISION,
    passed BOOLEAN NOT NULL,
    details JSONB
);
"""


class PredictionStore:
    """All database access in one place, through a small pool of reusable connections."""

    def __init__(self, url: str, max_size: int = 5, timeout: float = 2.0) -> None:
        # timeout: how long a request waits for a connection before failing fast.
        self._pool = ConnectionPool(
            url,
            min_size=1,
            max_size=max_size,
            timeout=timeout,
            open=True,
            kwargs={"autocommit": True},
        )
        self._schema_ready = False
        self._schema_lock = threading.Lock()

    def close(self) -> None:
        self._pool.close()

    def _ensure_schema(self) -> None:
        # Created lazily, so the API recovers if Postgres was down when it started.
        if self._schema_ready:
            return
        with self._schema_lock:
            if not self._schema_ready:
                with self._pool.connection() as conn:
                    conn.execute(SCHEMA)
                self._schema_ready = True

    def _query(self, sql: str, params: tuple) -> pd.DataFrame:
        self._ensure_schema()
        with self._pool.connection() as conn:
            cursor = conn.execute(sql, params)
            columns = [column.name for column in cursor.description]
            return pd.DataFrame(cursor.fetchall(), columns=columns)

    def log_prediction(
        self,
        prediction_id: uuid.UUID,
        model_version: str,
        features: dict[str, Any],
        predicted_price: float,
    ) -> None:
        self._ensure_schema()
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO predictions (prediction_id, model_version, features, predicted_price)"
                " VALUES (%s, %s, %s, %s)",
                (prediction_id, model_version, Jsonb(features), predicted_price),
            )

    def add_feedback(self, prediction_id: uuid.UUID, actual_price: float) -> bool:
        """Store the real sale price. Returns False if the prediction ID is unknown."""
        self._ensure_schema()
        try:
            with self._pool.connection() as conn:
                conn.execute(
                    "INSERT INTO feedback (prediction_id, actual_price) VALUES (%s, %s)"
                    " ON CONFLICT (prediction_id) DO UPDATE"
                    " SET actual_price = EXCLUDED.actual_price, received_at = now()",
                    (prediction_id, actual_price),
                )
        except errors.ForeignKeyViolation:
            return False
        return True

    def recent_predictions(self, days: int) -> pd.DataFrame:
        return self._query(
            "SELECT prediction_id, created_at, model_version, features, predicted_price"
            " FROM predictions WHERE created_at >= now() - %s",
            (timedelta(days=days),),
        )

    def labelled_predictions(self, days: int, model_version: str) -> pd.DataFrame:
        return self._query(
            "SELECT p.prediction_id, p.created_at, p.predicted_price, f.actual_price"
            " FROM predictions p JOIN feedback f USING (prediction_id)"
            " WHERE p.model_version = %s AND p.created_at >= now() - %s",
            (model_version, timedelta(days=days)),
        )

    def record_check(
        self,
        check_name: str,
        model_version: str | None,
        value: float | None,
        threshold: float,
        passed: bool,
        details: dict[str, Any],
    ) -> None:
        self._ensure_schema()
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO monitoring_results"
                " (check_name, model_version, value, threshold, passed, details)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                (check_name, model_version, value, threshold, passed, Jsonb(details)),
            )
