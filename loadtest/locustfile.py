"""Load test.

Usage (macOS/Linux):
  API_KEY=... uv run locust -f loadtest/locustfile.py --headless -u 20 -r 5 -t 2m \
      --host http://127.0.0.1:8000
PowerShell: $env:API_KEY="..."; then the same command on one line.
"""

import os
import random

import pandas as pd
from locust import HttpUser, between, task

from mlops import settings

HOUSES = pd.read_parquet(settings.PROCESSED_PATH, columns=settings.FEATURES).to_dict("records")


class PredictionUser(HttpUser):
    wait_time = between(0.1, 0.5)

    def on_start(self) -> None:
        self.client.headers["X-API-Key"] = os.environ["API_KEY"]

    @task
    def predict(self) -> None:
        self.client.post("/predict", json=random.choice(HOUSES))
