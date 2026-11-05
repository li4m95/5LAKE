"""Metrics, behaviour checks and the promotion gate. No I/O, easy to unit test."""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mlops import settings


class Regressor(Protocol):
    """Anything with a predict() method. Lets the tests use a tiny fake model."""

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


REFERENCE_HOUSE = {
    "OverallQual": 6.0,
    "GrLivArea": 1500.0,
    "GarageCars": 2.0,
    "TotalBsmtSF": 1000.0,
    "YearBuilt": 1990.0,
}

# Increasing these features must increase the predicted price.
MONOTONIC_CHECKS = {"GrLivArea": 500.0, "OverallQual": 2.0}


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Standard regression metrics.

    mae:  average absolute error, in dollars. Easy to explain ("off by $20k on average").
    rmse: like MAE but punishes large errors more. Used by the promotion gate.
    r2:   share of the price variation the model explains (1.0 = perfect, 0 = useless).
    mape: average error as a percentage of the real price.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / y_true))),
    }


def _predict_one(model: Regressor, house: dict[str, float]) -> float:
    frame = pd.DataFrame([house], columns=settings.FEATURES)
    return float(np.asarray(model.predict(frame))[0])


def behaviour_checks(model: Regressor) -> list[str]:
    """Sanity checks a good metric cannot replace. Returns a list of failures (empty = pass).

    We predict one ordinary "reference" house, then the same house made bigger or better, and
    require the price to go up.
    """
    base = _predict_one(model, REFERENCE_HOUSE)
    if not np.isfinite(base) or base <= 0:
        return [f"Reference house predicted at {base}; expected a positive price."]
    failures = []
    for feature, step in MONOTONIC_CHECKS.items():
        bigger = {**REFERENCE_HOUSE, feature: REFERENCE_HOUSE[feature] + step}
        if _predict_one(model, bigger) <= base:
            failures.append(f"Price did not increase when {feature} increased by {step:g}.")
    return failures


@dataclass
class GateDecision:
    passed: bool
    reasons: list[str] = field(default_factory=list)


def promotion_gate(
    candidate: dict[str, float],
    champion: dict[str, float] | None,
    behaviour_failures: list[str],
    min_r2: float,
) -> GateDecision:
    """Decide whether a candidate model may become the challenger.

    Every rule that fails adds a human-readable reason, so a rejected model always says why.
    `champion` is None for the very first model; then only the absolute rules apply.
    """
    reasons = list(behaviour_failures)
    if candidate["r2"] < min_r2:
        reasons.append(f"R2 {candidate['r2']:.3f} is below the minimum {min_r2:.2f}.")
    if champion is not None and candidate["rmse"] > champion["rmse"]:
        reasons.append(
            f"RMSE {candidate['rmse']:,.0f} is worse than the champion's {champion['rmse']:,.0f}."
        )
    return GateDecision(passed=not reasons, reasons=reasons)
