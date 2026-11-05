import numpy as np
import pandas as pd

from mlops.evaluation import behaviour_checks, promotion_gate, regression_metrics


class LinearFakeModel:
    def __init__(self, area_effect: float) -> None:
        self.area_effect = area_effect

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        prices = 50_000 + self.area_effect * frame["GrLivArea"] + 10_000 * frame["OverallQual"]
        return prices.to_numpy()


def test_metrics_for_perfect_predictions():
    metrics = regression_metrics([100.0, 200.0, 300.0], [100.0, 200.0, 300.0])
    assert metrics["rmse"] == 0
    assert metrics["r2"] == 1


def test_sensible_model_passes_behaviour_checks():
    assert behaviour_checks(LinearFakeModel(area_effect=100)) == []


def test_model_pricing_bigger_houses_lower_fails():
    failures = behaviour_checks(LinearFakeModel(area_effect=-10))
    assert any("GrLivArea" in failure for failure in failures)


def test_gate_passes_first_model_above_floor():
    decision = promotion_gate({"r2": 0.8, "rmse": 30_000}, None, [], min_r2=0.7)
    assert decision.passed


def test_gate_rejects_low_r2():
    decision = promotion_gate({"r2": 0.5, "rmse": 30_000}, None, [], min_r2=0.7)
    assert not decision.passed


def test_gate_rejects_model_worse_than_champion():
    decision = promotion_gate(
        {"r2": 0.8, "rmse": 35_000}, {"r2": 0.82, "rmse": 30_000}, [], min_r2=0.7
    )
    assert not decision.passed
    assert "champion" in decision.reasons[0]


def test_gate_includes_behaviour_failures():
    decision = promotion_gate({"r2": 0.9, "rmse": 20_000}, None, ["bad"], min_r2=0.7)
    assert decision.reasons == ["bad"]
