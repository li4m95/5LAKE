"""Train a candidate model, compare it with the champion and register it in MLflow."""

import json
import logging
from dataclasses import dataclass
from importlib.metadata import version as package_version

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops import settings
from mlops.evaluation import GateDecision, behaviour_checks, promotion_gate, regression_metrics
from mlops.features import get_training_frame

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    run_id: str
    model_version: str
    metrics: dict[str, float]
    champion_version: str | None
    champion_metrics: dict[str, float] | None
    decision: GateDecision


def build_model(alpha: float) -> TransformedTargetRegressor:
    """Ridge regression on log(price), packaged with its preprocessing.

    Keeping imputation and scaling inside the pipeline means the exact same preprocessing is
    saved with the model and applied at prediction time: no training/serving mismatch.
    """
    regressor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    return TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)


def time_split(frame: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Older sales for training, the newest `test_fraction` of sales for testing."""
    cutoff = frame[settings.TIMESTAMP].quantile(1 - test_fraction)
    train = frame[frame[settings.TIMESTAMP] < cutoff]
    test = frame[frame[settings.TIMESTAMP] >= cutoff]
    if train.empty or test.empty:
        raise ValueError("Time-based split produced an empty train or test set.")
    return train, test


def load_champion(client: MlflowClient):
    try:
        champion = client.get_model_version_by_alias(settings.MODEL_NAME, settings.CHAMPION_ALIAS)
    except MlflowException:
        logger.info("No champion model yet; the gate will only apply the absolute rules.")
        return None
    model = mlflow.sklearn.load_model(f"models:/{settings.MODEL_NAME}/{champion.version}")
    return str(champion.version), model


def render_model_card(
    version: str,
    metrics: dict[str, float],
    champion_version: str | None,
    champion_metrics: dict[str, float] | None,
    decision: GateDecision,
    train: pd.DataFrame,
    test: pd.DataFrame,
    data_summary: dict,
) -> str:
    def period(frame: pd.DataFrame) -> str:
        dates = frame[settings.TIMESTAMP]
        return f"{dates.min():%Y-%m} to {dates.max():%Y-%m} ({len(frame)} sales)"

    rows = [f"| {name} | {value:,.3f} |" for name, value in metrics.items()]
    if champion_metrics:
        comparison = f"Champion v{champion_version} RMSE on the same holdout: "
        comparison += f"{champion_metrics['rmse']:,.0f}"
    else:
        comparison = "No champion existed when this version was trained."
    gate = "PASSED" if decision.passed else "FAILED: " + "; ".join(decision.reasons)
    lines = [
        f"# Model card: {settings.MODEL_NAME} v{version}",
        "",
        "## Intended use",
        "Estimates the sale price of a house in Ames, Iowa, from five basic attributes.",
        "Student project: not for real valuations, lending or pricing decisions.",
        "",
        "## Training data",
        "- Source: OpenML `house_prices` (Ames housing dataset)",
        f"- Raw file SHA-256: `{data_summary.get('raw_sha256', 'unknown')}`",
        f"- Training period: {period(train)}",
        f"- Holdout period: {period(test)}",
        f"- Features: {', '.join(settings.FEATURES)}",
        "",
        "## Model",
        "Ridge regression on log(price), with median imputation and standard scaling.",
        "",
        "## Holdout metrics",
        "| metric | value |",
        "|---|---|",
        *rows,
        "",
        comparison,
        "",
        "## Promotion gate",
        gate,
        "",
        "## Known limitations",
        "- Sales from 2006-2010 in one city; current prices are very different.",
        "- Five features only: no location, condition or lot size.",
        "- A linear model on log price extrapolates poorly for unusual houses.",
    ]
    return "\n".join(lines)


def train_candidate(alpha: float = settings.RIDGE_ALPHA) -> TrainingResult:
    """Train, evaluate, register and gate one candidate model. Never changes the champion."""
    # 1. Data: features from Feast, split by date.
    frame = get_training_frame()
    train, test = time_split(frame, settings.TEST_FRACTION)
    X_train, y_train = train[settings.FEATURES].astype(float), train[settings.TARGET]
    X_test, y_test = test[settings.FEATURES].astype(float), test[settings.TARGET]
    data_summary = {}
    if settings.DATA_SUMMARY_PATH.exists():
        data_summary = json.loads(settings.DATA_SUMMARY_PATH.read_text())

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.EXPERIMENT_NAME)
    client = MlflowClient()

    with mlflow.start_run() as run:
        # 2. Traceability: record exactly which data, code and settings produced this model.
        mlflow.set_tags(
            {
                "raw_data_sha256": data_summary.get("raw_sha256", "unknown"),
                "code_version": package_version("modern-mlops"),
                "split": "time-based",
            }
        )
        mlflow.log_params(
            {
                "estimator": "Ridge on log1p(price)",
                "alpha": alpha,
                "features": ",".join(settings.FEATURES),
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "train_end": f"{train[settings.TIMESTAMP].max():%Y-%m-%d}",
                "test_start": f"{test[settings.TIMESTAMP].min():%Y-%m-%d}",
            }
        )
        if settings.VALIDATION_REPORT_PATH.exists():
            mlflow.log_artifact(str(settings.VALIDATION_REPORT_PATH), artifact_path="data")

        # 3. Train on the older sales, measure on the newest ones.
        model = build_model(alpha).fit(X_train, y_train)
        metrics = regression_metrics(y_test, model.predict(X_test))
        mlflow.log_metrics(metrics)
        behaviour_failures = behaviour_checks(model)

        # 4. Store the model and register it as a new version. The signature records the
        #    expected input columns and types, so a wrong input is caught when loading.
        example = X_train.head(3)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=infer_signature(example, model.predict(example)),
            input_example=example,
        )
        registered = mlflow.register_model(f"runs:/{run.info.run_id}/model", settings.MODEL_NAME)
        version = str(registered.version)

        # 5. Compare with the current champion on the SAME holdout rows: a fair comparison.
        champion_version, champion_metrics = None, None
        champion = load_champion(client)
        if champion is not None:
            champion_version, champion_model = champion
            champion_metrics = regression_metrics(y_test, champion_model.predict(X_test))
            mlflow.log_metrics({f"champion_{k}": v for k, v in champion_metrics.items()})

        # 6. Apply the gate and write the decision onto the model version, where anyone can
        #    see it in MLflow. Passing only earns the "challenger" alias: a person promotes it.
        decision = promotion_gate(metrics, champion_metrics, behaviour_failures, settings.MIN_R2)
        tags = {
            "gate_passed": str(decision.passed).lower(),
            "gate_reasons": "; ".join(decision.reasons) or "none",
            "holdout_rmse": f"{metrics['rmse']:.0f}",
            "compared_with_champion": champion_version or "none",
        }
        for key, value in tags.items():
            client.set_model_version_tag(settings.MODEL_NAME, version, key, value)
        if decision.passed:
            client.set_registered_model_alias(
                settings.MODEL_NAME, settings.CHALLENGER_ALIAS, version
            )

        card = render_model_card(
            version,
            metrics,
            champion_version,
            champion_metrics,
            decision,
            train,
            test,
            data_summary,
        )
        mlflow.log_text(card, "model_card.md")

    return TrainingResult(
        run_id=run.info.run_id,
        model_version=version,
        metrics=metrics,
        champion_version=champion_version,
        champion_metrics=champion_metrics,
        decision=decision,
    )


if __name__ == "__main__":
    result = train_candidate()
    print(f"Registered version {result.model_version}; gate passed: {result.decision.passed}")
    for reason in result.decision.reasons:
        print(f"  - {reason}")
