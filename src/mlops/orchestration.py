"""Dagster assets, jobs and schedules for training and monitoring."""

from dagster import (
    AssetExecutionContext,
    DefaultScheduleStatus,
    Definitions,
    Failure,
    MaterializeResult,
    MetadataValue,
    ScheduleDefinition,
    asset,
    define_asset_job,
)
from mlflow.exceptions import MlflowException

from mlops import settings
from mlops.data import download_raw, file_sha256, prepare_data
from mlops.db import PredictionStore
from mlops.features import apply_definitions, build_feature_source
from mlops.monitoring import CheckResult, champion_info, check_drift, check_quality
from mlops.train import train_candidate

# ---------- training ----------
# Each @asset is one step that produces something (a file, a model version). `deps=[...]` says
# which asset must be built first, so Dagster always runs them in the right order. A step that
# raises an exception stops the run, and later steps do not run.


@asset(group_name="training", description="Raw Ames housing CSV from OpenML.")
def raw_housing_data() -> MaterializeResult:
    path = download_raw()
    return MaterializeResult(metadata={"path": str(path), "sha256": file_sha256(path)})


@asset(
    group_name="training",
    deps=[raw_housing_data],
    description="Validated and cleaned Parquet data (PySpark). Fails on bad data.",
)
def processed_housing_data() -> MaterializeResult:
    summary = prepare_data()
    return MaterializeResult(metadata=summary)


@asset(group_name="training", deps=[processed_housing_data], description="Feast feature view.")
def feast_feature_view() -> MaterializeResult:
    rows = build_feature_source()
    apply_definitions()
    return MaterializeResult(metadata={"rows": rows})


@asset(
    group_name="training",
    deps=[feast_feature_view],
    description="New registered model version, checked by the promotion gate.",
)
def candidate_model(context: AssetExecutionContext) -> MaterializeResult:
    result = train_candidate()
    if result.decision.passed:
        context.log.info(f"v{result.model_version} passed the gate and is the new challenger.")
    else:
        context.log.warning(
            f"v{result.model_version} failed the gate: {'; '.join(result.decision.reasons)}"
        )
    return MaterializeResult(
        metadata={
            "model_version": result.model_version,
            "gate_passed": result.decision.passed,
            "gate_reasons": MetadataValue.text("; ".join(result.decision.reasons) or "none"),
            "rmse": result.metrics["rmse"],
            "r2": result.metrics["r2"],
            "mlflow_run_id": result.run_id,
        }
    )


# ---------- monitoring ----------


def _record(store: PredictionStore, check: CheckResult, model_version: str) -> MaterializeResult:
    """Save the result to Postgres (for Grafana), then fail the run if the check failed."""
    store.record_check(
        check.name, model_version, check.value, check.threshold, check.passed, check.details
    )
    metadata = {
        "passed": check.passed,
        "value": check.value if check.value is not None else "skipped",
        "threshold": check.threshold,
        "details": MetadataValue.json(check.details),
    }
    if not check.passed:
        raise Failure(
            description=f"{check.name} failed: {check.value:.3f} > {check.threshold:.3f}",
            metadata=metadata,
        )
    return MaterializeResult(metadata=metadata)


def _run_check(check_name: str) -> MaterializeResult:
    try:
        model_version, holdout_rmse = champion_info()
    except MlflowException:
        return MaterializeResult(metadata={"skipped": "no champion model yet"})
    store = PredictionStore(settings.require_database_url())
    try:
        if check_name == "drift":
            check = check_drift(store)
        else:
            check = check_quality(store, model_version, holdout_rmse)
        return _record(store, check, model_version)
    finally:
        store.close()


@asset(group_name="monitoring", description="Evidently drift check on recent requests.")
def input_drift_check() -> MaterializeResult:
    return _run_check("drift")


@asset(group_name="monitoring", description="Live RMSE of the champion, from feedback.")
def live_quality_check() -> MaterializeResult:
    return _run_check("quality")


# ---------- jobs and schedules ----------

training_job = define_asset_job(
    "training_job",
    selection=[raw_housing_data, processed_housing_data, feast_feature_view, candidate_model],
)
monitoring_job = define_asset_job(
    "monitoring_job", selection=[input_drift_check, live_quality_check]
)

defs = Definitions(
    assets=[
        raw_housing_data,
        processed_housing_data,
        feast_feature_view,
        candidate_model,
        input_drift_check,
        live_quality_check,
    ],
    jobs=[training_job, monitoring_job],
    schedules=[
        ScheduleDefinition(
            job=training_job,
            cron_schedule="0 6 * * 1",
            default_status=DefaultScheduleStatus.RUNNING,
        ),
        ScheduleDefinition(
            job=monitoring_job,
            cron_schedule="0 7 * * *",
            default_status=DefaultScheduleStatus.RUNNING,
        ),
    ],
)
