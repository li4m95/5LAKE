"""Project configuration. Every value can be overridden with an environment variable."""

import os
from pathlib import Path

# In containers the package is installed into site-packages, so the project root is passed in.
ROOT = Path(os.getenv("PROJECT_ROOT") or Path(__file__).resolve().parents[2])

DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "housing.csv"
PROCESSED_PATH = DATA_DIR / "processed" / "housing.parquet"
FEATURES_PATH = DATA_DIR / "features" / "housing_features.parquet"
REPORTS_DIR = ROOT / "artifacts" / "reports"
VALIDATION_REPORT_PATH = REPORTS_DIR / "validation.json"
DATA_SUMMARY_PATH = REPORTS_DIR / "data_summary.json"
FEATURE_REPO = ROOT / "feature_repo"

# Data contract
ID_COLUMN = "Id"
ENTITY_KEY = "house_id"
TIMESTAMP = "event_timestamp"
FEATURES = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "YearBuilt"]
TARGET = "SalePrice"

# Data validation
MIN_ROWS = int(os.getenv("MIN_ROWS", "1000"))
MAX_NULL_RATE = float(os.getenv("MAX_NULL_RATE", "0.05"))
SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "2g")

# Training and promotion gate
RIDGE_ALPHA = float(os.getenv("RIDGE_ALPHA", "1.0"))
TEST_FRACTION = 0.2
MIN_R2 = float(os.getenv("MIN_R2", "0.70"))

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "house-price-regression")
MODEL_NAME = os.getenv("MODEL_NAME", "house-price-model")
CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"

# Serving
DATABASE_URL = os.getenv("DATABASE_URL", "")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Monitoring
DRIFT_WINDOW_DAYS = 7
DRIFT_MIN_ROWS = 50
DRIFT_SHARE_THRESHOLD = 0.5  # fail if more than half of the features drifted
QUALITY_WINDOW_DAYS = 30
QUALITY_MIN_ROWS = 30
QUALITY_RMSE_RATIO = 1.5  # fail if live RMSE is 50% worse than the holdout RMSE


def api_keys() -> set[str]:
    """Read on each call, so tests can set keys and a restart picks up rotated keys."""
    return {key.strip() for key in os.getenv("API_KEYS", "").split(",") if key.strip()}


def require_database_url() -> str:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set.")
    return DATABASE_URL
