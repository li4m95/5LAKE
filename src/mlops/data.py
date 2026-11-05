"""Download the raw dataset and prepare it with PySpark."""

import hashlib
import json
import os
import sys
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from sklearn.datasets import fetch_openml

from mlops import settings
from mlops.validation import ValidationError, validate

REQUIRED_COLUMNS = [settings.ID_COLUMN, "YrSold", "MoSold", *settings.FEATURES, settings.TARGET]


def file_sha256(path: Path) -> str:
    """Fingerprint of a file: any change to the data, even one character, changes the hash."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_raw(force: bool = False) -> Path:
    """Download the Ames housing data once. The file is then reused, so its hash stays stable."""
    if settings.RAW_PATH.exists() and not force:
        return settings.RAW_PATH
    dataset = fetch_openml(name="house_prices", version=1, as_frame=True, parser="auto")
    frame = dataset.frame
    if frame is None or frame.empty:
        raise RuntimeError("OpenML returned an empty dataset.")
    if settings.ID_COLUMN not in frame.columns:
        # Some OpenML versions drop the row-id column. Rows keep the source order, and the raw
        # file is versioned by its hash, so this key is stable for a given raw file.
        frame.insert(0, settings.ID_COLUMN, range(1, len(frame) + 1))
    settings.RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = settings.RAW_PATH.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(settings.RAW_PATH)  # atomic: never leaves a half-written file
    return settings.RAW_PATH


def create_spark(app_name: str = "modern-mlops") -> SparkSession:
    # Spark workers must run the same Python as this process (with the same packages installed).
    # Without this, Spark starts whatever "python" is first on PATH, which fails on many machines.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    return (
        SparkSession.builder.appName(app_name)
        .master(os.getenv("SPARK_MASTER", "local[2]"))
        .config("spark.driver.memory", settings.SPARK_DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        # Local mode only needs loopback. Without this, Spark can resolve the machine name to
        # another address (e.g. Docker Desktop's host.docker.internal) and workers cannot connect.
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )


def load_raw(spark: SparkSession) -> DataFrame:
    # Read everything as text, then cast explicitly: values like "NA" become null.
    df = spark.read.option("header", True).csv(str(settings.RAW_PATH))
    for column in REQUIRED_COLUMNS:
        if column in df.columns:
            df = df.withColumn(column, F.col(column).cast(DoubleType()))
    return df


def prepare_data() -> dict:
    """Validate the raw CSV, clean it and write it as Parquet. Returns a small summary.

    Raises ValidationError when the data breaks a rule, so no model is trained on bad data.
    """
    if not settings.RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data not found at {settings.RAW_PATH}.")

    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    spark = create_spark()
    try:
        raw = load_raw(spark)
        # Validate the data as it arrived, BEFORE any cleaning, and always save the report
        # (also when it fails: that is exactly when someone needs to read it).
        report = validate(raw)
        settings.VALIDATION_REPORT_PATH.write_text(json.dumps(report.to_dict(), indent=2))
        if not report.passed:
            raise ValidationError("; ".join(report.failures))

        # The sale date (year + month) becomes the event time. Day 1 is used because the
        # dataset only records the month of the sale.
        sale_date = F.format_string(
            "%04d-%02d-01", F.col("YrSold").cast("int"), F.col("MoSold").cast("int")
        )
        clean = (
            # Validation guaranteed that at most 5% of values are missing, so dropping those
            # rows is safe. Without that check, this line could silently remove most of the data.
            raw.dropna(subset=REQUIRED_COLUMNS)
            .withColumn(settings.ENTITY_KEY, F.col(settings.ID_COLUMN).cast("long"))
            .withColumn(settings.TIMESTAMP, F.to_timestamp(sale_date, "yyyy-MM-dd"))
            .select(settings.ENTITY_KEY, settings.TIMESTAMP, *settings.FEATURES, settings.TARGET)
        )
        settings.PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        clean.write.mode("overwrite").parquet(str(settings.PROCESSED_PATH))
        # Count what was actually written, not what we think we wrote.
        clean_rows = spark.read.parquet(str(settings.PROCESSED_PATH)).count()
    finally:
        spark.stop()

    summary = {
        "raw_sha256": file_sha256(settings.RAW_PATH),
        "raw_rows": report.row_count,
        "clean_rows": clean_rows,
        "dropped_rows": report.row_count - clean_rows,
    }
    settings.DATA_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    download_raw()
    print(json.dumps(prepare_data(), indent=2))
