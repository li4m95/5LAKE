from pyspark.sql.types import DoubleType, StructField, StructType

from mlops.validation import RANGES, validate

COLUMNS = ["Id", *RANGES]
SCHEMA = StructType([StructField(name, DoubleType(), nullable=True) for name in COLUMNS])
VALID = {
    "OverallQual": 7.0,
    "GrLivArea": 1800.0,
    "GarageCars": 2.0,
    "TotalBsmtSF": 900.0,
    "YearBuilt": 2005.0,
    "SalePrice": 200_000.0,
    "YrSold": 2008.0,
    "MoSold": 6.0,
}


def make_rows(count: int = 20, **overrides) -> list[dict]:
    return [{**VALID, "Id": float(i), **overrides} for i in range(count)]


def test_valid_data_passes(spark):
    report = validate(spark.createDataFrame(make_rows(), schema=SCHEMA), min_rows=10)
    assert report.passed, report.failures


def test_missing_column_fails(spark):
    df = spark.createDataFrame(make_rows(), schema=SCHEMA).drop("GrLivArea")
    report = validate(df, min_rows=10)
    assert not report.passed
    assert "GrLivArea" in report.failures[0]


def test_too_few_rows_fails(spark):
    report = validate(spark.createDataFrame(make_rows(5), schema=SCHEMA), min_rows=10)
    assert not report.passed


def test_out_of_range_value_fails(spark):
    report = validate(
        spark.createDataFrame(make_rows(OverallQual=42.0), schema=SCHEMA), min_rows=10
    )
    assert any("OverallQual" in failure for failure in report.failures)


def test_too_many_missing_values_fails(spark):
    rows = make_rows()
    for row in rows[:5]:  # 25% missing, limit is 5%
        row["GrLivArea"] = None
    report = validate(spark.createDataFrame(rows, schema=SCHEMA), min_rows=10)
    assert any("GrLivArea" in failure for failure in report.failures)


def test_duplicate_ids_fail(spark):
    report = validate(spark.createDataFrame(make_rows(Id=1.0), schema=SCHEMA), min_rows=10)
    assert report.duplicate_ids == 19
    assert not report.passed
