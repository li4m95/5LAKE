"""Data-quality rules that must pass before any model is trained."""

from dataclasses import asdict, dataclass, field

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from mlops import settings

# Allowed (minimum, maximum) for each numeric column. Keep in sync with the API limits.
RANGES: dict[str, tuple[float, float]] = {
    "OverallQual": (1, 10),
    "GrLivArea": (1, 10_000),
    "GarageCars": (0, 10),
    "TotalBsmtSF": (0, 10_000),
    "YearBuilt": (1800, 2100),
    "SalePrice": (1, 10_000_000),
    "YrSold": (1900, 2100),
    "MoSold": (1, 12),
}


class ValidationError(Exception):
    """Raised when the dataset breaks one or more data-quality rules."""


@dataclass
class ValidationReport:
    row_count: int
    null_rates: dict[str, float] = field(default_factory=dict)
    out_of_range: dict[str, int] = field(default_factory=dict)
    duplicate_ids: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict:
        return {**asdict(self), "passed": self.passed}


def validate(
    df: DataFrame,
    min_rows: int = settings.MIN_ROWS,
    max_null_rate: float = settings.MAX_NULL_RATE,
) -> ValidationReport:
    """Check the raw data against the rules above and return a report (never raises).

    The caller decides what to do with a failed report; in the pipeline it stops the run.
    """
    id_column = settings.ID_COLUMN
    required = [id_column, *RANGES]

    # 1. Structure: are all the columns we rely on present?
    missing = [column for column in required if column not in df.columns]
    if missing:
        return ValidationReport(row_count=0, failures=[f"Missing required columns: {missing}"])

    # 2. Volume: a file that is suddenly much smaller usually means a broken export.
    row_count = df.count()
    if row_count < min_rows:
        return ValidationReport(
            row_count=row_count,
            failures=[f"Only {row_count} rows; at least {min_rows} are required."],
        )

    # 3. Content: count missing and out-of-range values for every column in ONE pass over the
    #    data. Building all the expressions first and running a single agg() is much faster
    #    than running one Spark job per column.
    null_exprs = [
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(f"null:{c}") for c in required
    ]
    range_exprs = [
        F.sum(F.when((F.col(c) < low) | (F.col(c) > high), 1).otherwise(0)).alias(f"range:{c}")
        for c, (low, high) in RANGES.items()
    ]
    counts = df.agg(*null_exprs, *range_exprs).first().asDict()

    null_rates = {c: counts[f"null:{c}"] / row_count for c in required}
    out_of_range = {c: int(counts[f"range:{c}"]) for c in RANGES}
    duplicate_ids = row_count - df.select(id_column).distinct().count()

    # 4. Turn the numbers into readable failure messages. An empty list means "passed".
    failures: list[str] = []
    if null_rates[id_column] > 0:  # every row needs an ID: it becomes the entity key
        failures.append(f"{null_rates[id_column]:.1%} of rows have no {id_column}.")
    for column, rate in null_rates.items():
        if column != id_column and rate > max_null_rate:
            failures.append(f"{column}: {rate:.1%} missing values (limit {max_null_rate:.0%}).")
    for column, count in out_of_range.items():
        if count:
            failures.append(f"{column}: {count} values outside {RANGES[column]}.")
    if duplicate_ids:
        failures.append(f"{duplicate_ids} duplicate {id_column} values.")

    return ValidationReport(
        row_count=row_count,
        null_rates=null_rates,
        out_of_range=out_of_range,
        duplicate_ids=duplicate_ids,
        failures=failures,
    )
