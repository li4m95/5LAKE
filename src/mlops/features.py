"""Feast feature definitions, feature-source build and training-data retrieval."""

from datetime import timedelta

import pandas as pd
from feast import Entity, FeatureStore, FeatureView, Field, FileSource
from feast.types import Float64

from mlops import settings

# An entity is "the thing features describe". Here: one house sale, identified by house_id.
house = Entity(
    name="house",
    join_keys=[settings.ENTITY_KEY],
    description="One house sale in the Ames housing dataset.",
)

# Where the feature values live, and which column says WHEN each value was true.
housing_source = FileSource(
    name="housing_source",
    path=settings.FEATURES_PATH.as_posix(),
    timestamp_field=settings.TIMESTAMP,
)

# A feature view groups features of one entity from one source. ttl = how old a feature value
# may be (relative to the event time) and still be used.
housing_features = FeatureView(
    name="housing_features",
    entities=[house],
    ttl=timedelta(days=30),
    schema=[Field(name=name, dtype=Float64) for name in settings.FEATURES],
    source=housing_source,
    online=False,  # the API receives features in the request; no online store is needed
)


def get_store() -> FeatureStore:
    return FeatureStore(repo_path=str(settings.FEATURE_REPO))


def build_feature_source() -> int:
    """Write the feature table (no labels) as a single Parquet file for Feast."""
    columns = [settings.ENTITY_KEY, settings.TIMESTAMP, *settings.FEATURES]
    frame = pd.read_parquet(settings.PROCESSED_PATH, columns=columns)
    settings.FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(settings.FEATURES_PATH, index=False)
    return len(frame)


def apply_definitions() -> None:
    get_store().apply([house, housing_features])


def get_training_frame() -> pd.DataFrame:
    """Labels come from the processed data; features come from Feast (point-in-time join)."""
    labels = pd.read_parquet(
        settings.PROCESSED_PATH, columns=[settings.ENTITY_KEY, settings.TIMESTAMP, settings.TARGET]
    )
    feature_refs = [f"{housing_features.name}:{name}" for name in settings.FEATURES]
    # For each (house_id, event_timestamp) in `labels`, Feast looks up the latest feature values
    # recorded at or before that time: a "point-in-time join". It never uses future values.
    frame = get_store().get_historical_features(entity_df=labels, features=feature_refs).to_df()

    missing = [c for c in [*settings.FEATURES, settings.TARGET] if c not in frame.columns]
    if missing:
        raise ValueError(f"Feast output is missing columns: {missing}")
    if frame[settings.FEATURES].isna().any().any():
        raise ValueError("Feast returned rows without features; check the TTL and timestamps.")
    return frame.sort_values(settings.TIMESTAMP).reset_index(drop=True)
