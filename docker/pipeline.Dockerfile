# syntax=docker/dockerfile:1

# NOTE: the course text pins 0.8.0. We use the same uv version that generated uv.lock
# (lock revision 5), so --frozen can read it without surprises.
ARG UV_VERSION=0.13.7
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:3.11-slim-bookworm

# Java is required by Spark; procps provides `ps`, which Spark's start scripts use.
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencies first (cached until the lockfile changes), then our code.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --group pipeline --no-install-project
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --group pipeline --no-editable

COPY feature_repo/feature_store.yaml ./feature_repo/feature_store.yaml
COPY docker/workspace.yaml ./workspace.yaml
COPY docker/dagster.yaml /opt/dagster/home/dagster.yaml

# A normal user that owns only the folders it must write to.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data /app/artifacts /app/feature_repo/data \
        /opt/dagster/home/storage /mlflow/artifacts \
    && chown -R app:app /app/data /app/artifacts /app/feature_repo /opt/dagster /mlflow

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PROJECT_ROOT=/app \
    DAGSTER_HOME=/opt/dagster/home
USER app
