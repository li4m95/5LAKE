# syntax=docker/dockerfile:1

# NOTE: the course text pins 0.8.0. We use the same uv version that generated uv.lock
# (lock revision 5), so --frozen can read it without surprises.
ARG UV_VERSION=0.13.7
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# Stage 1 (builder): install dependencies into /app/.venv. Build tools stay in this stage.
FROM python:3.15-slim-bookworm AS builder
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencies first: this layer is cached until the lockfile changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

# Stage 2 (runtime): a clean image with only the virtual environment copied in.
FROM python:3.15-slim-bookworm

# Run as a normal user, not root: limits the damage if the app is compromised.
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PROJECT_ROOT=/app
USER app
EXPOSE 8000

# Docker marks the container unhealthy if /health stops answering.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]
CMD ["uvicorn", "mlops.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
