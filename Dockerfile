# ---------- Build stage ----------
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN pip install --no-cache-dir setuptools wheel

# Copy only dependency definition first (layer caching)
COPY pyproject.toml .
COPY src/ src/

# Build wheel
RUN pip wheel --no-deps --wheel-dir /wheels .

# Install all dependencies as wheels
RUN pip wheel --wheel-dir /wheels \
    "hyperliquid-python-sdk>=0.4" \
    "pandas>=2.1" \
    "pyarrow>=15.0" \
    "numpy>=1.26" \
    "click>=8.1" \
    "pydantic>=2.6" \
    "structlog>=24.0" \
    "PyYAML>=6.0"

# ---------- Runtime stage ----------
FROM python:3.12-slim

WORKDIR /app

# Install wheels from builder
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

# Copy scripts (for data refresh)
COPY scripts/ scripts/

# Copy config directory
COPY config/ config/

# Create directories for runtime data
RUN mkdir -p data/candles data/funding data/live_state output

# Non-root user
RUN groupadd -r app && useradd -r -g app -d /app app
RUN chown -R app:app /app
USER app

# Default entrypoint
ENTRYPOINT ["crypto-trader"]
CMD ["--help"]
