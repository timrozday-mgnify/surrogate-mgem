# Light training image: torch + sklearn + the M3 jax stack (no solver stack).
# Used by TRAIN_SURROGATE, COLLECT_METRICS, TRAIN_VALUE, BASELINE_RF,
# COLLECT_VALUE_METRICS and COLLECT_D4.
#   docker build -f docker/train.Dockerfile -t ghcr.io/timrozday-mgnify/surrogate-mgem-train:<ver> .
FROM python:3.11-slim

# procps supplies `ps`, which nextflow requires to collect task metrics.
RUN apt-get update && apt-get install -y --no-install-recommends procps \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch keeps the image small; the surrogate MLP is tiny.
ENV PIP_NO_CACHE_DIR=1 PIP_INDEX_URL=https://download.pytorch.org/whl/cpu \
    PIP_EXTRA_INDEX_URL=https://pypi.org/simple

WORKDIR /opt/surrogate-mgem
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# `[jax]` is M3 Head A (`cfs train-value`): jax/equinox/optax, plus the pyarrow
# `baseline-rf` also needs to read the label shards. None of it is on the torch CPU
# index above, so it resolves through PIP_EXTRA_INDEX_URL.
RUN pip install ".[jax]"

WORKDIR /work
