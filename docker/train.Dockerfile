# Light training image: torch + sklearn only (no solver stack).
# Used by TRAIN_SURROGATE and COLLECT_METRICS.
#   docker build -f docker/train.Dockerfile -t ghcr.io/timrozday-mgnify/surrogate-mgem-train:<ver> .
FROM python:3.11-slim

# CPU-only torch keeps the image small; the surrogate MLP is tiny.
ENV PIP_NO_CACHE_DIR=1 PIP_INDEX_URL=https://download.pytorch.org/whl/cpu \
    PIP_EXTRA_INDEX_URL=https://pypi.org/simple

WORKDIR /opt/surrogate-mgem
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

WORKDIR /work
