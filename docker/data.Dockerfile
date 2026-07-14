# Heavy data image: micom + cobra + HiGHS on top of the base package.
# Used by GENERATE_DATA and ACTIVE_ROUND (both need the real solver oracle).
#   docker build -f docker/data.Dockerfile -t ghcr.io/timrozday-mgnify/surrogate-mgem-data:<ver> .
FROM python:3.11-slim

# cobra/micom pull in scientific wheels that need a compiler toolchain at build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential glpk-utils libglpk-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1

WORKDIR /opt/surrogate-mgem
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[data]"

WORKDIR /work
