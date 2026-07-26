# Heavy data image: micom + cobra + HiGHS + memote on top of the base package.
# Used by GENERATE_DATA, ACTIVE_ROUND (real solver oracle) and the M0/M1 QC stage
# QC_MODELS + DEGENERACY_SURVEY (cobra FVA; QC_MODELS also needs the `memote` CLI,
# pulled in by the `data` extra). Rebuild + bump the tag when the extra changes.
#   docker build -f docker/data.Dockerfile -t ghcr.io/timrozday-mgnify/surrogate-mgem-data:<ver> .
FROM python:3.11-slim

# cobra/micom pull in scientific wheels that need a compiler toolchain at build.
# libexpat1 is a runtime dep of libsbml (cobra's SBML reader) not in python:slim.
# procps supplies `ps`, which nextflow requires to collect task metrics.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential glpk-utils libglpk-dev libexpat1 procps \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1

WORKDIR /opt/surrogate-mgem
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[data]"

WORKDIR /work
