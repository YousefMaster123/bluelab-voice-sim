# syntax=docker/dockerfile:1
# BlueLab voice SIM — the dev stand-in backend + 1-page frontend (see sim/README.md).
#
# Mirrors the agent Dockerfile (same base, same venv build), with two differences:
#   * it also COPYs `sim/`, and runs `sim/server.py` instead of the worker;
#   * no `download-files` — the sim never runs a voice session, so it needs no VAD/turn-detector
#     model weights. It DOES import bluelab_voice.agent (for the real prompt assembler), which is
#     why the project itself still has to be installed.
#
# Build context is the REPO ROOT, not sim/:
#     docker build -f sim/Dockerfile -t bluelab-sim .
# On Railway, set the service's Dockerfile path to `sim/Dockerfile`.

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
# The personas + prompts are Arabic/French; force UTF-8 stdio so logging can't die on a
# UnicodeEncodeError the way a cp1252 console does.
ENV PYTHONIOENCODING=utf-8

# --- Build stage: compile native deps + install the project into a venv ---
FROM base AS build

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN python -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"
RUN pip install --no-cache-dir .

# --- Production stage ---
FROM base

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

WORKDIR /app

COPY --from=build --chown=appuser:appuser /app /app
# The sim itself (server + personas + the page). Copied after the venv so editing the harness
# doesn't invalidate the expensive dependency layer.
COPY --chown=appuser:appuser sim ./sim

ENV PATH="/app/.venv/bin:$PATH"
USER appuser

# PORT is injected by the platform; server.py reads it and binds 0.0.0.0.
EXPOSE 8000
CMD ["python", "sim/server.py"]
