# syntax=docker/dockerfile:1
# LiveKit Agents worker — bluelab-voice.
# Rewritten from the generic requirements.txt template for THIS repo, which is a
# hatchling / pyproject.toml project (no requirements.txt): install via `pip install .`,
# run the worker module (NOT agent.py — it has no entrypoint), and pin Python 3.12 to
# match dev and maximize native-wheel availability (av, onnxruntime, speechmatics-rt).
# Docs: https://docs.livekit.io/agents/ops/deployment/builds/

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS base

# Unbuffered stdio so logs aren't lost on crash; skip pip's version self-check.
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# --- Build stage: compile native deps + install the project into a venv ---
FROM base AS build

# Toolchain for any package without a manylinux wheel (kept out of the final image).
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# hatchling builds a wheel from pyproject + the two src packages
# (bluelab_voice + the vendored bluelab_runtime_bundle). Copy only what the build
# needs so this layer caches until pyproject/src actually change.
COPY pyproject.toml ./
COPY src ./src

RUN python -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"
RUN pip install --no-cache-dir .

# Pre-download plugin model files (multilingual turn detector + Silero VAD) so the
# first call doesn't stall on a cold download. The generic command discovers the
# installed livekit-plugins-* packages WITHOUT importing our agent, so it needs no
# env vars / provider keys at build time.
RUN python -m livekit.agents download-files

# --- Production stage: slim runtime, no build toolchain, non-root ---
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

# Bring over the installed venv (with the package + downloaded models) in one layer.
COPY --from=build --chown=appuser:appuser /app /app

ENV PATH="/app/.venv/bin:$PATH"
USER appuser

# The worker registers with LiveKit Cloud and waits for dispatched jobs (07 §6).
# `start` = production mode. Entry point is bluelab_voice.worker:main (cli.run_app).
CMD ["python", "-m", "bluelab_voice.worker", "start"]
