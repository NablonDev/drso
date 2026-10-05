# Beeline route planner — one image for the API and the GPU worker.
# CUDA comes from the PyTorch wheel; the host driver is injected by the NVIDIA container toolkit.

# ---- web build ----------------------------------------------------------------------
FROM node:22-bookworm-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# ---- runtime ------------------------------------------------------------------------
FROM mambaorg/micromamba:1.5.10-jammy
USER root
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git && rm -rf /var/lib/apt/lists/*
COPY deploy/environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && micromamba clean -a -y
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
ARG BEELINE_COMMIT=unknown
ENV PATH=/opt/conda/bin:$PATH \
    BEELINE_COMMIT=${BEELINE_COMMIT} \
    PROJ_DATA=/opt/conda/share/proj \
    GDAL_DATA=/opt/conda/share/gdal \
    PYTHONUNBUFFERED=1 \
    BEELINE_DATA=/data/bucket \
    BEELINE_CONFIG_DIR=/app/config
WORKDIR /app
COPY beeline_engine/ beeline_engine/
COPY beeline_api/ beeline_api/
COPY lulc_models/ lulc_models/
COPY config/ config/
COPY scripts/ scripts/
COPY tests/ tests/
COPY --from=web /web/dist web/dist
RUN python -c "import beeline_api.app, beeline_engine.pipeline, torch; print('image ok', torch.__version__)"
EXPOSE 8000
CMD ["uvicorn", "beeline_api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
