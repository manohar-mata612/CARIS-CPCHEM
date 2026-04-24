# ── Stage 1: builder ─────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# copy application code
COPY data/       ./data/
COPY ml/         ./ml/
COPY rag/        ./rag/
COPY agents/     ./agents/
COPY api/        ./api/
COPY simulator/  ./simulator/

# make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# GCP injects PORT env variable — Cloud Run requires this
ENV PORT=8080

# expose port
EXPOSE 8080

# health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/health')"

# start FastAPI on Cloud Run
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]