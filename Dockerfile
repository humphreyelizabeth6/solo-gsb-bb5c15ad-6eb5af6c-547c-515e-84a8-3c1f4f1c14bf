FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code.
COPY app ./app

# Run as an unprivileged user that owns the persistent data directory.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /data /app
USER appuser

VOLUME ["/data"]

ENV DB_PATH=/data/scheduler.db \
    TOTAL_GPUS=8 \
    BASE_QUANTUM=60 \
    AGING_RATE=1 \
    AGING_CAP_FACTOR=10

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import json,urllib.request,sys; \
r=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health')); \
sys.exit(0 if r['status']=='ok' else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
