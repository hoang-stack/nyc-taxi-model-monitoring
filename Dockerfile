FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    TAXIMON_MODEL=/app/models/current TAXIMON_LOGS=/app/logs PORT=8000
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd --create-home app && mkdir -p logs && chown app logs
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD curl -fs http://localhost:8000/health || exit 1
# The model is not baked into the image: mount a registry (models/<version> + current link).
CMD ["taximon", "serve"]
