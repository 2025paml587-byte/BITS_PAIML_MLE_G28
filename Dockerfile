# Serves the pre-trained trip-duration models (models/*.joblib) via
# the FastAPI app in src/api - no DVC, MLflow, or training data needed
# at runtime, since the models are already trained and baked in below.

# ---- Build stage: install dependencies into an isolated venv ----
FROM python:3.12-slim AS builder

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# ---- Runtime stage: slim image, non-root user, app code + models only ----
FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
