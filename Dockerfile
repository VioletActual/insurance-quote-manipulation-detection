# Container for the Quote Integrity Console + API.
# Builds a lean image: runtime deps only, the package, and the model artifact.
FROM python:3.11-slim

WORKDIR /app

# Install runtime deps first so this layer caches across code changes.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# The package (includes src/qm/models/quote_risk_model.joblib and the static UI).
COPY src ./src

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Hosts set $PORT; default to 7860 (the Hugging Face Spaces convention).
EXPOSE 7860
CMD ["sh", "-c", "uvicorn src.qm.api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
