FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-vie \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY gateway/app /app/gateway/app

ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["uvicorn", "gateway.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
