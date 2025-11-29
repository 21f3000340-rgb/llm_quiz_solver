# Use slim Python
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system deps (keep small)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git unzip \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
    libxkbcommon0 libgtk-3-0 libgbm1 libasound2 \
    libxcomposite1 libxdamage1 libxrandr2 libxfixes3 \
    libpango-1.0-0 libcairo2 \
    tesseract-ocr \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for caching (create requirements.txt if you don't have one)
COPY requirements.txt ./

# Install python deps
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# Install Playwright and browsers
RUN pip install playwright \
    && playwright install --with-deps chromium

# Copy the rest of the app
COPY . .

# Railway exposes PORT env var. Use 0.0.0.0 host.
EXPOSE 8080

# Use uvicorn to run FastAPI app object in app.py (app:app)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
