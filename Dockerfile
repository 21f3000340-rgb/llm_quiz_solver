FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for Playwright minimal build
RUN apt-get update && apt-get install -y \
    wget curl git \
    chromium \
    libffi-dev libssl-dev \
    libjpeg-dev zlib1g-dev \
    g++ gcc \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install minimal Playwright Chromium (saves ~1.2 GB)
RUN python -m playwright install --with-deps chromium

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
