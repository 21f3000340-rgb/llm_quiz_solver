# ==============================
# 1) Base Image (Python 3.12 OK)
# ==============================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ==============================
# 2) System Dependencies
# ==============================
RUN apt-get update && apt-get install -y \
    wget curl git unzip \
    # Playwright deps
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
    libxkbcommon0 libgtk-3-0 libgbm1 libasound2 \
    libxcomposite1 libxdamage1 libxrandr2 libxfixes3 \
    libpango-1.0-0 libcairo2 \
    # OCR Engine
    tesseract-ocr \
    # Build tools
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ==============================
# 3) Install Playwright
# ==============================
RUN pip install playwright
RUN playwright install --with-deps chromium

# ==============================
# 4) Install uv (Your project uses uv.lock)
# ==============================
RUN pip install uv

# ==============================
# 5) Copy Project
# ==============================
WORKDIR /app
COPY . .

# ==============================
# 6) Install Dependencies
# ==============================
RUN uv sync --frozen

# ==============================
# 7) Expose Port & Run App
# ==============================
EXPOSE 8080

CMD ["uv", "run", "app.py"]
