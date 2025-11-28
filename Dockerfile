# ==============================
# 1) Base Image
# ==============================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ==============================
# 2) System Dependencies
# ==============================
RUN apt-get update && apt-get install -y \
    wget curl git unzip \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
    libxkbcommon0 libgtk-3-0 libgbm1 libasound2 \
    libxcomposite1 libxdamage1 libxrandr2 libxfixes3 \
    libpango-1.0-0 libcairo2 \
    tesseract-ocr \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ==============================
# 3) Install Playwright
# ==============================
RUN pip install playwright
RUN playwright install --with-deps chromium

# ==============================
# 4) Install uv
# ==============================
RUN pip install uv

# ==============================
# 5) Prepare App Folder
# ==============================
WORKDIR /app

# Copy only dependency files FIRST
COPY pyproject.toml uv.lock ./

# Install deps with uv
RUN uv sync --frozen

# Copy code AFTER deps installation
COPY . .

# ==============================
# 6) Expose port (Railway uses 8080)
# ==============================
EXPOSE 8080

# ==============================
# 7) Run server
# ==============================
CMD ["uv", "run", "app.py"]
