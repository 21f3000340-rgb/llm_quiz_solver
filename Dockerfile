# ==============================
# 1️⃣ Base Image
# ==============================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ==============================
# 2️⃣ Working Directory
# ==============================
WORKDIR /app

# ==============================
# 3️⃣ System Dependencies
# ==============================
RUN apt-get update && apt-get install -y \
    wget curl git chromium \
    g++ gcc \
    libffi-dev libssl-dev \
    libjpeg-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# ==============================
# 4️⃣ Copy Files
# ==============================
COPY . .

# ==============================
# 5️⃣ Install Python Packages
# ==============================
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install Playwright Chromium
RUN python -m playwright install chromium

# ==============================
# 6️⃣ Expose Port & Run
# ==============================
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
