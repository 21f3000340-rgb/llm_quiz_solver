FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    wget curl git \
    g++ gcc \
    libffi-dev libssl-dev \
    libjpeg-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN python -m playwright install chromium --only-shell
RUN rm -rf /root/.cache

EXPOSE 8000

# ⭐ This is the important fix
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"]
