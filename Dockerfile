# ---------- Stage 1: base (DEV / TEST) ----------
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure logs are flushed immediately
ENV PYTHONUNBUFFERED=1

# Working directory inside the container
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Default command (flexible)
# 👉 This is what Docker Compose will use
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]


# ---------- Stage 2: prod (LOCKED IMAGE) ----------
FROM base AS prod

# Lock the execution contract
# 👉 This overrides CMD and CANNOT be replaced accidentally
ENTRYPOINT ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]