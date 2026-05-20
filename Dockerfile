# Use a slim Python base image compatible with the repo dependencies
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Install system dependencies required for some Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       ca-certificates \
       libglib2.0-0 \
       libsm6 \
       libxext6 \
       libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list first to leverage Docker cache
COPY requirements.txt ./

# Upgrade pip and install dependencies
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . /app

# Expose any application ports here if needed (example placeholder)
# EXPOSE 8080

# Default command to run the trading bot
CMD ["python", "main.py"]
