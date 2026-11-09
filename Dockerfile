# ==============================================================================
# Dockerfile - Production Container Image for Loan Approval AI
# ==============================================================================

# 1. Use an official lightweight Python runtime
FROM python:3.12-slim

# 2. Prevent Python from buffering stdout/stderr and writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Install build dependencies and clean up apt cache to keep image lean
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy and install Python dependencies (cached Docker layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy application code, trained ML model, and configuration files
COPY . .

# 7. Expose the port FastAPI runs on
EXPOSE 8000

# 8. Start the FastAPI server using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
