FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose the API port used by Hugging Face container runtime
EXPOSE 7860

# Default command to run the OpenEnv API server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
