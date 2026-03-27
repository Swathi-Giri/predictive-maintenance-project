FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (Docker caches this layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose ports: 8000 for API, 8501 for Dashboard
EXPOSE 8000 8501

# Default: run the API (override in docker-compose for dashboard)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
