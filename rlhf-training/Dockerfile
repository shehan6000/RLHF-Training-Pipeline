FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY main.py .

# Set Python path
ENV PYTHONPATH=/app

# Run the application
CMD ["python", "main.py", "--config", "configs/base_config.yaml"]