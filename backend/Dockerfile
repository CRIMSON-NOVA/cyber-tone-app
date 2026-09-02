FROM python:3.11-slim

WORKDIR /app

# System deps for audio/torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Render/Railway inject $PORT at runtime; main.py reads it via os.environ.
EXPOSE 8000
CMD ["python", "main.py"]
