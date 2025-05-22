FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (if needed, e.g., for Pillow or numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the app
COPY server.py .
COPY GradCam ./GradCam
COPY Models ./Models
COPY templates ./templates
COPY static ./static

EXPOSE 5000

CMD ["python", "server.py"]