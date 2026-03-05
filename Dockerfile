FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
CMD ["python", "app.py"]