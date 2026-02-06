FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia TUTTO il progetto
COPY . /app

CMD ["python", "Main.py"]
