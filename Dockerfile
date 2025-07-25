# Verwende ein aktuelles Python-Image als Basis
FROM python:3.12-slim-bookworm # Oder python:3.11-slim-bookworm

# Setze Umgebungsvariablen, um interaktive Abfragen während des Builds zu vermeiden
ENV DEBIAN_FRONTEND=noninteractive

# Installiere Systemabhängigkeiten für OpenCV und FFmpeg
# Hier können Sie die RUN-Befehle wieder zusammenfassen,
# sobald Sie das ursprüngliche Installationsproblem gelöst haben.
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Lege das Arbeitsverzeichnis im Container fest
WORKDIR /app

# Kopiere die Python-Abhängigkeiten und installiere sie
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiere dein Python-Skript in den Container
COPY motion_detector_s3.py .

# Befehl, der ausgeführt wird, wenn der Container startet
CMD ["python", "motion_detector_s3.py"]
