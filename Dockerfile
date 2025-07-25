# Verwende ein schlankes Python-Image als Basis
FROM python:3.9-slim-buster

# Installiere Systemabhängigkeiten für OpenCV
# libgl1-mesa-glx und libsm6 sind oft für OpenCV nötig, auch in Headless-Versionen
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Lege das Arbeitsverzeichnis im Container fest
WORKDIR /app

# Kopiere die Python-Abhängigkeiten und installiere sie
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiere dein Python-Skript in den Container
COPY motion_detector_s3.py .

# Befehl, der ausgeführt wird, wenn der Container startet
CMD ["python", "motion_detector_s3.py"]
