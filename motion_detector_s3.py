import cv2
import numpy as np
import boto3
import os
import time
from datetime import datetime

# --- Konfiguration ---
# ANPASSEN: RTSP-Stream URL deiner IP-Kamera
RTSP_STREAM_URL = "rtsp://admin:112Burger@192.168.178.62:554/h264Preview_01_main"

# ANPASSEN: Dein S3 Bucket Name
S3_BUCKET_NAME = "meine-tor-kamera-bilder"

# ANPASSEN: Pfad im S3 Bucket, wo die Bilder gespeichert werden sollen (optional)
S3_UPLOAD_PREFIX = "bewegungserkennung/"

# Bewegungserkennungs-Parameter
MIN_AREA = 500       # Minimale Fläche (in Pixeln) für die Bewegungserkennung
THRESHOLD_DELTA = 5  # Schwellenwert für die Bilddifferenz (je höher, desto weniger empfindlich)
BLUR_SIZE = (21, 21) # Größe des Gaußschen Weichzeichners

# Bereich für Bewegungserkennung definieren (optional)
# Wenn du einen spezifischen Bereich überwachen möchtest, setze USE_ROI_ENABLED auf True
# und passe die ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT an.
# Andernfalls bleibt USE_ROI_ENABLED auf False und das gesamte Bild wird überwacht.
USE_ROI_ENABLED = True
# Werte aus env variablen lesen
# WICHTIG: Setze sinnvolle Standardwerte für W/H, die zur Kameraauflösung passen.
# Beispiel: 1280x720 oder 1920x1080. Die hier sind nur Platzhalter.
ROI_X = int(os.environ.get("ROI_X", 0))
ROI_Y = int(os.environ.get("ROI_Y", 0))
ROI_WIDTH = int(os.environ.get("ROI_W", 640))   # ANGEPASST: Beispielwert, bitte anpassen!
ROI_HEIGHT = int(os.environ.get("ROI_H", 480)) # ANGEPASST: Beispielwert, bitte anpassen!


# Upload-Intervall nach Bewegungserkennung (um nicht zu viele Bilder hochzuladen)
UPLOAD_COOLDOWN_SECONDS = 5 # Wartezeit nach einem Upload, bevor ein neues Bild hochgeladen wird

# --- AWS S3 Client initialisieren ---
s3_client = boto3.client('s3')

def upload_image_to_s3(image_data, s3_key):
    """Lädt ein Bild (als Bytes) nach S3 hoch."""
    try:
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=s3_key, Body=image_data, ContentType='image/jpeg')
        print(f"DEBUG_UPLOAD: Bild {s3_key} erfolgreich nach S3 hochgeladen.")
        return True
    except Exception as e:
        print(f"DEBUG_UPLOAD: Fehler beim Hochladen von {s3_key} nach S3: {e}")
        return False

def run_motion_detection():
    cap = cv2.VideoCapture(RTSP_STREAM_URL)

    if not cap.isOpened():
        print(f"ERROR: Kann RTSP-Stream nicht öffnen: {RTSP_STREAM_URL}")
        print("INFO: Bitte überprüfen Sie die URL, Benutzername/Passwort und ob die Kamera erreichbar ist.")
        exit(1)

    print(f"INFO: RTSP-Stream '{RTSP_STREAM_URL}' geöffnet. Starte Bewegungserkennung...")

    avg_frame = None
    last_upload_time = 0

    frame_counter = 0 # Zähler für Debug-Zwecke

    while True:
        ret, frame = cap.read()
        frame_counter += 1 # Frame-Zähler erhöhen
        
        if not ret:
            print("WARNING: Stream verloren, versuche neu zu verbinden...")
            cap.release()
            time.sleep(5) # Kurze Pause vor dem Neuverbinden
            cap = cv2.VideoCapture(RTSP_STREAM_URL)
            if not cap.isOpened():
                print("ERROR: Neuverbindung fehlgeschlagen, beende Skript.")
                break
            avg_frame = None # Reset des Referenzbildes
            print("INFO: Erfolgreich neu verbunden.")
            continue

        # Ermittle die tatsächliche Frame-Größe
        frame_height, frame_width = frame.shape[:2]
        # Adding debug print for frame dimensions
        print(f"DEBUG_FRAME: Frame {frame_counter} gelesen. Dimensionen: {frame_width}x{frame_height}.")

        # Bestimme den ROI für die Analyse
        use_roi_for_this_frame = USE_ROI_ENABLED
        frame_for_analysis = frame # Standardmäßig das gesamte Bild verwenden

        if use_roi_for_this_frame:
            # Adding debug print for ROI coordinates before validation
            print(f"DEBUG_ROI: USE_ROI_ENABLED ist True. Versuche ROI: X={ROI_X}, Y={ROI_Y}, W={ROI_WIDTH}, H={ROI_HEIGHT}.")
            # Prüfen, ob der ROI gültig ist und innerhalb des Bildes liegt
            if (ROI_X < 0 or ROI_Y < 0 or
                ROI_
