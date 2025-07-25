import cv2
import numpy as np
import boto3
import time
import os
from datetime import datetime

# --- Konfiguration ---
# ANPASSEN: Dein Kamera RTSP URL
# Beispiel: "rtsp://user:password@192.168.1.100:554/stream"
RTSP_URL = "rtsp://admin:112Burger@192.168.178.62:554/h264Preview_01_main"
S3_BUCKET_NAME = "meine-tor-kamera-bilder"             # ANPASSEN: Dein S3 Bucket Name (muss existieren!)
UPLOAD_INTERVAL_SECONDS = 5                                # Wie oft ein Bild hochgeladen werden soll (in Sekunden)
IMAGE_QUALITY = 85                                         # JPEG-Qualität (0-100), 90 ist ein guter Start

# AWS S3 Client initialisieren
# boto3 sucht automatisch nach AWS_ACCESS_KEY_ID und AWS_SECRET_ACCESS_KEY aus Umgebungsvariablen.
s3_client = boto3.client('s3')

def capture_and_upload():
    print(f"Starte Uploader für Bucket: {S3_BUCKET_NAME}, Intervall: {UPLOAD_INTERVAL_SECONDS}s")

    while True:
        cap = cv2.VideoCapture(RTSP_URL) # Stream bei jedem Versuch neu öffnen für Robustheit

        if not cap.isOpened():
            print(f"Fehler: Konnte den RTSP-Stream von {RTSP_URL} nicht öffnen. Nächster Versuch in {UPLOAD_INTERVAL_SECONDS} Sekunden.")
            time.sleep(UPLOAD_INTERVAL_SECONDS)
            continue

        print(f"RTSP-Stream von {RTSP_URL} erfolgreich geöffnet. Warte auf Frame...")

        ret, frame = cap.read() # Versuch, einen Frame zu lesen

        # Stream sofort wieder freigeben, um Ressourcen zu schonen
        cap.release() 
        
        if not ret:
            print("Fehler: Konnte keinen Frame vom Stream lesen (evtl. nur schwarzes Bild?). Nächster Versuch...")
            time.sleep(UPLOAD_INTERVAL_SECONDS)
            continue
            
        # ROI-Werte aus Environment lesen
        x = int(os.environ.get("ROI_X", 0))
        y = int(os.environ.get("ROI_Y", 0))
        w = int(os.environ.get("ROI_W", 100))
        h = int(os.environ.get("ROI_H", 100))
        frame_roi= frame[y:y+h, x:x+w]
        
        
        (h, w) = frame_roi.shape[:2]
        center = (w // 2, h // 2)
        angle = 90

        # Rotationsmatrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Neue Dimensionen berechnen
        abs_cos = abs(M[0, 0])
        abs_sin = abs(M[0, 1])
        new_w = int(h * abs_sin + w * abs_cos)
        new_h = int(h * abs_cos + w * abs_sin)

        # Matrix anpassen
        M[0, 2] += new_w / 2 - center[0]
        M[1, 2] += new_h / 2 - center[1]

        # Bild rotieren
        frame_roi_rot = cv2.warpAffine(frame_roi, M, (new_w, new_h))


        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Generiere einen eindeutigeren Dateinamen, falls mehrere Bilder pro Sekunde hochgeladen werden sollen
        # oder nur zur besseren Übersicht
        s3_object_key = f"kamera_frames/Ueberwachung_Ch6_Einfahrt_{timestamp_str}.jpg" 

        # Frame als JPEG im Speicher kodieren
        ret, buffer = cv2.imencode('.jpg', frame_roi_rot, [int(cv2.IMWRITE_JPEG_QUALITY), IMAGE_QUALITY])
        if not ret:
            print(f"Fehler: Konnte Frame nicht als JPEG kodieren.")
            time.sleep(UPLOAD_INTERVAL_SECONDS)
            continue

        try:
            # Bild direkt von Bytes-Buffer nach S3 hochladen
            s3_client.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=s3_object_key,
                Body=buffer.tobytes(), # Der Bytes-Buffer des Bildes
                ContentType='image/jpeg'
            )
            print(f"Bild {s3_object_key} erfolgreich nach S3 hochgeladen.")
        except Exception as e:
            print(f"Fehler beim Hochladen nach S3: {e}")

        time.sleep(UPLOAD_INTERVAL_SECONDS)

if __name__ == "__main__":
    capture_and_upload()
