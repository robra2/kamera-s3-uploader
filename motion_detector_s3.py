import cv2
import numpy as np
import boto3
import os
import time
from datetime import datetime

# --- Konfiguration ---
# ANPASSEN: RTSP-Stream URL deiner IP-Kamera
# Beispiel: rtsp://username:password@192.168.1.100:554/stream
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
# Wenn du einen spezifischen Bereich überwachen möchtest, setze USE_ROI auf True
# und passe die ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT an.
# Andernfalls bleibt USE_ROI auf False und das gesamte Bild wird überwacht.
USE_ROI = True
# Werte aus env variablen lesen
ROI_X = int(os.environ.get("ROI_X", 0))    # X-Koordinate der oberen linken Ecke des Bereichs
ROI_Y = int(os.environ.get("ROI_Y", 0))    # Y-Koordinate der oberen linken Ecke des Bereichs
ROI_WIDTH = int(os.environ.get("ROI_W", 100))  # Breite des Bereichs
ROI_HEIGHT = int(os.environ.get("ROI_H", 100)) # Höhe des Bereichs

# Upload-Intervall nach Bewegungserkennung (um nicht zu viele Bilder hochzuladen)
UPLOAD_COOLDOWN_SECONDS = 5 # Wartezeit nach einem Upload, bevor ein neues Bild hochgeladen wird

# --- AWS S3 Client initialisieren ---
s3_client = boto3.client('s3')

def upload_image_to_s3(image_data, s3_key):
    """Lädt ein Bild (als Bytes) nach S3 hoch."""
    try:
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=s3_key, Body=image_data, ContentType='image/jpeg')
        print(f"Bild {s3_key} erfolgreich nach S3 hochgeladen.")
        return True
    except Exception as e:
        print(f"Fehler beim Hochladen von {s3_key} nach S3: {e}")
        return False

def run_motion_detection():
    cap = cv2.VideoCapture(RTSP_STREAM_URL)

    if not cap.isOpened():
        print(f"Fehler: Kann RTSP-Stream nicht öffnen: {RTSP_STREAM_URL}")
        print("Bitte überprüfen Sie die URL, Benutzername/Passwort und ob die Kamera erreichbar ist.")
        exit(1)

    print(f"RTSP-Stream '{RTSP_STREAM_URL}' geöffnet. Starte Bewegungserkennung...")

    avg_frame = None
    last_upload_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream verloren, versuche neu zu verbinden...")
            cap.release()
            time.sleep(5) # Kurze Pause vor dem Neuverbinden
            cap = cv2.VideoCapture(RTSP_STREAM_URL)
            if not cap.isOpened():
                print("Neuverbindung fehlgeschlagen, beende Skript.")
                break
            avg_frame = None # Reset des Referenzbildes
            continue

        # Optional: Bereich von Interesse (ROI) zuschneiden
        if USE_ROI:
            frame_for_analysis = frame[ROI_Y:ROI_Y+ROI_HEIGHT, ROI_X:ROI_X+ROI_WIDTH]
            if frame_for_analysis.size == 0: # Prüfen, ob der ROI gültig ist
                print("Fehler: ROI ist außerhalb des Bildbereichs oder hat keine Größe. Deaktiviere ROI.")
                USE_ROI = False
                frame_for_analysis = frame # Fallback auf gesamtes Bild
        else:
            frame_for_analysis = frame

        # Graustufen und Weichzeichnen für bessere Erkennung
        gray = cv2.cvtColor(frame_for_analysis, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, BLUR_SIZE, 0)

        if avg_frame is None:
            avg_frame = gray.copy().astype("float")
            continue

        # Aktuellen Frame mit dem Durchschnittsframe gewichten
        cv2.accumulateWeighted(gray, avg_frame, 0.5) # 0.5 ist der Gewichtungsfaktor
        frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(avg_frame))

        # Schwellenwert anwenden
        thresh = cv2.threshold(frame_delta, THRESHOLD_DELTA, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2) # Dilatieren, um Lücken zu schließen

        # Konturen finden
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        for c in contours:
            if cv2.contourArea(c) < MIN_AREA:
                continue
            motion_detected = True
            break # Bewegung im ROI erkannt, kein weiterer Kontur-Check nötig

        current_time = time.time()
        if motion_detected and (current_time - last_upload_time) > UPLOAD_COOLDOWN_SECONDS:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bewegung_{timestamp}.jpg"
            s3_key = os.path.join(S3_UPLOAD_PREFIX, filename)

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

            

            # Bild in JPEG-Format kodieren
            ret_code, jpg_buffer = cv2.imencode(".jpg", frame_roi_rot)
            if ret_code:
                if upload_image_to_s3(jpg_buffer.tobytes(), s3_key):
                    last_upload_time = current_time
            else:
                print("Fehler beim Kodieren des Bildes.")



    cap.release()
    # cv2.destroyAllWindows() # Nur nötig, wenn imshow verwendet wird

if __name__ == "__main__":
    run_motion_detection()










import cv2


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
