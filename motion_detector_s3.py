import cv2
import boto3
import os
import time
from datetime import datetime

# --- Konfiguration ---
# ANPASSEN: RTSP-Stream URL deiner IP-Kamera
# Beispiel: rtsp://username:password@192.168.1.100:554/stream
RTSP_STREAM_URL = "rtsp://DEIN_BENUTZERNAME:DEIN_PASSWORT@DEINE_KAMERA_IP:DEIN_PORT/DEIN_STREAM_PFAD"

# ANPASSEN: Dein S3 Bucket Name
S3_BUCKET_NAME = "mein-kamera-bewegung-bilder"

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
USE_ROI = False
ROI_X = 100    # X-Koordinate der oberen linken Ecke des Bereichs
ROI_Y = 100    # Y-Koordinate der oberen linken Ecke des Bereichs
ROI_WIDTH = 400  # Breite des Bereichs
ROI_HEIGHT = 300 # Höhe des Bereichs

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

            # Bild in JPEG-Format kodieren
            ret_code, jpg_buffer = cv2.imencode(".jpg", frame)
            if ret_code:
                if upload_image_to_s3(jpg_buffer.tobytes(), s3_key):
                    last_upload_time = current_time
            else:
                print("Fehler beim Kodieren des Bildes.")

        # Optional: Zeige den Frame zur Debugging-Zwecken (nicht für Docker-Container in Produktion)
        # cv2.imshow("Kamera Stream", frame)
        # cv2.imshow("Threshold", thresh)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    # cv2.destroyAllWindows() # Nur nötig, wenn imshow verwendet wird

if __name__ == "__main__":
    run_motion_detection()
