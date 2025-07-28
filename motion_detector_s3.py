import cv2
import numpy as np
import boto3
import os
import time
from datetime import datetime

### DEBUG ###
print("--- Skript wird gestartet ---")

# --- Konfiguration ---
# ANPASSEN: RTSP-Stream URL deiner IP-Kamera
# Beispiel: rtsp://username:password@192.168.1.100:554/stream
RTSP_STREAM_URL = "rtsp://admin:112Burger@192.168.178.62:554/h264Preview_01_sub"

# ANPASSEN: Dein S3 Bucket Name
S3_BUCKET_NAME = "meine-tor-kamera-bilder"

# ANPASSEN: Pfad im S3 Bucket, wo die Bilder gespeichert werden sollen (optional)
S3_UPLOAD_PREFIX = "bewegungserkennung/"

# Bewegungserkennungs-Parameter
MIN_AREA = 2500
THRESHOLD_DELTA = 40 # Schwellenwert für die Differenz zwischen Frames
BLUR_SIZE = (41, 41) # Größe des Gaußschen Weichzeichners

# Bereich für Bewegungserkennung definieren (optional)
# Wenn du einen spezifischen Bereich überwachen möchtest, setze USE_ROI auf True
# und passe die ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT an.
# Andernfalls bleibt USE_ROI auf False und das gesamte Bild wird überwacht.
USE_ROI_INIT = True
# Werte aus env variablen lesen
ROI_X = int(os.environ.get("ROI_X", 0))      # X-Koordinate der oberen linken Ecke des Bereichs
ROI_Y = int(os.environ.get("ROI_Y", 0))      # Y-Koordinate der oberen linken Ecke des Bereichs
ROI_WIDTH = int(os.environ.get("ROI_W", 100))  # Breite des Bereichs
ROI_HEIGHT = int(os.environ.get("ROI_H", 100)) # Höhe des Bereichs

# Upload-Intervall nach Bewegungserkennung (um nicht zu viele Bilder hochzuladen)
UPLOAD_COOLDOWN_SECONDS = 5 # Wartezeit nach einem Upload, bevor ein neues Bild hochgeladen wird

### DEBUG ###
print("--- Konfiguration geladen ---")
print(f"RTSP_STREAM_URL: {RTSP_STREAM_URL}")
print(f"S3_BUCKET_NAME: {S3_BUCKET_NAME}")
print(f"S3_UPLOAD_PREFIX: {S3_UPLOAD_PREFIX}")
print(f"MIN_AREA: {MIN_AREA}")
print(f"THRESHOLD_DELTA: {THRESHOLD_DELTA}")
print(f"BLUR_SIZE: {BLUR_SIZE}")
print(f"USE_ROI_INIT: {USE_ROI_INIT}")
print(f"ROI_X: {ROI_X}")
print(f"ROI_Y: {ROI_Y}")
print(f"ROI_WIDTH: {ROI_WIDTH}")
print(f"ROI_HEIGHT: {ROI_HEIGHT}")
print(f"UPLOAD_COOLDOWN_SECONDS: {UPLOAD_COOLDOWN_SECONDS}")
print("--------------------------")


# --- AWS S3 Client initialisieren ---
### DEBUG ####
print("### DEBUG ### Initialisiere AWS S3 Client...")
s3_client = boto3.client('s3')
### DEBUG ###
print("### DEBUG ### AWS S3 Client initialisiert.")


def upload_image_to_s3(image_data, s3_key):
    """Lädt ein Bild (als Bytes) nach S3 hoch."""
    try:
        ### DEBUG ###
        print(f"### DEBUG ### Versuche, Bild nach S3 hochzuladen: Bucket={S3_BUCKET_NAME}, Key={s3_key}")
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=s3_key, Body=image_data, ContentType='image/jpeg')
        print(f"Bild {s3_key} erfolgreich nach S3 hochgeladen.")
        return True
    except Exception as e:
        print(f"Fehler beim Hochladen von {s3_key} nach S3: {e}")
        return False

def run_motion_detection():
    ### DEBUG ###
    print("### DEBUG ### Starte run_motion_detection Funktion.")
    cap = cv2.VideoCapture(RTSP_STREAM_URL)

    if not cap.isOpened():
        print(f"Fehler: Kann RTSP-Stream nicht öffnen: {RTSP_STREAM_URL}")
        print("Bitte überprüfen Sie die URL, Benutzername/Passwort und ob die Kamera erreichbar ist.")
        exit(1)

    print(f"RTSP-Stream '{RTSP_STREAM_URL}' geöffnet. Starte Bewegungserkennung...")

    avg_frame = None
    last_upload_time = 0
    USE_ROI = USE_ROI_INIT
    frame_count = 0

    Upload_Frames = True

    while True:
        ### DEBUG ###
        frame_count += 1
        print(f"\n### DEBUG ### Hauptschleife, Frame #{frame_count}")

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

        ### DEBUG ###
        print(f"### DEBUG ### Frame empfangen. Dimensionen: {frame.shape}")

        # Optional: Bereich von Interesse (ROI) zuschneiden
        if USE_ROI:
            ### DEBUG ###
            print(f"### DEBUG ### ROI ist aktiviert. Schneide Frame zu: Y({ROI_Y}:{ROI_Y+ROI_HEIGHT}), X({ROI_X}:{ROI_X+ROI_WIDTH})")
            frame_for_analysis = frame[ROI_Y:ROI_Y+ROI_HEIGHT, ROI_X:ROI_X+ROI_WIDTH]
            if frame_for_analysis.size == 0: # Prüfen, ob der ROI gültig ist
                print("Fehler: ROI ist außerhalb des Bildbereichs oder hat keine Größe. Deaktiviere ROI.")
                USE_ROI = False
                frame_for_analysis = frame # Fallback auf gesamtes Bild
        else:
            ### DEBUG ###
            print("### DEBUG ### ROI ist deaktiviert. Nutze gesamten Frame für Analyse.")
            frame_for_analysis = frame

        ### DEBUG ###
        print(f"### DEBUG ### Frame für Analyse hat Dimensionen: {frame_for_analysis.shape}")

        # Graustufen und Weichzeichnen für bessere Erkennung
        gray = cv2.cvtColor(frame_for_analysis, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 21)  # Median-Filter gegen Blockrauschen

        if avg_frame is None:
            ### DEBUG ###
            print("### DEBUG ### Initialisiere Referenzbild (avg_frame).")
            avg_frame = gray.copy().astype("float")
            continue

        # Aktuellen Frame mit dem Durchschnittsframe gewichten
        cv2.accumulateWeighted(gray, avg_frame, 0.2) # 0.5 ist der Gewichtungsfaktor
        frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(avg_frame))
        if Upload_Frames:

            timestamp_delta = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_delta = f"Delta_{timestamp_delta}.jpg"
            s3_key_delta = os.path.join(S3_UPLOAD_PREFIX, filename_delta)

            ret_code_delta, jpg_buffer_delta = cv2.imencode(".jpg", frame_delta)

            upload_image_to_s3(jpg_buffer_delta.tobytes(), s3_key_delta)
            Upload_Frames = False # Nur einmal pro Frame hochladen

        # Schwellenwert anwenden
        thresh = cv2.threshold(frame_delta, THRESHOLD_DELTA, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=3) # Dilatieren, um Lücken zu schließen

        # Konturen finden
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ### DEBUG ###
        print(f"### DEBUG ### {len(contours)} Konturen gefunden.")

        motion_detected = False
        for c in contours:
            contour_area = cv2.contourArea(c)
            ### DEBUG ###
            print(f"### DEBUG ### Prüfe Kontur mit Fläche: {contour_area}")
            if contour_area < MIN_AREA:
                continue
            
            ### DEBUG ###
            print(f"### DEBUG ### BEWEGUNG ERKANNT! Fläche {contour_area} >= Min_Area {MIN_AREA}")
            motion_detected = True
            break # Bewegung im ROI erkannt, kein weiterer Kontur-Check nötig

        current_time = time.time()
        
        ### DEBUG ###
        cooldown_check = (current_time - last_upload_time) > UPLOAD_COOLDOWN_SECONDS
        print(f"### DEBUG ### motion_detected: {motion_detected}, Cooldown abgelaufen: {cooldown_check}")

        if motion_detected and cooldown_check:
            ### DEBUG ###
            print("### DEBUG ### Bedingungen für Upload erfüllt. Starte Upload-Prozess.")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bewegung_{timestamp}.jpg"
            s3_key = os.path.join(S3_UPLOAD_PREFIX, filename)
            
            ### DEBUG ###
            print(f"### DEBUG ### Zieldateiname: {s3_key}")

            # ROI-Werte aus Environment lesen
            x = int(os.environ.get("ROI_X", 0))
            y = int(os.environ.get("ROI_Y", 0))
            w = int(os.environ.get("ROI_W", 100))
            h = int(os.environ.get("ROI_H", 100))
            
            ### DEBUG ###
            print(f"### DEBUG ### Lese ROI-Werte für Upload-Bild: X={x}, Y={y}, W={w}, H={h}")
            
            frame_roi = frame[y:y+h, x:x+w]
            
            ### DEBUG ###
            print(f"### DEBUG ### Upload-Bild zugeschnitten. Dimensionen: {frame_roi.shape}")
            
            (h, w) = frame_roi.shape[:2]
            center = (w // 2, h // 2)
            angle = 90

            ### DEBUG ###
            print(f"### DEBUG ### Rotiere Bild um {angle} Grad.")

            # Rotationsmatrix
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Neue Dimensionen berechnen
            abs_cos = abs(M[0, 0])
            abs_sin = abs(M[0, 1])
            new_w = int(h * abs_sin + w * abs_cos)
            new_h = int(h * abs_cos + w * abs_sin)
            
            ### DEBUG ###
            print(f"### DEBUG ### Neue Dimensionen nach Rotation: {new_w}x{new_h}")

            # Matrix anpassen
            M[0, 2] += new_w / 2 - center[0]
            M[1, 2] += new_h / 2 - center[1]

            # Bild rotieren
            frame_roi_rot = cv2.warpAffine(frame_roi, M, (new_w, new_h))

            # Bild in JPEG-Format kodieren
            ### DEBUG ###
            print("### DEBUG ### Kodiere rotiertes Bild nach JPEG.")
            ret_code, jpg_buffer = cv2.imencode(".jpg", frame_roi_rot)
            if ret_code:
                if upload_image_to_s3(jpg_buffer.tobytes(), s3_key):
                    last_upload_time = current_time
                    ### DEBUG ###
                    print(f"### DEBUG ### Upload erfolgreich. last_upload_time auf {last_upload_time} gesetzt.")
            else:
                print("Fehler beim Kodieren des Bildes.")

    ### DEBUG ###
    print("### DEBUG ### Hauptschleife beendet.")
    cap.release()
    ### DEBUG ###
    print("### DEBUG ### Video-Capture freigegeben.")
    # cv2.destroyAllWindows() # Nur nötig, wenn imshow verwendet wird

if __name__ == "__main__":
    ### DEBUG ###
    print("### DEBUG ### Skript wird direkt ausgeführt (__name__ == '__main__').")
    run_motion_detection()
    ### DEBUG ###
    print("--- Skript beendet ---")
