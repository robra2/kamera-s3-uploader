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
                ROI_X + ROI_WIDTH > frame_width or
                ROI_Y + ROI_HEIGHT > frame_height or
                ROI_WIDTH <= 0 or ROI_HEIGHT <= 0):
                print(f"WARNING_ROI: Konfigurierter ROI ({ROI_X},{ROI_Y},{ROI_WIDTH},{ROI_HEIGHT}) ist ungültig oder außerhalb des Bildbereichs ({frame_width}x{frame_height}). "
                      f"Verwende das gesamte Bild für diese Analyse.")
                use_roi_for_this_frame = False # Temporär für diesen Frame deaktivieren
            else:
                frame_for_analysis = frame[ROI_Y : ROI_Y + ROI_HEIGHT, ROI_X : ROI_X + ROI_WIDTH]
                # Adding debug print for successful ROI application
                print(f"DEBUG_ROI: ROI erfolgreich zugeschnitten. Analysiere Bereich: X={ROI_X}, Y={ROI_Y}, W={ROI_WIDTH}, H={ROI_HEIGHT}.")
        else:
            # Adding debug print when ROI is disabled
            print("DEBUG_ROI: ROI deaktiviert, analysiere gesamtes Bild.")
        
        # Graustufen und Weichzeichnen für bessere Erkennung
        gray = cv2.cvtColor(frame_for_analysis, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, BLUR_SIZE, 0)
        # Adding debug print for pre-processing steps
        print(f"DEBUG_PROCESSING: Bild in Graustufen konvertiert und Weichzeichner angewendet (Blur: {BLUR_SIZE}).")

        if avg_frame is None:
            avg_frame = gray.copy().astype("float")
            print("DEBUG_INIT: Initialisiere Durchschnitts-Frame.")
            continue

        # Aktuellen Frame mit dem Durchschnittsframe gewichten
        cv2.accumulateWeighted(gray, avg_frame, 0.5) 
        frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(avg_frame))
        # Adding debug print for frame delta calculation
        print(f"DEBUG_DIFF: Frame-Differenz berechnet. Min Delta: {np.min(frame_delta)}, Max Delta: {np.max(frame_delta)}.")

        # Schwellenwert anwenden
        thresh = cv2.threshold(frame_delta, THRESHOLD_DELTA, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2) # Dilatieren, um Lücken zu schließen
        # Adding debug print for thresholding and dilation
        print(f"DEBUG_THRESHOLD: Schwellenwert {THRESHOLD_DELTA} angewendet und dilatiert. Anzahl der weißen Pixel im Schwellenbild: {np.sum(thresh == 255)}.")

        # Konturen finden
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        significant_areas = [] # Speichere Flächen der relevanten Konturen
        # Adding debug print for number of contours found
        print(f"DEBUG_CONTOURS: {len(contours)} Konturen gefunden.")
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            # Adding debug print for each contour's area
            print(f"DEBUG_CONTOUR: Kontur {i}, Fläche: {area:.2f} Pixel. Mindestfläche (MIN_AREA): {MIN_AREA}.")
            if area < MIN_AREA:
                # Adding debug print for contours being too small
                print(f"DEBUG_CONTOUR: Kontur {i} zu klein ({area:.2f} < {MIN_AREA}). Überspringe.")
                continue
            
            motion_detected = True
            significant_areas.append(area)
            # break # Auskommentiert, um alle relevanten Flächen zu protokollieren, statt nur die erste

        if motion_detected:
            # Adding more detailed print when motion is detected
            print(f"DEBUG_MOTION: Bewegung erkannt! Signifikante Flächen: {significant_areas}. (Mindestfläche: {MIN_AREA}).")
        else:
            # Adding print when no motion is detected
            print("DEBUG_MOTION: Keine Bewegung erkannt.")


        current_time = time.time()
        cooldown_remaining = UPLOAD_COOLDOWN_SECONDS - (current_time - last_upload_time)

        # DEBUGGING DER UPLOAD-BEDINGUNG
        print(f"DEBUG_UPLOAD_CONDITION: Frame {frame_counter}. Bewegung erkannt: {motion_detected}. "
              f"Letzter Upload vor: {current_time - last_upload_time:.2f}s. Cooldown verbleibend: {cooldown_remaining:.2f}s.")

        if motion_detected and (current_time - last_upload_time) > UPLOAD_COOLDOWN_SECONDS:
            print(f"INFO: Bewegung erkannt und Cooldown abgelaufen. Bereite Bild-Upload vor...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bewegung_{timestamp}.jpg"
            s3_key = os.path.join(S3_UPLOAD_PREFIX, filename)

            # Für den Upload wird immer der aktuelle *Original*-Frame zugeschnitten und ggf. rotiert.
            upload_roi_x = ROI_X # Nutze die global definierten ROI-Werte
            upload_roi_y = ROI_Y
            upload_roi_w = ROI_WIDTH
            upload_roi_h = ROI_HEIGHT

            frame_to_upload = frame # Standardmäßig das gesamte Bild für den Upload

            # Prüfe, ob der Upload-ROI gültig ist und innerhalb des Bildes liegt
            if (USE_ROI_ENABLED and # Nur anwenden, wenn ROI generell aktiviert ist
                upload_roi_x >= 0 and upload_roi_y >= 0 and
                upload_roi_x + upload_roi_w <= frame_width and
                upload_roi_y + upload_roi_h <= frame_height and
                upload_roi_w > 0 and upload_roi_h > 0):
                
                frame_to_upload = frame[upload_roi_y : upload_roi_y + upload_roi_h,
                                        upload_roi_x : upload_roi_x + upload_roi_w]
                print(f"DEBUG_UPLOAD_ROI: Schneide Bild für Upload auf ROI ({upload_roi_x},{upload_roi_y},{upload_roi_w},{upload_roi_h}).")
            else:
                if USE_ROI_ENABLED:
                    print(f"WARNING_UPLOAD_ROI: Konfigurierter Upload-ROI ({upload_roi_x},{upload_roi_y},{upload_roi_w},{upload_roi_h}) ist ungültig oder außerhalb des Bildbereichs ({frame_width}x{frame_height}). "
                          f"Lade das gesamte Bild hoch.")
                else:
                    print("DEBUG_UPLOAD_ROI: Lade gesamtes Bild hoch (ROI ist deaktiviert).")
            
            # Rotation anwenden (wie in deinem Originalcode)
            (h_upload, w_upload) = frame_to_upload.shape[:2]
            center_upload = (w_upload // 2, h_upload // 2)
            angle_upload = 90 # Hardcoded, anpassbar

            M_upload = cv2.getRotationMatrix2D(center_upload, angle_upload, 1.0)
            abs_cos_upload = abs(M_upload[0, 0])
            abs_sin_upload = abs(M_upload[0, 1])
            new_w_upload = int(h_upload * abs_sin_upload + w_upload * abs_cos_upload)
            new_h_upload = int(h_upload * abs_cos_upload + w_upload * abs_sin_upload)
            M_upload[0, 2] += new_w_upload / 2 - center_upload[0]
            M_upload[1, 2] += new_h_upload / 2 - center_h_upload # Korrigiert: h_upload nicht h_center
            frame_to_upload_rotated = cv2.warpAffine(frame_to_upload, M_upload, (new_w_upload, new_h_upload))
            print(f"DEBUG_UPLOAD_ROTATION: Bild vor Upload um {angle_upload} Grad rotiert.")
            
            # Bild in JPEG-Format kodieren
            ret_code, jpg_buffer = cv2.imencode(".jpg", frame_to_upload_rotated)
            if ret_code:
                if upload_image_to_s3(jpg_buffer.tobytes(), s3_key):
                    last_upload_time = current_time
            else:
                print("ERROR: Fehler beim Kodieren des Bildes.")

    cap.release()

if __name__ == "__main__":
    run_motion_detection()
