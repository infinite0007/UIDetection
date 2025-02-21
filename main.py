# main.py
import cv2 as cv
import numpy as np
import easyocr  # https://github.com/jaidedai/easyocr
import sys
import imutils

# Import der JSON-Funktionen aus dem neuen Skript
from display_manager import save_display_corners, load_display_corners

def read_image(path):
    """Liest ein Bild ein, ohne es zu skalieren."""
    img = cv.imread(path)
    if img is None:
        sys.exit(f"Could not read the image at {path}.")
    return img

def detect_display_corners(img, canny_thresh1=50, canny_thresh2=150, approx_factor=0.007):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, canny_thresh1, canny_thresh2)

    # Kanten "verbreitern"
    kernel = np.ones((2, 2), np.uint8)
    edges = cv.dilate(edges, kernel, iterations=1)

    cv.imshow("Canny After Morphology", edges)
    cv.waitKey(0)
    cv.destroyAllWindows()

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_approx = None
    best_area = 0

    # Definiere eine Mindestfläche, z.B. 2000
    min_area = 2000

    for cnt in contours:
        perimeter = cv.arcLength(cnt, True)
        if perimeter < 10:
            continue
        epsilon = approx_factor * perimeter
        approx = cv.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:
            area = cv.contourArea(approx)
            # Neu: Ignoriere zu kleine Vierecke
            if area < min_area:
                continue

            # Wenn groß genug, checken wir, ob es das größte ist
            if area > best_area:
                best_area = area
                best_approx = approx

    if best_approx is None:
        return None

    # Sortieren [TL, TR, BR, BL]
    corners = sorted(best_approx[:, 0], key=lambda p: (p[1], p[0]))
    if corners[0][0] > corners[1][0]:
        corners[0], corners[1] = corners[1], corners[0]
    if corners[2][0] < corners[3][0]:
        corners[2], corners[3] = corners[3], corners[2]

    return np.array(corners, dtype=np.float32)

def apply_perspective_transform(img, corners, target_size):
    """
    Wendet eine Perspektivtransformation anhand gegebener Ecken auf das Bild an.
    Gibt das transformierte Bild mit Größe target_size zurück.
    """
    dst_pts = np.array([
        [0, 0],                # Top Left
        [target_size[0], 0],   # Top Right
        [target_size[0], target_size[1]],   # Bottom Right
        [0, target_size[1]]    # Bottom Left
    ], dtype=np.float32)

    M = cv.getPerspectiveTransform(corners, dst_pts)
    warped = cv.warpPerspective(img, M, target_size)
    return warped

def calibrate_display(img):
    """
    Sucht die Display-Ecken im Bild und speichert sie per JSON.
    Falls kein Display gefunden wird, wird use_full_image=True gespeichert.
    """
    corners = detect_display_corners(img)
    if corners is not None:
        save_display_corners(corners, use_full_image=False)
        print("Kalibrierung erfolgreich. Display-Ecken erkannt und gespeichert.")
    else:
        save_display_corners(None, use_full_image=True)
        print("Keine Display-Ecken erkannt. Verwende Originalbild.")

def rectify_display_image(img, target_size=(320, 240)):
    """
    Lädt die Kalibrierungsdaten (Ecken / use_full_image) aus JSON und
    führt die Transformation nur aus, wenn Ecken vorhanden sind.
    Ansonsten wird das Bild unverändert zurückgegeben.
    """
    corners, use_full_image = load_display_corners()
    # Prüfen, ob use_full_image oder corners fehlerhaft
    if use_full_image or corners is None or len(corners) != 4:
        return img
    # Ansonsten Transformation anwenden
    return apply_perspective_transform(img, corners, target_size)

def analyze_colors(img):
    """
    Analysiert die Anzahl der Pixel pro Farbe (B, G, R, Schwarz, Weiß).
    """
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    black_tolerance = 30
    white_tolerance = 220
    
    # Alle möglichen Farbnamen vorab mit 0 initialisieren
    color_names = ["black", "white", "red", "yellow", "green", "cyan", "blue", "magenta", "unknown"]
    color_counts = {name: 0 for name in color_names}

    for row in hsv_image:
        for pixel in row:
            h, s, v = pixel

            if v < black_tolerance:
                color_name = "black"
            elif v > white_tolerance and s < 30:
                color_name = "white"
            else:
                if (0 <= h <= 15) or (165 <= h <= 179):
                    color_name = "red"
                elif 15 < h <= 45:
                    color_name = "yellow"
                elif 45 < h <= 75:
                    color_name = "green"
                elif 75 < h <= 105:
                    color_name = "cyan"
                elif 105 < h <= 135:
                    color_name = "blue"
                elif 135 < h <= 165:
                    color_name = "magenta"
                else:
                    color_name = "unknown"

            color_counts[color_name] += 1

    return color_counts

def calculate_histograms(img):
    """
    Berechnet die Histogramme für die Farbkanäle Rot, Grün, Blau.
    """
    hist_r = cv.calcHist([img], [2], None, [256], [0, 256])  # Rot
    hist_g = cv.calcHist([img], [1], None, [256], [0, 256])  # Grün
    hist_b = cv.calcHist([img], [0], None, [256], [0, 256])  # Blau
    return hist_r, hist_g, hist_b

def get_ocr_text(img):
    """
    Führt OCR auf dem Bild durch und gibt den erkannten Text zurück.
    """
    reader = easyocr.Reader(['en','de'], gpu=False) # https://github.com/JaidedAI/EasyOCR/tree/master/easyocr/character / https://github.com/JaidedAI/EasyOCR/tree/master/easyocr/dict
    results = reader.readtext(img, detail=0)
    return ' '.join(results).strip()

def is_substring_in_string(ocr_text, test_string):
    """
    Prüft, ob ein gegebener Test-String im OCR-Ergebnis enthalten ist.
    """
    return test_string.lower() in ocr_text.lower()

def isIconInImage(ui_image, icon, visualize=False, debug=False, match_threshold=0.8):
    # Konvertiere das Icon (Template) in Graustufen und berechne den Canny-Kantenausgang durch die Methode der: Multi-Scale Template Matching siehe details: https://pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/
    template_gray = cv.cvtColor(icon, cv.COLOR_BGR2GRAY)
    template_edges = cv.Canny(template_gray, 50, 200)
    
    if visualize:
        cv.imshow("Icon - Kanten", template_edges)
        cv.waitKey(0)
    
    (tH, tW) = template_edges.shape[:2]
    
    # Konvertiere das UI-Bild in Graustufen
    gray = cv.cvtColor(ui_image, cv.COLOR_BGR2GRAY)
    
    best_match = None  # (maxVal, maxLoc, r, scale, resized)
    
    # Durchlaufe mehrere Skalierungen (von 120% bis 20% der Breite in 60 Schritten umso mehr Schritte umso genauer - aber dafür zeit und rechenaufwendiger)
    for scale in np.linspace(0.2, 1.2, 60)[::-1]:
        resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])
        
        # Abbruch, wenn das verkleinerte Bild kleiner als das Template ist
        if resized.shape[0] < tH or resized.shape[1] < tW:
            if debug:
                print(f"Skalierung {scale:.2f}: Bild zu klein (resized: {resized.shape[1]}x{resized.shape[0]}), Abbruch der Schleife.")
            break
        
        edged = cv.Canny(resized, 50, 200)
        result = cv.matchTemplate(edged, template_edges, cv.TM_CCOEFF_NORMED) #https://stackoverflow.com/questions/55469431/what-does-the-tm-ccorr-and-tm-ccoeff-in-opencv-mean or https://docs.opencv.org/3.4/de/da9/tutorial_template_matching.html
        (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)
        
        if debug:
            print(f"Skalierung {scale:.2f} | maxVal: {maxVal:.2f} | maxLoc: {maxLoc}")
        
        if best_match is None or maxVal > best_match[0]:
            best_match = (maxVal, maxLoc, r, scale, resized)
        
        if visualize:
            clone = np.dstack([edged, edged, edged])
            cv.rectangle(clone, maxLoc, (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
            cv.imshow("Visualisierung", clone)
            cv.waitKey(500)
    
    # Auswertung des besten Treffers
    if best_match is not None:
        (maxVal, maxLoc, r, best_scale, best_resized) = best_match
        # best_resized speichert das UI-Bild in der Skalierung, bei der der höchste Korrelationswert gefunden wurde - kann später nützlich sein
        if debug:
            print(f"\nBester Treffer bei Skalierung {best_scale:.2f} mit maxVal: {maxVal:.4f}")
        
        if maxVal >= match_threshold:
            # Optional: Bounding Box im Originalbild zeichnen
            (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
            (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
            if visualize:
                output = ui_image.copy()
                cv.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv.imshow("Match gefunden", output)
                cv.waitKey(0)
            # Rückgabe: Icon gefunden mit zugehörigem Korrelationswert
            return True, maxVal
        else:
            if debug:
                print("Übereinstimmung nicht ausreichend (maxVal unter dem Schwellenwert).")
            return False, maxVal
    else:
        if debug:
            print("Kein Treffer gefunden.")
        return False, None

# Beispiel-Hauptprogramm
if __name__ == "__main__":
    input_image_path = "./UIHD/settingslogi.jpg"
    template_image_path = "./UIHD/071.bmp"

    # 1. Bild einlesen
    input_image = read_image(input_image_path)

    # 2. Kalibrierung einmalig - Finde das größte, deutlichste Polygon mit 4 Kanten --> Vier/Rechteck
    calibrate_display(input_image)

    # 3. Beliebig viele Bilder zu kalibrierter Form gebracht --> Wird den Displaymaßen entsprechend skaliert, gezogen
    ui_display = rectify_display_image(input_image, target_size=(320, 240))

    # Farbanalyse
    pixel_stats = analyze_colors(ui_display)
    print("Pixelanzahl pro Farbe:")
    for color, count in pixel_stats.items():
        print(f"{color}: {count}")

    # Zeige das evtl. erkannte UI-Bild
    cv.imshow("Optimized UI-Display", ui_display)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # 4. OCR
    detected_text = get_ocr_text(ui_display)
    print("\nDetected Text:", detected_text)

    test_string = "alarm"
    is_string_present = is_substring_in_string(detected_text, test_string)
    print(f"\nIs '{test_string}' in OCR text? {is_string_present}")

    # 5. Bildvergleich
    template_image = read_image(template_image_path)
    found, probability = isIconInImage(ui_display, template_image, visualize=True, debug=True, match_threshold=0.5) # Wahrscheinlichkeit ab dem Bilder minimal noch als OK bewertet werden, bei manchen verschwommenen wird das benötigt, natürlich wird aber die höchste Wahrscheinlichekit gewertet
    print(f"Wurde Icon gefunden?: {found} (Wahrscheinlichkeitswert: {probability})")