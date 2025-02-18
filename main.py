# main.py
import cv2 as cv
import numpy as np
import easyocr  # https://github.com/jaidedai/easyocr
import sys

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

def optimize_image(img, target_size=(320, 240)):
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

def analyze_colors(image):
    """
    Analysiert die Anzahl der Pixel pro Farbe (B, G, R, Schwarz, Weiß).
    """
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    black_tolerance = 30
    white_tolerance = 220
    color_counts = {}

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

            color_counts[color_name] = color_counts.get(color_name, 0) + 1

    return color_counts

def calculate_histograms(image):
    """
    Berechnet die Histogramme für die Farbkanäle Rot, Grün, Blau.
    """
    hist_r = cv.calcHist([image], [2], None, [256], [0, 256])  # Rot
    hist_g = cv.calcHist([image], [1], None, [256], [0, 256])  # Grün
    hist_b = cv.calcHist([image], [0], None, [256], [0, 256])  # Blau
    return hist_r, hist_g, hist_b

def perform_ocr(img):
    """
    Führt OCR auf dem Bild durch und gibt den erkannten Text zurück.
    """
    reader = easyocr.Reader(['en','de'], gpu=False) # https://github.com/JaidedAI/EasyOCR/tree/master/easyocr/character / https://github.com/JaidedAI/EasyOCR/tree/master/easyocr/dict
    results = reader.readtext(img, detail=0)
    return ' '.join(results).strip()

def test_string_in_ocr(ocr_text, test_string):
    """
    Prüft, ob ein gegebener Test-String im OCR-Ergebnis enthalten ist.
    """
    return test_string.lower() in ocr_text.lower()

def check_image_presence(input_img, template_img, min_matches=5):
    """
    Prüft, ob ein gegebenes Bild Teil des Eingabebildes ist,
    indem es BRISK-Features miteinander vergleicht.
    Gibt True/False + Anzahl der Matches zurück.
    Zeigt außerdem das Matching-Bild an.
    """

    gray_input = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)
    gray_template = cv.cvtColor(template_img, cv.COLOR_BGR2GRAY)

    brisk = cv.BRISK_create()
    kp1, des1 = brisk.detectAndCompute(gray_input, None)
    kp2, des2 = brisk.detectAndCompute(gray_template, None)

    if des1 is None or des2 is None:
        return False, 0

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < min_matches:
        return False, len(matches)

    # Nur zur besseren Visualisierung: wir zeigen die Matches
    match_img = cv.drawMatches(
        gray_input, kp1,
        gray_template, kp2,
        matches, None,
        matchColor=(0, 255, 0),
        singlePointColor=(0, 0, 255),
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv.imshow("Feature Matches", match_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return True, len(matches)


# Beispiel-Hauptprogramm
if __name__ == "__main__":
    input_image_path = "./UIHD/doorerrorshrinked.bmp"
    template_image_path = "./UIHD/090.bmp"

    # 1. Bild einlesen
    input_image = read_image(input_image_path)

    # 2. Kalibrierung einmalig
    calibrate_display(input_image)

    # 3. Beliebig viele Bilder "optimieren": Hier testweise dasselbe Bild
    ui_display = optimize_image(input_image, target_size=(320, 240))

    # Farbanalyse
    pixel_stats = analyze_colors(ui_display)
    print("Pixelanzahl pro Farbe:")
    for color, count in pixel_stats.items():
        print(f"{color.capitalize()}: {count}")

    # Zeige das evtl. erkannte UI-Bild
    cv.imshow("Optimized UI-Display", ui_display)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # 4. OCR
    detected_text = perform_ocr(ui_display)
    print("\nDetected Text:", detected_text)

    test_string = "alarm"
    is_string_present = test_string_in_ocr(detected_text, test_string)
    print(f"\nIs '{test_string}' in OCR text? {is_string_present}")

    # 5. Bildvergleich
    template_image = read_image(template_image_path)
    is_present, num_matches = check_image_presence(ui_display, template_image)

    print("\nTemplate Found:", is_present)
    print("Number of Matches:", num_matches)
