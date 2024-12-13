import cv2 as cv
import numpy as np
import easyocr
import sys

# Globale Konstanten
UI_TARGET_WIDTH = 800
UI_TARGET_HEIGHT = 800

def read_and_resize_image(path, target_dim=(UI_TARGET_WIDTH, UI_TARGET_HEIGHT)):
    """
    Liest ein Bild ein und skaliert es auf die Zielgröße.
    """
    img = cv.imread(path)
    if img is None:
        sys.exit(f"Could not read the image at {path}.")
    resized_img = cv.resize(img, target_dim, interpolation=cv.INTER_AREA)
    return resized_img

def detect_display_area(img):
    """
    Ermittelt das viereckige UI im Bild.
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150)
    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        epsilon = 0.02 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:  # Suche nach Vierecken
            x, y, w, h = cv.boundingRect(approx)
            if w > 50 and h > 50:  # Filter für Größe
                return img[y:y+h, x:x+w]  # Ausschneiden des UI
    return None

def analyze_colors(image):
    """
    Analysiert die Anzahl der Pixel pro Farbe (B, G, R, Schwarz, Weiß).
    """
    pixel_count = {
        "blue": 0,
        "green": 0,
        "red": 0,
        "black": 0,
        "white": 0
    }

    black_tolerance=30
    white_tolerance=30
    color_counts = {}

    for row in image:  # Iteriere durch alle Pixelzeilen
        for pixel in row:  # Iteriere durch alle Pixel in der Zeile
            b, g, r = pixel  # Extrahiere die BGR-Werte

            # Nahe Schwarz
            if b < black_tolerance and g < black_tolerance and r < black_tolerance:
                color_name = "black"
            # Nahe Weiß
            elif b > 255 - white_tolerance and g > 255 - white_tolerance and r > 255 - white_tolerance:
                color_name = "white"
            else:
                # Bestimme die dominante Farbe
                if r > b and r > g:
                    color_name = "red"
                elif g > b and g > r:
                    color_name = "green"
                elif b > r and b > g:
                    color_name = "blue"
                elif r > b and g > b:  # Gelb (Rot + Grün)
                    color_name = "yellow"
                elif r > g and b > g:  # Magenta (Rot + Blau)
                    color_name = "magenta"
                elif g > r and b > r:  # Cyan (Blau + Grün)
                    color_name = "cyan"
                else:
                    color_name = "other"  # Falls eine Farbe nicht eindeutig zugeordnet werden kann

            # Zähle die Farbe
            if color_name not in color_counts:
                color_counts[color_name] = 0
            color_counts[color_name] += 1

    return color_counts

def calculate_histograms(image):
    """
    Berechnet die Histogramme für die Farbkanäle B, G, R.
    """
    hist_b = cv.calcHist([image], [0], None, [256], [0, 256])
    hist_g = cv.calcHist([image], [1], None, [256], [0, 256])
    hist_r = cv.calcHist([image], [2], None, [256], [0, 256])

    return hist_b, hist_g, hist_r

def perform_ocr(img):
    """
    Führt OCR auf dem Bild durch und gibt den erkannten Text zurück.
    """
    reader = easyocr.Reader(['en'], gpu=False)  # CPU oder GPU --> wäre schneller aber Cuda/MPS erforderlich
    results = reader.readtext(img, detail=0)  # Detail=0 gibt nur den erkannten Text zurück
    return ' '.join(results).strip()  # Kombiniert die Texte in einen String

def test_string_in_ocr(ocr_text, test_string):
    """
    Prüft, ob ein gegebener Test-String im OCR-Ergebnis enthalten ist.
    """
    return test_string.lower() in ocr_text.lower()

def check_image_presence(input_img, template_img, min_matches=10):
    """
    Prüft, ob ein gegebenes Bild ein Teil des Eingabebildes ist und gibt die Position zurück.
    """
    gray_input = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)
    gray_template = cv.cvtColor(template_img, cv.COLOR_BGR2GRAY)
    
    # BRISK Feature Detection
    brisk = cv.BRISK_create()
    kp1, des1 = brisk.detectAndCompute(gray_input, None)
    kp2, des2 = brisk.detectAndCompute(gray_template, None)
    
    if des1 is None or des2 is None:
        return False, None

    # Matcher und Homographie
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < min_matches:
        return False, None

    # Berechnung der Homographie mit RANSAC
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    """
    Die Homographie ist eine Transformation, die beschreibt, wie ein Bild in ein anderes überführt werden kann.
    Sie wird als Matrix dargestellt und kann verwendet werden, um Punkte von einem Bild in das andere zu
    projizieren. Das ist nützlich, um herauszufinden, ob ein Teilbild (wie eine Vorlage)
    im größeren Bild enthalten ist und wo es sich genau befindet.
    """
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    if M is None:
        return False, None

    # Gibt die Transformationsmatrix und Position zurück
    return True, M

# Hauptprogramm
if __name__ == "__main__":
    input_image_path = "./UIHD/doorerror.bmp"
    template_image_path = "./UIHD/doorerrorshrinked.bmp"
    
    # 1. Bild einlesen und skalieren
    input_image = read_and_resize_image(input_image_path)
    
    # 2. UI-Display erkennen
    ui_display = detect_display_area(input_image)
    if ui_display is None:
        sys.exit("UI Display not detected.")
    
    # Analysiere Farben
    pixel_stats = analyze_colors(ui_display)
    hist_b, hist_g, hist_r = calculate_histograms(ui_display)


    # Ergebnisse anzeigen
    print("Pixelanzahl pro Farbe:")
    for color, count in pixel_stats.items():
        print(f"{color.capitalize()}: {count}")

    # Zeige das Bild (optional)
    cv.imshow("Image", ui_display)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # # Optional: Histogramm anzeigen
    # print("\nHistogramme der Farbkanäle:")
    # print("Intensität | Blau | Grün | Rot")
    # for i in range(256):
    #     print(f"{i:10} | {int(hist_b[i][0]):4} | {int(hist_g[i][0]):4} | {int(hist_r[i][0]):4}")

    # 4. OCR ausführen
    detected_text = perform_ocr(ui_display)
    print("\nDetected Text:", detected_text)
    
    # Prüfen, ob ein Test-String im OCR-Ergebnis vorhanden ist
    test_string = "alarm"
    is_string_present = test_string_in_ocr(detected_text, test_string)
    print(f"\nIs '{test_string}' in OCR text? {is_string_present}")
    
    # 5. Bildvergleich
    template_image = read_and_resize_image(template_image_path)
    is_present, homography = check_image_presence(input_image, template_image)
    print("\nTemplate Found:", is_present)
    if is_present:
        print("Homography Matrix:", homography)