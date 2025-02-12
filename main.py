import cv2 as cv
import numpy as np
import easyocr # https://github.com/jaidedai/easyocr
import sys

def read_image(path):
    """Liest ein Bild ein, ohne es zu skalieren."""
    img = cv.imread(path)
    if img is None:
        sys.exit(f"Could not read the image at {path}.")
    return img

def detect_display_area(img, target_size):
    """
    Ermittelt das viereckige UI im Bild und skaliert es auf die tatsächliche Display-Größe.
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
                display_img = img[y:y+h, x:x+w]
                # **Hier sicherstellen, dass es proportional auf 320x240 skaliert wird**
                display_img = cv.resize(display_img, target_size, interpolation=cv.INTER_AREA)
                return display_img
    return None

def analyze_colors(image):
    """
    Analysiert die Anzahl der Pixel pro Farbe (B, G, R, Schwarz, Weiß).
    """
    # Bild in HSV umwandeln
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    black_tolerance=30
    white_tolerance = 220
    color_counts = {}

    for row in hsv_image:  # Iteriere durch alle Pixelzeilen
        for pixel in row:  # Iteriere durch alle Pixel in der Zeile
            h, s, v = pixel  # Hue, Saturation, Value

            # Schwarz-Erkennung (niedrige Helligkeit)
            if v < black_tolerance:
                color_name = "black"
            # Weiß-Erkennung (hohe Helligkeit + niedrige Sättigung)
            elif v > white_tolerance and s < 30:
                color_name = "white"
            else:
                # **Farberkennung über den Hue-Wert**
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

            # Zähle die Farbe -> nur wenn auch gegeben
            if color_name not in color_counts:
                color_counts[color_name] = 0
            color_counts[color_name] += 1

    return color_counts

def calculate_histograms(image):
    """
    Berechnet die Histogramme für die Farbkanäle Rot, Grün, Blau -> RGB ist hier besser für Histogramme, da es die Pixelintensitäten direkt zeigt, anstatt Farbtonänderungen (Hue).

    Ein Histogramm zeigt, wie viele Pixel einen bestimmten Farbwert haben (0-255).
    Es hilft zu erkennen:
    - Welche Farben dominant sind.
    - Ob das Bild eher dunkel oder hell ist.
    - Wie sich Farben über das gesamte Bild verteilen.
    """
    hist_r = cv.calcHist([image], [2], None, [256], [0, 256]) # Rot
    hist_g = cv.calcHist([image], [1], None, [256], [0, 256]) # Grün
    hist_b = cv.calcHist([image], [0], None, [256], [0, 256]) # Blau

    return hist_r, hist_g, hist_b,  

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

def check_image_presence(input_img, template_img, min_matches=5):
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
        return False, 0, None

    # Matcher und Homographie
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < min_matches:
        return False, len(matches), None

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
        return False, len(matches), None
    
    # Visualisierung der Matches
    matches_mask = mask.ravel().tolist()
    good_matches = [m for i, m in enumerate(matches) if matches_mask[i]]

    match_img = cv.drawMatches(gray_input, kp1, gray_template, kp2, good_matches, None, 
                               matchColor=(0, 255, 0), singlePointColor=(0, 0, 255), 
                               flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Zeige das Bild mit den Matches
    cv.imshow("Feature Matches", match_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Gibt die Transformationsmatrix und Position zurück
    return True, len(matches), M

# Hauptprogramm
if __name__ == "__main__":
    input_image_path = "./UIHD/Test.png"
    template_image_path = "./UIHD/090.bmp"
    
    # 1. Bild einlesen
    input_image = read_image(input_image_path)
    
    # 2. UI-Display erkennen und auf echte Größe skalieren aktuell: (320x240)
    ui_display = detect_display_area(input_image, target_size=(320, 240))
    if ui_display is None:
        sys.exit("UI Display not detected.")
    
    # Analysiere Farben
    pixel_stats = analyze_colors(ui_display)
    
    # Histogramme berechnen
    # hist_r, hist_g, hist_b  = calculate_histograms(ui_display)

    # Ergebnisse anzeigen
    print("Pixelanzahl pro Farbe:")
    for color, count in pixel_stats.items():
        print(f"{color.capitalize()}: {count}")

    # Zeige das erkannte UI-Bild
    cv.imshow("Image", ui_display)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # # Optional: Histogramm anzeigen
    # print("\nHistogramme der Farbkanäle:")
    # print("Intensität | Rot | Grün | Blau")
    # for i in range(256):
    #     print(f"{i:10} | {int(hist_r[i][0]):4} | {int(hist_g[i][0]):4} | {int(hist_b[i][0]):4}")

    # 4. OCR ausführen
    detected_text = perform_ocr(ui_display)
    print("\nDetected Text:", detected_text)
    
    # Prüfen, ob ein Test-String im OCR-Ergebnis vorhanden ist
    test_string = "alarm"
    is_string_present = test_string_in_ocr(detected_text, test_string)
    print(f"\nIs '{test_string}' in OCR text? {is_string_present}")
    
    # 5. Bildvergleich
    template_image = read_image(template_image_path)
    is_present, num_matches, homography = check_image_presence(ui_display, template_image)
    
    print("\nTemplate Found:", is_present)
    print("Number of Matches:", num_matches)
    if is_present:
        print("Homography Matrix:", homography)