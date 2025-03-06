# main.py
import cv2 as cv
import numpy as np
import easyocr  # https://github.com/jaidedai/easyocr
import sys
import imutils

# Import der JSON-Funktionen aus dem neuen Skript
from display_manager import save_display_corners, load_display_corners


def read_image(path):
    """
    Reads an image from the specified file path without scaling it.
    
    Parameters:
        path (str): The file path to the image.
    
    Returns:
        img (numpy.ndarray): The image loaded into a NumPy array.
        
    Raises:
        Exits the program if the image cannot be read.
    """
    img = cv.imread(path)
    if img is None:
        sys.exit(f"Could not read the image at {path}.")
    return img

def detect_display_corners(img, visualize=False, canny_thresh1=50, canny_thresh2=150, approx_factor=0.007):
    """
    Detects the corners of a display (assumed to be a quadrilateral) in the input image.
    It converts the image to grayscale, performs edge detection, dilates the edges, 
    finds contours, approximates polygonal curves, and selects the best quadrilateral based on area.
    
    Parameters:
        img (numpy.ndarray): The input image in BGR format.
        visualize (bool): If True, shows the processed edge image (default is False).
        canny_thresh1 (int): The first threshold for the Canny edge detector.
        canny_thresh2 (int): The second threshold for the Canny edge detector.
        approx_factor (float): The factor used in polygonal approximation (epsilon = approx_factor * perimeter).
    
    Returns:
        corners (numpy.ndarray or None): A NumPy array of shape (4, 2) representing the four sorted corner points 
                                         (ordered as top-left, top-right, bottom-right, bottom-left) if detected,
                                         or None if no suitable quadrilateral is found.
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, canny_thresh1, canny_thresh2)

    # Kanten "verbreitern"
    kernel = np.ones((2, 2), np.uint8)
    edges = cv.dilate(edges, kernel, iterations=1)

    if visualize:
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
            # Ignoriere zu kleine Vierecke
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
    Applies a perspective transformation to the input image based on the provided and saved corner points.
    
    Parameters:
        img (numpy.ndarray): The input image.
        corners (numpy.ndarray): A NumPy array of shape (4, 2) containing the corner points of the region of interest.
        target_size (tuple): A tuple (width, height) specifying the size of the output (warped) image.
    
    Returns:
        warped (numpy.ndarray): The transformed image warped to the specified target size.
    """
    dst_pts = np.array([
        [0, 0],                             # Top Left
        [target_size[0], 0],                # Top Right
        [target_size[0], target_size[1]],   # Bottom Right
        [0, target_size[1]]                 # Bottom Left
    ], dtype=np.float32)

    M = cv.getPerspectiveTransform(corners, dst_pts)
    warped = cv.warpPerspective(img, M, target_size)
    return warped

def calibrate_display(img):
    """
    Detects the display corners in the given image and saves the calibration data (corners and a flag)
    into a JSON file. If no display is detected, it marks the calibration to use the full image.
    
    Parameters:
        img (numpy.ndarray): The input image for calibration.
    
    Returns:
        bool: Display cornors detected --> True // False
    """
    corners = detect_display_corners(img, visualize=False)
    if corners is not None:
        save_display_corners(corners, use_full_image=False)
        # Kalibrierung erfolgreich. Display-Ecken erkannt und gespeichert.
        return True
    else:
        save_display_corners(None, use_full_image=True)
        # Keine Display-Ecken erkannt. Verwende Originalbild.
        return False

def rectify_display_image(img, target_size=(320, 240), visualize=False):
    """
    Rectifies the input image using previously saved calibration data. It loads the display corners and a flag
    from JSON and applies a perspective transformation if valid corner data is available.
    
    Parameters:
        img (numpy.ndarray): The input image to be rectified.
        target_size (tuple): A tuple (width, height) for the desired output size (default is (320, 240)).
        visualize (bool): If True, displays will shown with applied cornor position or without (default is False).
            
    Returns:
        rectified_img (numpy.ndarray): The transformed image if calibration data is valid; otherwise, returns the original image.
    """
    corners, use_full_image = load_display_corners()
    # Prüfen, ob use_full_image oder corners fehlerhaft
    if use_full_image or corners is None or len(corners) != 4:
        if visualize:
            # Zeige das evtl. erkannte UI-Bild (Tesing)
            cv.imshow("Keine Kanten erkannt", img)
            cv.waitKey(0)
            cv.destroyAllWindows()
        return img
    # Ansonsten Transformation anwenden
    applied_perspective_transform = apply_perspective_transform(img, corners, target_size)
    if visualize:
            # Zeige das evtl. erkannte UI-Bild (Tesing)
            cv.imshow("Kanten erkannt", applied_perspective_transform)
            cv.waitKey(0)
            cv.destroyAllWindows()
    return applied_perspective_transform

def analyze_colors(img):
    """
    Analyzes the input image by converting it to the HSV color space and counting the number of pixels for each color category:
    black, white, red, yellow, green, cyan, blue, magenta, and unknown.
    
    Parameters:
        img (numpy.ndarray): The input image in BGR format.
    
    Returns:
        color_counts (dict): A dictionary where the keys are color names and the values are the pixel counts for each color.
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
    Calculates histograms for the red, green, and blue color channels of the input image.
    
    Parameters:
        img (numpy.ndarray): The input image in BGR format.
    
    Returns:
        tuple: A tuple containing three histograms (hist_r, hist_g, hist_b) corresponding to the red, green, and blue channels.
    """
    hist_r = cv.calcHist([img], [2], None, [256], [0, 256])  # Rot
    hist_g = cv.calcHist([img], [1], None, [256], [0, 256])  # Grün
    hist_b = cv.calcHist([img], [0], None, [256], [0, 256])  # Blau
    return hist_r, hist_g, hist_b

def get_ocr_text(img):
    """
    Performs Optical Character Recognition (OCR) on the input image using EasyOCR and returns the extracted text.
    
    Parameters:
        img (numpy.ndarray): The input image on which OCR is to be performed.
    
    Returns:
        str: A string containing the concatenated OCR result with extra whitespace removed.
    """
    reader = easyocr.Reader(['en','de'], gpu=False) # https://github.com/JaidedAI/EasyOCR/tree/master/easyocr/character / https://github.com/JaidedAI/EasyOCR/tree/master/easyocr/dict
    results = reader.readtext(img, detail=0)
    return ' '.join(results).strip()

def is_substring_in_string(ocr_text, test_string):
    """
    Checks whether the specified test string is present within the provided OCR text, ignoring case differences.
    
    Parameters:
        ocr_text (str): The text obtained from OCR.
        test_string (str): The substring to search for within the OCR text.
    
    Returns:
        bool: True if the test string is found within the OCR text (case-insensitive), otherwise False.
    """
    return test_string.lower() in ocr_text.lower()

def isIconInImage(ui_image, icon, visualize=False, debug=False, match_threshold=0.3): # Für unseren Fall hat sich 0.3 auch bei sehr pixligen Bildern gut bewährt.
    """
    Performs multi-scale template matching to determine if the icon (template) is present within the UI image.
    It converts both images to grayscale, computes Canny edge maps, and iterates over multiple scales of the UI image 
    to find the best match based on the normalized correlation coefficient. More instructions on: https://pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/
    
    Parameters:
        ui_image (numpy.ndarray): The input UI image in which to search for the icon.
        icon (numpy.ndarray): The icon (template) image to be matched.
        visualize (bool): If True, displays intermediate visualization of the matching process (default is False).
        debug (bool): If True, prints debug information about each scale's matching result (default is False).
        match_threshold (float): The minimum normalized correlation value (between 0.0 and 1.0) required to consider the icon as found.
    
    Returns:
        tuple: A tuple (found, maxVal) where:
            - found (bool): True if a match with a correlation value equal to or above the match_threshold is found, otherwise False.
            - maxVal (float or None): The highest normalized correlation value found. Returns None if no match is found.
    """
    template_gray = cv.cvtColor(icon, cv.COLOR_BGR2GRAY)
    template_edges = cv.Canny(template_gray, 70, 200)
    
    if visualize:
        cv.imshow("Icon - Kanten", template_edges)
        cv.waitKey(0)
    
    (tH, tW) = template_edges.shape[:2]
    
    # Konvertiere das UI-Bild in Graustufen
    gray = cv.cvtColor(ui_image, cv.COLOR_BGR2GRAY)
    
    best_match = None  # (maxVal, maxLoc, r, scale, resized)
    
    # Durchlaufe mehrere Skalierungen (von 100% bis 20% der Breite in 60 Schritten umso mehr Schritte umso genauer - aber dafür zeit und rechenaufwendiger)
    for scale in np.linspace(0.2, 1.5, 50)[::-1]:
        resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])
        
        # Abbruch, wenn das verkleinerte Bild kleiner als das Template ist
        if resized.shape[0] < tH or resized.shape[1] < tW:
            if debug:
                print(f"Skalierung {scale:.2f}: Bild zu klein (resized: {resized.shape[1]}x{resized.shape[0]}), Abbruch der Schleife.")
            break
        
        edged = cv.Canny(resized, 50, 200)
        result = cv.matchTemplate(edged, template_edges, cv.TM_CCOEFF_NORMED) # https://stackoverflow.com/questions/55469431/what-does-the-tm-ccorr-and-tm-ccoeff-in-opencv-mean or https://docs.opencv.org/3.4/de/da9/tutorial_template_matching.html Ist nicht die Prozentuale Chance sondern wie hoch der Koeffizient ist also wie gut das Bild zum Template passt --> hoher Wert ist eher wahrscheinlich
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
    input_image_path = "./UIHD/6f076908f37418eb.jpg"
    template_image_path = "./UIHD/107.bmp"

    # 1. Bild einlesen
    input_image = read_image(input_image_path)

    # 2. Kalibrierung einmalig - Finde das größte, deutlichste Polygon mit 4 Kanten --> Vier/Rechteck
    cornors_detected = calibrate_display(input_image)
    print("Calibration successfull and cornors detected: ")
    print(cornors_detected)

    # 3. Beliebig viele Bilder zu kalibrierter Form gebracht --> Wird den Displaymaßen entsprechend skaliert, gezogen
    ui_display = rectify_display_image(input_image, target_size=(320, 240))

    # 4. Farbanalyse
    pixel_stats = analyze_colors(ui_display)
    print("Pixelanzahl pro Farbe:")
    for color, count in pixel_stats.items():
        print(f"{color}: {count}")

    # 5. OCR
    detected_text = get_ocr_text(ui_display)
    print("\nDetected Text:", detected_text)

    test_string = "alarm"
    is_string_present = is_substring_in_string(detected_text, test_string)
    print(f"\nIs '{test_string}' in OCR text? {is_string_present}")

    # 6. Bildvergleich
    template_image = read_image(template_image_path)
    found, normed_Max_val = isIconInImage(ui_display, template_image, visualize=True, debug=True, match_threshold=0.3) # Wahrscheinlichkeit ab dem Bilder minimal noch als OK bewertet werden, bei manchen verschwommenen wird das benötigt, natürlich wird aber die höchste Wahrscheinlichekit gewertet
    print(f"Wurde Icon gefunden?: {found} (Höchster Wert: {normed_Max_val})")

    # # 7. Histogram if needed (R)ed, (G)reen, (B)lue
    # hist_r, hist_g, hist_b = calculate_histograms(ui_display)
    # print("Red channel histogram:")
    # print(hist_r)
    # print("\nGreen channel histogram:")
    # print(hist_g)
    # print("\nBlue channel histogram:")
    # print(hist_b)