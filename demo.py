from ui_detection import read_image, calibrate_display, rectify_display_image, analyze_colors, get_ocr_text, isIconInImage, calculate_histograms

# 1. Bild einlesen
input_image_path = "./UIHD/settingslogi.jpg"
input_image = read_image(input_image_path)

# 2. Kalibrierung einmalig - Finde das größte, deutlichste Polygon mit 4 Kanten --> Vier/Rechteck
# Einmalig pro Run aufrufen wenn man weiß, dass das UI mit nichtschwarzen Pixeln bereit ist um Ecken zu finden
cornors_detected = calibrate_display(input_image) # Rückgabewert mal mit rein genommen falls man ihn nicht braucht weglassen
print("Calibration successfull and cornors detected: ")
print(cornors_detected)

# 3. Beliebig viele Bilder zu kalibrierter Form gebracht --> Wird den Displaymaßen entsprechend skaliert, gezogen - für zukünftige UIs einfach die Pixelwerte ändern
# Muss für jedes Bild aufgerufen werden für den jeweiligen Run da Kalibrierung angewendet
ui_display = rectify_display_image(input_image, target_size=(320, 240), visualize=False)

# 4. Farbanalyse
pixel_stats = analyze_colors(ui_display)
print("Farben erkannt: ")
print(pixel_stats)

# 5. OCR Wichtig: in Benutzer/Username/.EasyOCR befindet sich nach erstmaligen Ausführen mit Internet das neuste benötigte OCR-KI Model. Falls Endgerät diese nicht hat einfach den Ordner mit neustem Model selber hinzufügen dann funktioniert es offline.
detected_text = get_ocr_text(ui_display)
print("Erkannter Text: ")
print(detected_text)

# 6. Bildvergleich
template_image_path = "./UIHD/071.bmp"
template_image = read_image(template_image_path)

found, normed_Max_val = isIconInImage(ui_display, template_image, visualize=False, debug=False, match_threshold=0.3) # Wahrscheinlichkeit ab dem Bilder minimal noch als OK bewertet werden, bei manchen verschwommenen wird das benötigt, natürlich wird aber die höchste Wahrscheinlichekit gewertet
print("Ist Icon in Image?: ")
print(found)

# # Histogram ist drin und voll funktionsfähig wer will kann folgendes aufrufen:
# # (R)ed, (G)reen, (B)lue
# hist_r, hist_g, hist_b = calculate_histograms(ui_display)
# # Print the histogram values
# print("Red channel histogram:")
# print(hist_r)
# print("\nGreen channel histogram:")
# print(hist_g)
# print("\nBlue channel histogram:")
# print(hist_b)