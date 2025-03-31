import cv2 as cv
import sys
import numpy as np

# Lese das Bild input.png ein
# input_image = cv.imread("./BooksOnTable/IMG_20180103_151716_selection.jpg") # Dino Book
# input_image = cv.imread("./UI/9439808be83adceb.jpg")
input_image = cv.imread("./starry_night_test/weingarten.png") # Door Error
# input_image = cv.imread("./starry_night_test/starry_night_picture.jpg") # Starry Night Picture


# Lese das Bild IMG ein
# img = cv.imread("./BooksOnTable/IMG_20180103_151710.jpg") # Dino Book
# img = cv.imread("./UI/5eb9e7ba88e3f5e9.jpg")
img = cv.imread("./starry_night_test/test_gallery_weingarten.jpg") # Door Error
# img = cv.imread("./UI/9199efcdd2e196ff.jpg") # Door Error lowquality
# img = cv.imread("./starry_night_test/starry_night_gallery.jpg") # Starry Night Gallery

# Überprüfen, ob beide Bilder erfolgreich geladen wurden
if input_image is None or img is None:
    sys.exit("Could not read one or both images.")

# Zielgröße definieren
target_width = 800
target_height = 800
dim = (target_width, target_height)

# Skaliere beide Bilder auf die Zielgröße
resized_input = cv.resize(input_image, dim, interpolation=cv.INTER_AREA)
resized_img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

# Konvertiere beide Bilder in Graustufen
gray_input = cv.cvtColor(resized_input, cv.COLOR_BGR2GRAY)
gray_img = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)

# Wende Canny Edge Detection an
# edges_input = cv.Canny(gray_input, threshold1=50, threshold2=150)
# edges_img = cv.Canny(gray_img, threshold1=50, threshold2=150)

# Erstelle einen BRISK-Detector
brisk = cv.BRISK_create()

# Detektiere die Keypoints und Deskriptoren für beide Bilder
kp_input, des_input = brisk.detectAndCompute(gray_input, None)
kp_img, des_img = brisk.detectAndCompute(gray_img, None)

# Überprüfen, ob Deskriptoren erfolgreich berechnet wurden
if des_input is None or des_img is None:
    sys.exit("Feature detection failed for one or both images.")

# Erstelle einen Matcher (Brute-Force Matcher)
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Finde die besten Übereinstimmungen zwischen den Deskriptoren
matches = bf.match(des_input, des_img)

# Sortiere die Matches nach Entfernung (je kleiner, desto besser)
matches = sorted(matches, key=lambda x: x.distance)
print("matches:", len(matches))

# Visualisiere die Übereinstimmungen ohne RANSAC
output_img = cv.drawMatches(resized_input, kp_input, resized_img, kp_img, matches[:], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Ab hier mit RANSAC
# Finde die Punkte, die den Matches entsprechen
src_pts = np.float32([kp_input[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp_img[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Berechne die Homographie mit RANSAC
# RANSAC gibt die Homographie-Matrix und die "inlier" Übereinstimmungen zurück
M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

# Maskiere die schlechten Matches (outliers)
matches_mask = mask.ravel().tolist()
good_matches = [m for i, m in enumerate(matches) if matches_mask[i]]
print("best matches:", len(good_matches))

# Visualisiere die guten Übereinstimmungen (inlier) nach RANSAC
output_img = cv.drawMatches(resized_input, kp_input, resized_img, kp_img, good_matches, None, 
                            matchColor=(0, 255, 0), singlePointColor=(0, 0, 255), flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


# Zeige das Ergebnis
cv.imshow("Feature Matches", output_img)

# Wenn "s" gedrückt wird, speichere das Ergebnis
if cv.waitKey(0) == ord('s'):
    cv.imwrite("./save/matched_image_RANSAC_weingarten.png", output_img)

cv.waitKey(0)
cv.destroyAllWindows()