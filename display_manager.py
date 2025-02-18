# display_config.py
import json
import os
import numpy as np

CONFIG_FILE = "display_config.json"

def save_display_corners(corners, use_full_image=False):
    """
    Speichert die Display-Koordinaten als JSON-Datei.
    Falls keine UI erkannt wurde, speichert es ein Flag (use_full_image=True).
    """
    data = {
        "corners": [list(map(float, point)) for point in corners] if corners is not None else [],
        "use_full_image": use_full_image
    }
    with open(CONFIG_FILE, "w") as file:
        json.dump(data, file)

def load_display_corners():
    """
    Lädt die gespeicherten Display-Koordinaten.
    Gibt (corners, use_full_image) zurück.
    Falls keine Datei existiert oder fehlerhafte Daten vorliegen, gibt es use_full_image=True zurück.
    """
    if not os.path.exists(CONFIG_FILE):
        return None, True  # Keine Datei -> gesamtes Bild verwenden
    try:
        with open(CONFIG_FILE, "r") as file:
            data = json.load(file)
        corners_data = data.get("corners", [])
        use_full_image = data.get("use_full_image", False)

        if not corners_data:
            # Falls corners leer sind
            return None, use_full_image

        corners = np.array(corners_data, dtype=np.float32)
        return corners, use_full_image
    except (json.JSONDecodeError, KeyError, ValueError):
        # Falls die Datei korrupt ist oder Ecken fehlen
        return None, True