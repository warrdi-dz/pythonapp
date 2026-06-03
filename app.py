from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
import time
import traceback
import requests
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

YOLO_W = 900
YOLO_H = 500

# =========================
# YOLO API CALL
# =========================
def call_yolo(image_path):
    url = "https://warrdi.com/pytho/detect"
    try:
        with open(image_path, "rb") as f:
            files = {"image": f}
            r = requests.post(url, files=files, timeout=20)
        if r.status_code == 200:
            return r.json()
        return {"error": "YOLO failed", "status": r.status_code}
    except Exception as e:
        return {"error": "YOLO exception", "details": str(e)}

@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/")
def home():
    return jsonify({"status": "OK", "message": "GARAGE PRO V4 API"})


# =========================
# COULEUR HSV MÉDIANE
# =========================
def get_zone_color(hsv_img, mask, xA, yA, xB, yB):
    zone_mask = mask[yA:yB, xA:xB]
    zone_hsv  = hsv_img[yA:yB, xA:xB]
    valid     = zone_hsv[zone_mask > 0]
    if len(valid) < 80:
        return None, 0
    return np.array([
        float(np.median(valid[:, 0])),
        float(np.median(valid[:, 1])),
        float(np.median(valid[:, 2]))
    ]), len(valid)


# =========================
# AFFINER LE CROP
# =========================
def refine_car_bbox(img, x1, y1, x2, y2):
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return x1, y1, x2, y2

    hsv        = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask_dark  = cv2.inRange(hsv, (0, 0, 0),   (180, 255, 40))
    mask_sky   = cv2.inRange(hsv, (0, 0, 215), (180, 15, 255))
    mask_valid = cv2.bitwise_not(cv2.bitwise_or(mask_dark, mask_sky))
    k          = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_valid = cv2.morphologyEx(mask_valid, cv2.MORPH_CLOSE, k, iterations=2)

    col_sum = mask_valid.sum(axis=0).astype(float)
    row_sum = mask_valid.sum(axis=1).astype(float)
    col_sum = np.convolve(col_sum, np.ones(15) / 15, mode='same')
    row_sum = np.convolve(row_sum, np.ones(15) / 15, mode='same')

    col_thresh = col_sum.max() * 0.08
    row_thresh = row_sum.max() * 0.08
    valid_cols = np.where(col_sum > col_thresh)[0]
    valid_rows = np.where(row_sum > row_thresh)[0]

    if len(valid_cols) < 20 or len(valid_rows) < 20:
        return x1, y1, x2, y2

    PAD    = 8
    new_x1 = max(0,            x1 + int(valid_cols[0])  - PAD)
    new_x2 = min(img.shape[1], x1 + int(valid_cols[-1]) + PAD)
    new_y1 = max(0,            y1 + int(valid_rows[0])  - PAD)
    new_y2 = min(img.shape[0], y1 + int(valid_rows[-1]) + PAD)

    if (new_x2 - new_x1) < 100 or (new_y2 - new_y1) < 80:
        return x1, y1, x2, y2

    return new_x1, new_y1, new_x2, new_y2


# =============================================
# DÉTECTER QUELLE VUE EST VISIBLE
# Retourne le type de vue et l'orientation
#
# Vues possibles :
# - "side"      : vue de côté (profil complet)
# - "front"     : vue de face (avant)
# - "rear"      : vue de derrière (arrière)
# - "front_3q"  : 3/4 avant (angle avant-côté)
# - "rear_3q"   : 3/4 arrière (angle arrière-côté)
# =============================================
def detect_view_and_orientation(car_crop, detections):
    """
    Analyse la vue de la voiture pour définir
    quelles zones sont logiquement analysables.

    Méthode :
    1. Compte les feux rouges (arrière) et phares blancs (avant)
       sur TOUTE l'image (pas juste les bords)
    2. Regarde la répartition gauche/droite des éléments
    3. Mesure le ratio largeur/hauteur du crop
    4. Détecte la présence de côtés longs (vue profil)
    """
    crop_h, crop_w = car_crop.shape[:2]
    hsv = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)

    # --- Ratio largeur/hauteur ---
    ratio_wh = crop_w / max(crop_h, 1)

    # --- Compter feux rouges sur toute l'image ---
    mask_red1 = cv2.inRange(hsv, (0,   60, 60), (12,  255, 255))
    mask_red2 = cv2.inRange(hsv, (168, 60, 60), (180, 255, 255))
    mask_red  = cv2.bitwise_or(mask_red1, mask_red2)
    total_red = cv2.countNonZero(mask_red)

    # Répartition gauche/droite des feux rouges
    red_left  = cv2.countNonZero(mask_red[:, :crop_w//2])
    red_right = cv2.countNonZero(mask_red[:, crop_w//2:])

    # --- Compter phares blancs/jaunes sur toute l'image ---
    mask_wh1  = cv2.inRange(hsv, (0,  0,  180), (180, 70, 255))
    mask_wh2  = cv2.inRange(hsv, (15, 40, 180), (40,  200, 255))
    mask_white= cv2.bitwise_or(mask_wh1, mask_wh2)
    total_white = cv2.countNonZero(mask_white)

    white_left  = cv2.countNonZero(mask_white[:, :crop_w//2])
    white_right = cv2.countNonZero(mask_white[:, crop_w//2:])

    # --- Détecter vitres (zones sombres hautes) ---
    top_half   = gray[:crop_h//2, :]
    dark_top   = (top_half < 80).astype(np.uint8)
    dark_left  = dark_top[:, :crop_w//2].sum()
    dark_right = dark_top[:, crop_w//2:].sum()

    log = []

    # =============================================
    # RÈGLE 1 : Vue de côté (profil)
    # ratio > 1.4 ET vitres réparties des deux côtés
    # ET feux d'un seul côté
    # =============================================
    if ratio_wh > 1.4:
        # Vue latérale probable
        if total_red > 200:
            if red_left > red_right * 1.5:
                log.append(f"VUE COTE: feux rouge gauche → arriere GAUCHE")
                return "side", "right", log   # avant à droite
            elif red_right > red_left * 1.5:
                log.append(f"VUE COTE: feux rouge droite → arriere DROITE")
                return "side", "left", log    # avant à gauche
        if total_white > 200:
            if white_left > white_right * 1.5:
                log.append("VUE COTE: phares gauche → avant GAUCHE")
                return "side", "left", log
            elif white_right > white_left * 1.5:
                log.append("VUE COTE: phares droite → avant DROITE")
                return "side", "right", log

        # Pare-brise
        if dark_left > dark_right * 1.2:
            return "side", "left", log
        else:
            return "side", "right", log

    # =============================================
    # RÈGLE 2 : Vue de face
    # Phares blancs présents des deux côtés
    # ET peu ou pas de feux rouges
    # ET ratio proche de 1 (carré ou légèrement large)
    # =============================================
    white_balanced = (white_left > 100 and white_right > 100 and
                      max(white_left, white_right) < min(white_left, white_right) * 3)
    red_absent     = total_red < 150

    if white_balanced and red_absent and ratio_wh < 1.8:
        log.append("VUE AVANT: phares des deux côtés, pas de feux rouges")
        return "front", "front", log

    # =============================================
    # RÈGLE 3 : Vue de derrière
    # Feux rouges présents des deux côtés
    # ET peu ou pas de phares blancs
    # =============================================
    red_balanced  = (red_left > 100 and red_right > 100 and
                     max(red_left, red_right) < min(red_left, red_right) * 3)
    white_absent  = total_white < 150

    if red_balanced and white_absent and ratio_wh < 1.8:
        log.append("VUE ARRIERE: feux rouges des deux côtés, pas de phares")
        return "rear", "rear", log

    # =============================================
    # RÈGLE 4 : 3/4 avant
    # Phares visibles d'un côté + un peu de côté visible
    # ratio entre 0.9 et 1.6
    # =============================================
    if total_white > 150 and total_red < 100 and 0.8 < ratio_wh < 1.7:
        if white_left > white_right:
            log.append("VUE 3/4 AVANT: phares à gauche")
            return "front_3q", "left", log
        else:
            log.append("VUE 3/4 AVANT: phares à droite")
            return "front_3q", "right", log

    # =============================================
    # RÈGLE 5 : 3/4 arrière
    # Feux rouges visibles d'un côté
    # ratio entre 0.9 et 1.6
    # =============================================
    if total_red > 150 and total_white < 100 and 0.8 < ratio_wh < 1.7:
        if red_left > red_right:
            log.append("VUE 3/4 ARRIERE: feux à gauche")
            return "rear_3q", "left", log
        else:
            log.append("VUE 3/4 ARRIERE: feux à droite")
            return "rear_3q", "right", log

    # Fallback : vue de côté par défaut
    log.append("FALLBACK: vue cote par defaut")
    return "side", "left", log


# =============================================
# DÉFINIR LES ZONES SELON LA VUE DÉTECTÉE
# C'est le cœur de la logique intelligente
# =============================================
def define_zones(view_type, orientation, crop_h, crop_w):
    """
    Retourne la liste des zones à analyser
    adaptées à la vue détectée.

    Chaque zone = {name, xA, xB, yA, yB}
    Les coordonnées sont relatives au crop.
    """

    band_y1 = int(crop_h * 0.10)
    band_y2 = int(crop_h * 0.85)

    # -----------------------------------------------
    # VUE DE CÔTÉ COMPLÈTE
    # On voit : aile avant, portes, aile arrière
    # -----------------------------------------------
    if view_type == "side":
        cut1 = int(crop_w * 0.33)
        cut2 = int(crop_w * 0.67)

        if orientation == "left":
            return [
                {"name": "Aile avant",   "xA": 0,    "xB": cut1,   "yA": band_y1, "yB": band_y2},
                {"name": "Portes",       "xA": cut1, "xB": cut2,   "yA": band_y1, "yB": band_y2},
                {"name": "Aile arriere", "xA": cut2, "xB": crop_w, "yA": band_y1, "yB": band_y2},
            ]
        else:
            return [
                {"name": "Aile arriere", "xA": 0,    "xB": cut1,   "yA": band_y1, "yB": band_y2},
                {"name": "Portes",       "xA": cut1, "xB": cut2,   "yA": band_y1, "yB": band_y2},
                {"name": "Aile avant",   "xA": cut2, "xB": crop_w, "yA": band_y1, "yB": band_y2},
            ]

    # -----------------------------------------------
    # VUE DE FACE
    # On voit : aile gauche, capot/pare-chocs, aile droite
    # -----------------------------------------------
    elif view_type == "front":
        cut1 = int(crop_w * 0.25)
        cut2 = int(crop_w * 0.75)
        # Capot = bande haute, pare-chocs = bande basse
        capot_y2   = int(crop_h * 0.50)
        parechoc_y1= int(crop_h * 0.55)

        return [
            {"name": "Aile gauche",    "xA": 0,    "xB": cut1,   "yA": band_y1,    "yB": band_y2},
            {"name": "Capot",          "xA": cut1, "xB": cut2,   "yA": band_y1,    "yB": capot_y2},
            {"name": "Pare-chocs av",  "xA": cut1, "xB": cut2,   "yA": parechoc_y1,"yB": band_y2},
            {"name": "Aile droite",    "xA": cut2, "xB": crop_w, "yA": band_y1,    "yB": band_y2},
        ]

    # -----------------------------------------------
    # VUE DE DERRIÈRE
    # On voit : aile gauche, coffre/pare-chocs, aile droite
    # -----------------------------------------------
    elif view_type == "rear":
        cut1 = int(crop_w * 0.25)
        cut2 = int(crop_w * 0.75)
        coffre_y2    = int(crop_h * 0.50)
        parechoc_y1  = int(crop_h * 0.55)

        return [
            {"name": "Aile arr gauche",  "xA": 0,    "xB": cut1,   "yA": band_y1,    "yB": band_y2},
            {"name": "Coffre",           "xA": cut1, "xB": cut2,   "yA": band_y1,    "yB": coffre_y2},
            {"name": "Pare-chocs arr",   "xA": cut1, "xB": cut2,   "yA": parechoc_y1,"yB": band_y2},
            {"name": "Aile arr droite",  "xA": cut2, "xB": crop_w, "yA": band_y1,    "yB": band_y2},
        ]

    # -----------------------------------------------
    # VUE 3/4 AVANT
    # On voit : capot, aile avant, pare-chocs avant
    # Pas de portes ni arrière visibles
    # -----------------------------------------------
    elif view_type == "front_3q":
        cut1 = int(crop_w * 0.40)
        # Capot haut, aile/pare-chocs bas
        capot_y2    = int(crop_h * 0.45)
        parechoc_y1 = int(crop_h * 0.50)

        if orientation == "left":
            return [
                {"name": "Capot",         "xA": cut1, "xB": crop_w, "yA": band_y1,    "yB": capot_y2},
                {"name": "Aile avant",    "xA": 0,    "xB": cut1,   "yA": band_y1,    "yB": band_y2},
                {"name": "Pare-chocs av", "xA": cut1, "xB": crop_w, "yA": parechoc_y1,"yB": band_y2},
            ]
        else:
            return [
                {"name": "Capot",         "xA": 0,    "xB": cut1,   "yA": band_y1,    "yB": capot_y2},
                {"name": "Aile avant",    "xA": cut1, "xB": crop_w, "yA": band_y1,    "yB": band_y2},
                {"name": "Pare-chocs av", "xA": 0,    "xB": cut1,   "yA": parechoc_y1,"yB": band_y2},
            ]

    # -----------------------------------------------
    # VUE 3/4 ARRIÈRE
    # On voit : coffre, aile arrière, pare-chocs arrière
    # -----------------------------------------------
    elif view_type == "rear_3q":
        cut1 = int(crop_w * 0.40)
        coffre_y2   = int(crop_h * 0.45)
        parechoc_y1 = int(crop_h * 0.50)

        if orientation == "left":
            return [
                {"name": "Coffre",          "xA": cut1, "xB": crop_w, "yA": band_y1,    "yB": coffre_y2},
                {"name": "Aile arriere",    "xA": 0,    "xB": cut1,   "yA": band_y1,    "yB": band_y2},
                {"name": "Pare-chocs arr",  "xA": cut1, "xB": crop_w, "yA": parechoc_y1,"yB": band_y2},
            ]
        else:
            return [
                {"name": "Coffre",          "xA": 0,    "xB": cut1,   "yA": band_y1,    "yB": coffre_y2},
                {"name": "Aile arriere",    "xA": cut1, "xB": crop_w, "yA": band_y1,    "yB": band_y2},
                {"name": "Pare-chocs arr",  "xA": 0,    "xB": cut1,   "yA": parechoc_y1,"yB": band_y2},
            ]

    # Fallback
    cut1 = int(crop_w * 0.33)
    cut2 = int(crop_w * 0.67)
    return [
        {"name": "Zone gauche",  "xA": 0,    "xB": cut1,   "yA": band_y1, "yB": band_y2},
        {"name": "Zone centre",  "xA": cut1, "xB": cut2,   "yA": band_y1, "yB": band_y2},
        {"name": "Zone droite",  "xA": cut2, "xB": crop_w, "yA": band_y1, "yB": band_y2},
    ]


# =========================
# ANALYSE
# =========================
@app.route("/analyse", methods=["POST"])
def analyse():
    try:
        if "image" not in request.files:
            return jsonify({"error": "no image"}), 400

        file     = request.files["image"]
        filename = str(int(time.time())) + "_" + secure_filename(file.filename)
        path     = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        img_orig = cv2.imread(path)
        if img_orig is None:
            return jsonify({"error": "image unreadable"}), 400

        orig_h, orig_w = img_orig.shape[:2]

        img_yolo     = cv2.resize(img_orig, (YOLO_W, YOLO_H))
        resized_path = os.path.join(UPLOAD_FOLDER, "resized_" + filename)
        cv2.imwrite(resized_path, img_yolo)
        yolo_result  = call_yolo(resized_path)

        detections = yolo_result.get("detections", [])
        cars = [d for d in detections if d.get("class") == 2]
        if not cars:
            return jsonify({"error": "Car not detected"}), 400

        scale_x = orig_w / YOLO_W
        scale_y = orig_h / YOLO_H

        raw_x1 = int(min(d["box"][0] for d in cars) * scale_x)
        raw_y1 = int(min(d["box"][1] for d in cars) * scale_y)
        raw_x2 = int(max(d["box"][2] for d in cars) * scale_x)
        raw_y2 = int(max(d["box"][3] for d in cars) * scale_y)

        pad_x = int(15 * scale_x)
        pad_y = int(10 * scale_y)
        thr_x = int(150 * scale_x)
        thr_y = int(80  * scale_y)

        x1 = 0      if raw_x1 < thr_x            else max(0,      raw_x1 - pad_x)
        x2 = orig_w if (orig_w - raw_x2) < thr_x else min(orig_w, raw_x2 + pad_x)
        y1 = 0      if raw_y1 < thr_y            else max(0,      raw_y1 - pad_y)
        y2 = orig_h if (orig_h - raw_y2) < thr_y else min(orig_h, raw_y2 + pad_y)

        x1, y1, x2, y2 = refine_car_bbox(img_orig, x1, y1, x2, y2)

        car_crop = img_orig[y1:y2, x1:x2]
        if car_crop.size == 0:
            return jsonify({"error": "invalid crop"}), 400

        crop_h, crop_w = car_crop.shape[:2]

        # ===============================================
        # DÉTECTION DE LA VUE ET ORIENTATION
        # ===============================================
        view_type, orientation, view_log = detect_view_and_orientation(
            car_crop, detections
        )

        # ===============================================
        # ZONES ADAPTÉES À LA VUE DÉTECTÉE
        # ===============================================
        zones = define_zones(view_type, orientation, crop_h, crop_w)

        # ===============================================
        # MASQUE CARROSSERIE
        # ===============================================
        hsv_full  = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
        mask_dark = cv2.inRange(hsv_full, (0, 0, 0),   (180, 255, 45))
        mask_sky  = cv2.inRange(hsv_full, (0, 0, 210), (180, 18, 255))
        mask_body = cv2.bitwise_not(cv2.bitwise_or(mask_dark, mask_sky))
        kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_CLOSE, kernel)

        # ===============================================
        # RÉFÉRENCE GLOBALE
        # ===============================================
        all_valid = hsv_full[mask_body > 0]
        if len(all_valid) < 100:
            return jsonify({"error": "No body pixels found"}), 400

        ref_color = np.array([
            float(np.median(all_valid[:, 0])),
            float(np.median(all_valid[:, 1])),
            float(np.median(all_valid[:, 2]))
        ])

        # ===============================================
        # DESSIN
        # ===============================================
        final_img      = img_orig.copy()
        thick_box      = max(3, int(5 * min(scale_x, scale_y)))
        thick_line     = max(1, int(1 * min(scale_x, scale_y)))
        font_scale_big = max(0.6, 0.6 * min(scale_x, scale_y))
        font_scale_med = max(0.5, 0.5 * min(scale_x, scale_y))
        font_thick_big = max(2,   int(2 * min(scale_x, scale_y)))
        overlay_h      = max(55,  int(55 * scale_y))

        cv2.rectangle(final_img, (x1, y1), (x2, y2), (220, 220, 220), thick_line)

        # Séparateurs entre zones
        drawn_cuts = set()
        step = max(10, int(12 * scale_y))
        dash = max(4,  int(6  * scale_y))
        for zone in zones:
            for cut_x in [zone["xA"], zone["xB"]]:
                if cut_x in drawn_cuts or cut_x == 0 or cut_x == crop_w:
                    continue
                drawn_cuts.add(cut_x)
                yA_d = y1 + zone["yA"]
                yB_d = y1 + zone["yB"]
                for dy in range(yA_d, yB_d, step):
                    cv2.line(
                        final_img,
                        (x1 + cut_x, dy),
                        (x1 + cut_x, dy + dash),
                        (255, 255, 255), thick_line
                    )

        results_zones = []
        detected = 0

        for zone in zones:
            xA, xB = zone["xA"], zone["xB"]
            yA, yB = zone["yA"], zone["yB"]

            zone_color, px_count = get_zone_color(
                hsv_full, mask_body, xA, yA, xB, yB
            )

            abs_x1 = x1 + xA
            abs_y1 = y1 + yA
            abs_x2 = x1 + xB
            abs_y2 = y1 + yB

            if zone_color is None:
                color_rect  = (150, 150, 150)
                label_score = "N/A"
                diff        = 0.0
                verdict     = "Non analysable"
            else:
                diff = float(np.linalg.norm(zone_color - ref_color))

                if diff >= 14 and diff < 26:
                    color_rect = (0, 0, 255)
                    verdict    = "Attention peinture refaite!"
                    detected  += 1
                elif diff < 14:
                    color_rect = (0, 165, 255)
                    verdict    = "Legere variation suspecte!"
                    detected  += 1
                else:
                    color_rect = (0, 210, 0)
                    verdict    = "OK"

                label_score = str(int(diff))

            cv2.rectangle(final_img, (abs_x1, abs_y1),
                          (abs_x2, abs_y2), color_rect, thick_box)

            overlay = final_img.copy()
            cv2.rectangle(overlay, (abs_x1, abs_y1),
                          (abs_x2, abs_y1 + overlay_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, final_img, 0.5, 0, final_img)

            cv2.putText(final_img, zone["name"],
                        (abs_x1 + 8, abs_y1 + int(overlay_h * 0.40)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale_big, (255, 255, 255), font_thick_big)

            cv2.putText(final_img, f"Ecart: {label_score}",
                        (abs_x1 + 8, abs_y1 + int(overlay_h * 0.80)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale_med, color_rect, font_thick_big)

            cv2.putText(final_img, verdict,
                        (abs_x1 + 8, abs_y2 - int(10 * scale_y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale_med, color_rect, font_thick_big)

            results_zones.append({
                "zone":    zone["name"],
                "diff":    round(diff, 1),
                "pixels":  px_count,
                "verdict": verdict
            })

        # ===============================================
        # SCORE GLOBAL
        # ===============================================
        diffs = [z["diff"] for z in results_zones if z["diff"] > 0]
        final_score = int(np.mean(diffs)) if diffs else 0
        final_score = min(final_score, 100)

        if final_score < 10:
            result = "Peinture homogene (OK)"
        elif final_score < 28:
            result = "Legeres variations detectees"
        else:
            result = "Difference importante — repeinture probable"

        analysed_name = "analysed_" + filename
        analysed_path = os.path.join(UPLOAD_FOLDER, analysed_name)
        cv2.imwrite(analysed_path, final_img)

        if os.path.exists(resized_path):
            os.remove(resized_path)

        return jsonify({
            "yolo":            yolo_result,
            "score":           final_score,
            "result":          result,
            "zones":           results_zones,
            "zones_detected":  detected,
            "view_type":       view_type,
            "orientation":     orientation,
            "view_log":        view_log,
            "image_size":      {"width": orig_w, "height": orig_h},
            "reference_hsv": {
                "H": round(ref_color[0], 1),
                "S": round(ref_color[1], 1),
                "V": round(ref_color[2], 1)
            },
            "image_result":    analysed_name,
            "image_url":       request.host_url + "uploads/" + analysed_name
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
