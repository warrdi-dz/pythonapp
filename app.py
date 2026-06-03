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


# =============================================
# MASQUE CARROSSERIE STRICT
# Exclure : vitres, roues, reflets, ombres,
# taches noires, plastique, chrome, ciel
# =============================================
def build_body_mask(car_crop, hsv):
    # Trop sombre = vitres, pneus, taches noires
    mask_dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, 40))

    # Reflets et surexposition = soleil, chrome brillant
    mask_reflet = cv2.inRange(hsv, (0, 0, 220), (180, 255, 255))

    # Ciel / fond blanc désaturé
    mask_sky = cv2.inRange(hsv, (0, 0, 210), (180, 20, 255))

    # Faible saturation = plastique, chrome mat, calandre
    mask_chrome = cv2.inRange(hsv, (0, 0, 0), (180, 30, 255))

    # Ombres très sombres (V < 35)
    mask_shadow = cv2.inRange(hsv, (0, 0, 0), (180, 255, 35))

    # Combiner toutes les exclusions
    exclude   = cv2.bitwise_or(mask_dark,   mask_reflet)
    exclude   = cv2.bitwise_or(exclude,     mask_sky)
    exclude   = cv2.bitwise_or(exclude,     mask_chrome)
    exclude   = cv2.bitwise_or(exclude,     mask_shadow)
    mask_body = cv2.bitwise_not(exclude)

    # Morphologie pour nettoyer
    k         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_CLOSE, k, iterations=2)
    mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_OPEN,  k, iterations=1)

    return mask_body


# =============================================
# COULEUR LAB MÉDIANE (anti-reflets, anti-ombre)
# On utilise LAB et on exclut les pixels extrêmes
# =============================================
def get_zone_color(lab_img, mask, xA, yA, xB, yB):
    zone_mask = mask[yA:yB, xA:xB]
    zone_lab  = lab_img[yA:yB, xA:xB]
    valid     = zone_lab[zone_mask > 0]

    if len(valid) < 80:
        return None, 0

    # Exclure les 10% extrêmes de luminosité
    # pour éliminer les derniers reflets et ombres
    L_vals  = valid[:, 0]
    p10     = np.percentile(L_vals, 10)
    p90     = np.percentile(L_vals, 90)
    keep    = (L_vals >= p10) & (L_vals <= p90)
    valid   = valid[keep]

    if len(valid) < 50:
        return None, 0

    return np.array([
        float(np.median(valid[:, 0])),
        float(np.median(valid[:, 1])),
        float(np.median(valid[:, 2]))
    ]), len(valid)


# =============================================
# DÉTECTER LA VUE + ÉLÉMENTS VISIBLES
# Retourne : view_type, orientation, elements
#
# view_type : "side" | "front" | "rear" |
#             "front_3q" | "rear_3q"
# elements  : liste des pièces détectées
#             {"phares_av", "feux_ar",
#              "calandre", "capot", "vitre_av"}
# =============================================
def detect_view(car_crop, detections):
    crop_h, crop_w = car_crop.shape[:2]
    hsv  = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
    log  = []

    ratio_wh = crop_w / max(crop_h, 1)

    # --- Feux rouges (arrière) ---
    mr1 = cv2.inRange(hsv, (0,   60, 60), (12,  255, 255))
    mr2 = cv2.inRange(hsv, (168, 60, 60), (180, 255, 255))
    mask_red   = cv2.bitwise_or(mr1, mr2)
    red_total  = cv2.countNonZero(mask_red)
    red_left   = cv2.countNonZero(mask_red[:, :crop_w//2])
    red_right  = cv2.countNonZero(mask_red[:, crop_w//2:])

    # --- Phares avant (blanc/jaune lumineux) ---
    mw1 = cv2.inRange(hsv, (0,  0,  185), (180, 60, 255))
    mw2 = cv2.inRange(hsv, (15, 40, 185), (40,  200, 255))
    mask_white  = cv2.bitwise_or(mw1, mw2)
    white_total = cv2.countNonZero(mask_white)
    white_left  = cv2.countNonZero(mask_white[:, :crop_w//2])
    white_right = cv2.countNonZero(mask_white[:, crop_w//2:])

    # --- Vitres (zones sombres dans la partie haute) ---
    top = gray[int(crop_h*0.05):int(crop_h*0.55), :]
    dk  = (top < 75).astype(np.uint8)
    dk_f = cv2.GaussianBlur(dk.astype(np.float32), (15,15), 0)
    dark_left  = float(dk_f[:, :crop_w//2].sum())
    dark_right = float(dk_f[:, crop_w//2:].sum())
    dark_total = dark_left + dark_right

    # --- Éléments détectés ---
    elements = set()
    if red_total   > 200: elements.add("feux_ar")
    if white_total > 200: elements.add("phares_av")
    if dark_total  > 5000: elements.add("vitre")

    log.append(f"rouge={red_total} blanc={white_total} vitres={int(dark_total)}")
    log.append(f"ratio_wh={ratio_wh:.2f}")

    # =============================================
    # RÈGLE 1 : VUE DE CÔTÉ
    # ratio > 1.5 ET vitres présentes ET feux
    # d'un seul côté
    # =============================================
    if ratio_wh > 1.5 and "vitre" in elements:
        elements.add("porte")
        elements.add("capot_lateral")

        # Déterminer orientation via feux
        if red_total > 200:
            if red_left > red_right * 1.4:
                log.append("SIDE: feux rouge gauche → avant DROITE")
                return "side", "right", elements, log
            elif red_right > red_left * 1.4:
                log.append("SIDE: feux rouge droite → avant GAUCHE")
                return "side", "left", elements, log

        if white_total > 200:
            if white_left > white_right * 1.4:
                log.append("SIDE: phares gauche → avant GAUCHE")
                return "side", "left", elements, log
            elif white_right > white_left * 1.4:
                log.append("SIDE: phares droite → avant DROITE")
                return "side", "right", elements, log

        # Pare-brise
        if dark_left > dark_right * 1.2:
            return "side", "left", elements, log
        return "side", "right", elements, log

    # =============================================
    # RÈGLE 2 : VUE DE FACE
    # Phares des deux côtés + pas de feux rouges
    # =============================================
    white_balanced = (white_left > 80 and white_right > 80 and
                      max(white_left, white_right) < min(white_left, white_right) * 3.5)

    if white_balanced and red_total < 100 and ratio_wh < 1.8:
        elements.update({"phares_av", "capot", "parechoc_av"})
        log.append("FRONT: phares des deux côtés")
        return "front", "front", elements, log

    # =============================================
    # RÈGLE 3 : VUE DE DERRIÈRE
    # Feux rouges des deux côtés + pas de phares
    # =============================================
    red_balanced = (red_left > 80 and red_right > 80 and
                    max(red_left, red_right) < min(red_left, red_right) * 3.5)

    if red_balanced and white_total < 100 and ratio_wh < 1.8:
        elements.update({"feux_ar", "coffre", "parechoc_ar"})
        log.append("REAR: feux rouges des deux côtés")
        return "rear", "rear", elements, log

    # =============================================
    # RÈGLE 4 : 3/4 AVANT
    # Phares visibles d'un côté, pas de vitres longues
    # =============================================
    if white_total > 150 and red_total < 80 and ratio_wh < 1.6:
        elements.update({"phares_av", "capot", "parechoc_av", "aile_av"})
        if white_left > white_right:
            log.append("FRONT_3Q: phares à gauche → avant GAUCHE")
            return "front_3q", "left", elements, log
        else:
            log.append("FRONT_3Q: phares à droite → avant DROITE")
            return "front_3q", "right", elements, log

    # =============================================
    # RÈGLE 5 : 3/4 ARRIÈRE
    # Feux rouges d'un côté, pas de vitres longues
    # =============================================
    if red_total > 150 and white_total < 80 and ratio_wh < 1.6:
        elements.update({"feux_ar", "coffre", "parechoc_ar", "aile_ar"})
        if red_left > red_right:
            log.append("REAR_3Q: feux à gauche → avant DROITE")
            return "rear_3q", "right", elements, log
        else:
            log.append("REAR_3Q: feux à droite → avant GAUCHE")
            return "rear_3q", "left", elements, log

    # Fallback vue de côté
    elements.update({"porte", "capot_lateral"})
    log.append("FALLBACK: side par défaut")
    return "side", "left", elements, log


# =============================================
# DÉFINIR LES ZONES SELON LA VUE ET ÉLÉMENTS
# Zones nommées d'après les vraies pièces
# visibles dans l'image
# =============================================
def define_zones(view_type, orientation, crop_h, crop_w, elements):
    band_y1 = int(crop_h * 0.10)
    band_y2 = int(crop_h * 0.88)

    # -----------------------------------------------
    # VUE DE CÔTÉ : Aile avant / Portes / Aile arrière
    # -----------------------------------------------
    if view_type == "side":
        cut1 = int(crop_w * 0.25)
        cut2 = int(crop_w * 0.65)

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
    # VUE DE FACE : Aile gauche / Capot / Aile droite
    # + Pare-chocs avant en bas
    # -----------------------------------------------
    elif view_type == "front":
        cut1      = int(crop_w * 0.22)
        cut2      = int(crop_w * 0.78)
        capot_y2  = int(crop_h * 0.52)
        pc_y1     = int(crop_h * 0.60)

        return [
            {"name": "Aile av. gauche", "xA": 0,    "xB": cut1,   "yA": band_y1, "yB": band_y2},
            {"name": "Capot",           "xA": cut1, "xB": cut2,   "yA": band_y1, "yB": capot_y2},
            {"name": "Pare-chocs av.",  "xA": cut1, "xB": cut2,   "yA": pc_y1,   "yB": band_y2},
            {"name": "Aile av. droite", "xA": cut2, "xB": crop_w, "yA": band_y1, "yB": band_y2},
        ]

    # -----------------------------------------------
    # VUE DE DERRIÈRE : Aile arr. gauche / Coffre /
    # Aile arr. droite + Pare-chocs arrière
    # -----------------------------------------------
    elif view_type == "rear":
        cut1    = int(crop_w * 0.22)
        cut2    = int(crop_w * 0.78)
        co_y2   = int(crop_h * 0.52)
        pc_y1   = int(crop_h * 0.60)

        return [
            {"name": "Aile arr. gauche", "xA": 0,    "xB": cut1,   "yA": band_y1, "yB": band_y2},
            {"name": "Coffre",           "xA": cut1, "xB": cut2,   "yA": band_y1, "yB": co_y2},
            {"name": "Pare-chocs arr.",  "xA": cut1, "xB": cut2,   "yA": pc_y1,   "yB": band_y2},
            {"name": "Aile arr. droite", "xA": cut2, "xB": crop_w, "yA": band_y1, "yB": band_y2},
        ]

    # -----------------------------------------------
    # 3/4 AVANT : Capot / Aile avant / Pare-chocs av.
    # Zones positionnées là où les pièces sont réellement
    # -----------------------------------------------
    elif view_type == "front_3q":
        # Le capot est en haut, l'aile en bas-côté,
        # le pare-chocs en bas-centre
        capot_y2 = int(crop_h * 0.48)
        pc_y1    = int(crop_h * 0.62)

        if orientation == "left":
            # Avant à gauche
            cut_capot = int(crop_w * 0.55)
            cut_aile  = int(crop_w * 0.30)
            return [
                {"name": "Capot avant",    "xA": cut_aile, "xB": crop_w, "yA": band_y1, "yB": capot_y2},
                {"name": "Aile avant",     "xA": 0,        "xB": cut_aile,"yA": band_y1, "yB": band_y2},
                {"name": "Pare-chocs av.", "xA": cut_aile, "xB": crop_w, "yA": pc_y1,   "yB": band_y2},
            ]
        else:
            # Avant à droite
            cut_capot = int(crop_w * 0.45)
            cut_aile  = int(crop_w * 0.70)
            return [
                {"name": "Capot avant",    "xA": 0,        "xB": cut_aile, "yA": band_y1, "yB": capot_y2},
                {"name": "Aile avant",     "xA": cut_aile, "xB": crop_w,   "yA": band_y1, "yB": band_y2},
                {"name": "Pare-chocs av.", "xA": 0,        "xB": cut_aile, "yA": pc_y1,   "yB": band_y2},
            ]

    # -----------------------------------------------
    # 3/4 ARRIÈRE : Coffre / Aile arrière / Pare-chocs
    # -----------------------------------------------
    elif view_type == "rear_3q":
        coffre_y2 = int(crop_h * 0.48)
        pc_y1     = int(crop_h * 0.62)

        if orientation == "left":
            cut_aile = int(crop_w * 0.30)
            return [
                {"name": "Coffre",          "xA": cut_aile, "xB": crop_w, "yA": band_y1, "yB": coffre_y2},
                {"name": "Aile arriere",    "xA": 0,        "xB": cut_aile,"yA": band_y1, "yB": band_y2},
                {"name": "Pare-chocs arr.", "xA": cut_aile, "xB": crop_w, "yA": pc_y1,   "yB": band_y2},
            ]
        else:
            cut_aile = int(crop_w * 0.70)
            return [
                {"name": "Coffre",          "xA": 0,        "xB": cut_aile, "yA": band_y1, "yB": coffre_y2},
                {"name": "Aile arriere",    "xA": cut_aile, "xB": crop_w,   "yA": band_y1, "yB": band_y2},
                {"name": "Pare-chocs arr.", "xA": 0,        "xB": cut_aile, "yA": pc_y1,   "yB": band_y2},
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
        # DÉTECTER VUE + ÉLÉMENTS VISIBLES
        # ===============================================
        view_type, orientation, elements, view_log = detect_view(
            car_crop, detections
        )

        # ===============================================
        # ZONES ADAPTÉES
        # ===============================================
        zones = define_zones(
            view_type, orientation, crop_h, crop_w, elements
        )

        # ===============================================
        # MASQUE + ESPACES COLORIMÉTRIQUES
        # ===============================================
        hsv_full  = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
        lab_full  = cv2.cvtColor(car_crop, cv2.COLOR_BGR2LAB)
        mask_body = build_body_mask(car_crop, hsv_full)

        # ===============================================
        # RÉFÉRENCE GLOBALE EN LAB
        # (anti-reflets : percentiles 10-90)
        # ===============================================
        vl_all = lab_full[mask_body > 0]
        if len(vl_all) < 100:
            return jsonify({"error": "No body pixels found"}), 400

        L_all  = vl_all[:, 0]
        p10    = np.percentile(L_all, 10)
        p90    = np.percentile(L_all, 90)
        keep   = (L_all >= p10) & (L_all <= p90)
        vl_ref = vl_all[keep]

        ref_color = np.array([
            float(np.median(vl_ref[:, 0])),
            float(np.median(vl_ref[:, 1])),
            float(np.median(vl_ref[:, 2]))
        ])

        # Variabilité naturelle normalisée
        nat_std_a = max(float(np.std(vl_ref[:, 1])), 1.0)
        nat_std_b = max(float(np.std(vl_ref[:, 2])), 1.0)

        # ===============================================
        # DESSIN
        # ===============================================
        final_img      = img_orig.copy()
        thick_box      = max(3, int(5 * min(scale_x, scale_y)))
        thick_line     = max(1, int(1 * min(scale_x, scale_y)))
        font_scale_big = max(0.55, 0.55 * min(scale_x, scale_y))
        font_scale_med = max(0.45, 0.45 * min(scale_x, scale_y))
        font_thick_big = max(2,    int(2  * min(scale_x, scale_y)))
        overlay_h      = max(55,   int(55 * scale_y))

        cv2.rectangle(final_img, (x1, y1), (x2, y2), (220, 220, 220), thick_line)

        # Séparateurs entre zones
        drawn_x = set()
        step = max(10, int(12 * scale_y))
        dash = max(4,  int(6  * scale_y))
        for zone in zones:
            for cx in [zone["xA"], zone["xB"]]:
                if cx in drawn_x or cx == 0 or cx == crop_w:
                    continue
                drawn_x.add(cx)
                yA_d = y1 + zone["yA"]
                yB_d = y1 + zone["yB"]
                for dy in range(yA_d, yB_d, step):
                    cv2.line(final_img,
                             (x1 + cx, dy),
                             (x1 + cx, dy + dash),
                             (255, 255, 255), thick_line)

        results_zones = []
        detected = 0

        for zone in zones:
            xA, xB = zone["xA"], zone["xB"]
            yA, yB = zone["yA"], zone["yB"]

            zone_color, px_count = get_zone_color(
                lab_full, mask_body, xA, yA, xB, yB
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
                # Delta LAB normalisé par variabilité naturelle
                da   = abs(zone_color[1] - ref_color[1]) / nat_std_a
                db   = abs(zone_color[2] - ref_color[2]) / nat_std_b
                diff = float(np.sqrt(da**2 + db**2))

                label_score = f"{diff:.1f}"

                if diff > 1.8:
                    color_rect = (0, 0, 255)
                    verdict    = "Peinture refaite!"
                    detected  += 1
                elif diff > 0.9:
                    color_rect = (0, 165, 255)
                    verdict    = "Variation suspecte"
                    detected  += 1
                else:
                    color_rect = (0, 210, 0)
                    verdict    = "OK"

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

            cv2.putText(final_img, f"Score: {label_score}",
                        (abs_x1 + 8, abs_y1 + int(overlay_h * 0.80)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale_med, color_rect, font_thick_big)

            cv2.putText(final_img, verdict,
                        (abs_x1 + 8, abs_y2 - int(10 * scale_y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale_med, color_rect, font_thick_big)

            results_zones.append({
                "zone":    zone["name"],
                "score":   round(diff, 2),
                "pixels":  px_count,
                "verdict": verdict
            })

        # ===============================================
        # SCORE GLOBAL
        # ===============================================
        scores = [z["score"] for z in results_zones if z["score"] > 0]
        final_score_raw = float(np.mean(scores)) if scores else 0.0
        final_score_100 = min(int(final_score_raw * 40), 100)

        if final_score_raw > 1.8:
            result = "Difference importante — repeinture probable"
        elif final_score_raw > 0.9:
            result = "Legeres variations detectees"
        else:
            result = "Peinture homogene (OK)"

        analysed_name = "analysed_" + filename
        analysed_path = os.path.join(UPLOAD_FOLDER, analysed_name)
        cv2.imwrite(analysed_path, final_img)

        if os.path.exists(resized_path):
            os.remove(resized_path)

        return jsonify({
            "yolo":           yolo_result,
            "score":          final_score_100,
            "score_raw":      round(final_score_raw, 2),
            "result":         result,
            "zones":          results_zones,
            "zones_detected": detected,
            "view_type":      view_type,
            "orientation":    orientation,
            "elements":       list(elements),
            "view_log":       view_log,
            "image_size":     {"width": orig_w, "height": orig_h},
            "calibration": {
                "nat_std_a": round(nat_std_a, 1),
                "nat_std_b": round(nat_std_b, 1),
                "ref_L":     round(ref_color[0], 1),
                "ref_a":     round(ref_color[1], 1),
                "ref_b":     round(ref_color[2], 1)
            },
            "image_result":   analysed_name,
            "image_url":      request.host_url + "uploads/" + analysed_name
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
