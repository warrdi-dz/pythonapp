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
# MASQUE CARROSSERIE
# Exclut : vitres, roues, reflets blancs forts,
# ombres noires, plastique, chrome, ciel
# =============================================
def build_body_mask(car_crop, hsv):
    # Trop sombre = vitres, pneus, taches noires
    mask_dark   = cv2.inRange(hsv, (0, 0,   0), (180, 255,  40))
    # Reflets blancs très forts = soleil sur carrosserie
    mask_reflet = cv2.inRange(hsv, (0, 0, 215), (180, 255, 255))
    # Ciel / fond blanc
    mask_sky    = cv2.inRange(hsv, (0, 0, 210), (180,  20, 255))
    # Faible saturation = plastique, chrome, calandre
    mask_chrome = cv2.inRange(hsv, (0, 0,   0), (180,  28, 255))

    exclude   = cv2.bitwise_or(mask_dark,   mask_reflet)
    exclude   = cv2.bitwise_or(exclude,     mask_sky)
    exclude   = cv2.bitwise_or(exclude,     mask_chrome)
    mask_body = cv2.bitwise_not(exclude)

    k         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_CLOSE, k, iterations=2)
    mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_OPEN,  k, iterations=1)
    return mask_body


# =============================================
# COULEUR LAB MÉDIANE ANTI-REFLETS
# Exclut les 10% extrêmes de luminosité
# pour éliminer reflets résiduels et ombres
# =============================================
def get_zone_color(lab_img, mask, xA, yA, xB, yB):
    zm   = mask[yA:yB, xA:xB]
    zl   = lab_img[yA:yB, xA:xB]
    valid = zl[zm > 0]

    if len(valid) < 80:
        return None, 0

    # Exclure 10% plus sombres et 10% plus brillants
    L     = valid[:, 0]
    p10   = np.percentile(L, 10)
    p90   = np.percentile(L, 90)
    keep  = (L >= p10) & (L <= p90)
    valid = valid[keep]

    if len(valid) < 50:
        return None, 0

    return np.array([
        float(np.median(valid[:, 0])),
        float(np.median(valid[:, 1])),
        float(np.median(valid[:, 2]))
    ]), len(valid)


# =============================================
# DÉTECTER LA VUE + CE QUI EST VISIBLE
#
# Retourne :
#   view_type   : "side_full" | "front_only" |
#                 "rear_only" | "rear_3q"
#   orientation : "left" | "right" | "rear"
#   info        : dict avec les comptages
# =============================================
def detect_view(car_crop):
    crop_h, crop_w = car_crop.shape[:2]
    hsv  = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
    log  = []

    ratio_wh = crop_w / max(crop_h, 1)

    # --- Feux arrière rouges ---
    mr1      = cv2.inRange(hsv, (0,   60, 60), (12,  255, 255))
    mr2      = cv2.inRange(hsv, (168, 60, 60), (180, 255, 255))
    mask_red = cv2.bitwise_or(mr1, mr2)
    red_tot  = cv2.countNonZero(mask_red)
    red_L    = cv2.countNonZero(mask_red[:, :crop_w//2])
    red_R    = cv2.countNonZero(mask_red[:, crop_w//2:])

    # --- Phares avant blancs/jaunes ---
    # On utilise un seuil plus strict pour éviter
    # de confondre reflets carrosserie et vrais phares
    mw1        = cv2.inRange(hsv, (0,  0,  195), (180, 50, 255))
    mw2        = cv2.inRange(hsv, (15, 40, 195), (40, 180, 255))
    mask_white = cv2.bitwise_or(mw1, mw2)

    # Chercher les phares dans la bande basse seulement
    # (les phares sont dans le tiers bas de la voiture)
    ph_zone     = mask_white[int(crop_h*0.45):, :]
    white_tot   = cv2.countNonZero(ph_zone)
    white_L     = cv2.countNonZero(ph_zone[:, :crop_w//2])
    white_R     = cv2.countNonZero(ph_zone[:, crop_w//2:])

    # --- Vitres (zones sombres partie haute) ---
    top       = gray[int(crop_h*0.05):int(crop_h*0.55), :]
    dk        = (top < 75).astype(np.uint8)
    dk_f      = cv2.GaussianBlur(dk.astype(np.float32), (15, 15), 0)
    glass_L   = float(dk_f[:, :crop_w//2].sum())
    glass_R   = float(dk_f[:, crop_w//2:].sum())
    glass_tot = glass_L + glass_R

    # --- Présence de portes (vitres latérales longues) ---
    has_doors = (glass_tot > 8000) and (ratio_wh > 1.3)

    log.append(f"ratio={ratio_wh:.2f} rouge={red_tot} "
               f"blanc={white_tot} vitres={int(glass_tot)} "
               f"doors={has_doors}")

    info = {
        "red_tot": red_tot, "red_L": red_L, "red_R": red_R,
        "white_tot": white_tot, "white_L": white_L, "white_R": white_R,
        "glass_tot": glass_tot, "has_doors": has_doors,
        "ratio_wh": ratio_wh
    }

    # =============================================
    # CAS 1 : VUE DE CÔTÉ COMPLÈTE
    # ratio > 1.4 ET vitres latérales présentes
    # → on voit les portes
    # =============================================
    if has_doors and ratio_wh > 1.4:
        log.append("→ VUE COTE COMPLETE (portes visibles)")

        # Orientation via feux rouges
        if red_tot > 300:
            if red_L > red_R * 1.4:
                log.append("feux gauche → avant DROITE")
                return "side_full", "right", info, log
            elif red_R > red_L * 1.4:
                log.append("feux droite → avant GAUCHE")
                return "side_full", "left", info, log

        # Orientation via phares
        if white_tot > 150:
            if white_L > white_R * 1.4:
                log.append("phares gauche → avant GAUCHE")
                return "side_full", "left", info, log
            elif white_R > white_L * 1.4:
                log.append("phares droite → avant DROITE")
                return "side_full", "right", info, log

        # Orientation via pare-brise
        if glass_L > glass_R * 1.3:
            return "side_full", "left", info, log
        return "side_full", "right", info, log

    # =============================================
    # CAS 2 : VUE AVANT SEULEMENT
    # Phares visibles + pas de feux rouges significatifs
    # + pas de portes (ratio < 1.5 ou vitres courtes)
    # =============================================
    if white_tot > 100 and red_tot < 150 and not has_doors:
        log.append("→ VUE AVANT SEULMENT (capot + phares + aile)")
        if white_L > white_R:
            return "front_only", "left", info, log
        else:
            return "front_only", "right", info, log

    # =============================================
    # CAS 3 : VUE ARRIÈRE SEULEMENT
    # Feux rouges visibles + pas de phares + pas de portes
    # =============================================
    if red_tot > 150 and white_tot < 100 and not has_doors:
        log.append("→ VUE ARRIERE SEULMENT (coffre + feux + aile arr)")
        if red_L > red_R:
            return "rear_only", "left", info, log
        else:
            return "rear_only", "right", info, log

    # =============================================
    # CAS 4 : 3/4 ARRIÈRE
    # Feux rouges + quelques vitres mais pas de portes longues
    # =============================================
    if red_tot > 100 and ratio_wh < 1.5:
        log.append("→ VUE 3/4 ARRIERE")
        if red_L > red_R:
            return "rear_3q", "left", info, log
        return "rear_3q", "right", info, log

    # Fallback côté
    log.append("→ FALLBACK: side_full")
    return "side_full", "left", info, log


# =============================================
# DÉFINIR LES ZONES SELON LA VUE
#
# CAS 1 — CÔTÉ COMPLET :
#   Aile avant | Portes | Aile arrière
#
# CAS 2 — AVANT SEULEMENT :
#   Capot avant | Aile avant | Pare-chocs av.
#   Les zones suivent la géométrie réelle :
#   - Capot = partie haute centrale
#   - Aile  = partie latérale (côté phares)
#   - Pare-chocs = partie basse
#
# CAS 3 — ARRIÈRE SEULEMENT :
#   Coffre/hayon | Aile arrière | Pare-chocs arr.
#
# CAS 4 — 3/4 ARRIÈRE :
#   Coffre | Aile arrière | Pare-chocs arr.
# =============================================
def define_zones(view_type, orientation, crop_h, crop_w):

    # Bande verticale carrosserie (exclut toit et bas)
    band_y1 = int(crop_h * 0.08)
    band_y2 = int(crop_h * 0.90)

    # --------------------------------------------------
    # CAS 1 : VUE DE CÔTÉ COMPLÈTE
    # --------------------------------------------------
    if view_type == "side_full":
        cut1 = int(crop_w * 0.25)
        cut2 = int(crop_w * 0.65)

        if orientation == "left":
            return [
                {"name": "Aile avant",   "xA": 0,    "xB": cut1,
                 "yA": band_y1, "yB": band_y2},
                {"name": "Portes",       "xA": cut1, "xB": cut2,
                 "yA": band_y1, "yB": band_y2},
                {"name": "Aile arriere", "xA": cut2, "xB": crop_w,
                 "yA": band_y1, "yB": band_y2},
            ]
        else:
            return [
                {"name": "Aile arriere", "xA": 0,    "xB": cut1,
                 "yA": band_y1, "yB": band_y2},
                {"name": "Portes",       "xA": cut1, "xB": cut2,
                 "yA": band_y1, "yB": band_y2},
                {"name": "Aile avant",   "xA": cut2, "xB": crop_w,
                 "yA": band_y1, "yB": band_y2},
            ]

    # --------------------------------------------------
    # CAS 2 : VUE AVANT SEULEMENT
    # Capot = bande haute (y: 8% → 50%)
    # Aile  = côté latéral où sont les phares
    # Pare-chocs = bande basse (y: 60% → 90%)
    # --------------------------------------------------
    elif view_type == "front_only":
        capot_y1    = int(crop_h * 0.08)
        capot_y2    = int(crop_h * 0.52)
        parechoc_y1 = int(crop_h * 0.60)
        parechoc_y2 = int(crop_h * 0.92)

        # L'aile avant est du côté des phares
        # Le capot occupe la partie centrale haute
        if orientation == "left":
            # Phares à gauche → aile à gauche
            aile_x2    = int(crop_w * 0.38)
            capot_x1   = int(crop_w * 0.30)
            capot_x2   = crop_w
            pc_x1      = int(crop_w * 0.20)
            return [
                {"name": "Capot avant",    "xA": capot_x1, "xB": capot_x2,
                 "yA": capot_y1, "yB": capot_y2},
                {"name": "Aile avant",     "xA": 0,        "xB": aile_x2,
                 "yA": capot_y1, "yB": parechoc_y2},
                {"name": "Pare-chocs av.", "xA": pc_x1,    "xB": crop_w,
                 "yA": parechoc_y1, "yB": parechoc_y2},
            ]
        else:
            # Phares à droite → aile à droite
            aile_x1    = int(crop_w * 0.62)
            capot_x1   = 0
            capot_x2   = int(crop_w * 0.70)
            pc_x2      = int(crop_w * 0.80)
            return [
                {"name": "Capot avant",    "xA": capot_x1, "xB": capot_x2,
                 "yA": capot_y1, "yB": capot_y2},
                {"name": "Aile avant",     "xA": aile_x1,  "xB": crop_w,
                 "yA": capot_y1, "yB": parechoc_y2},
                {"name": "Pare-chocs av.", "xA": 0,        "xB": pc_x2,
                 "yA": parechoc_y1, "yB": parechoc_y2},
            ]

    # --------------------------------------------------
    # CAS 3 : VUE ARRIÈRE SEULEMENT
    # Coffre/hayon = partie haute centrale
    # Aile arrière = côté latéral avec feux rouges
    # Pare-chocs arrière = partie basse
    # --------------------------------------------------
    elif view_type == "rear_only":
        coffre_y1   = int(crop_h * 0.08)
        coffre_y2   = int(crop_h * 0.52)
        parechoc_y1 = int(crop_h * 0.62)
        parechoc_y2 = int(crop_h * 0.92)

        if orientation == "left":
            # Feux à gauche → aile arrière à gauche
            aile_x2  = int(crop_w * 0.40)
            co_x1    = int(crop_w * 0.25)
            pc_x1    = int(crop_w * 0.15)
            return [
                {"name": "Coffre / hayon",  "xA": co_x1,  "xB": crop_w,
                 "yA": coffre_y1, "yB": coffre_y2},
                {"name": "Aile arriere",    "xA": 0,       "xB": aile_x2,
                 "yA": coffre_y1, "yB": parechoc_y2},
                {"name": "Pare-chocs arr.", "xA": pc_x1,   "xB": crop_w,
                 "yA": parechoc_y1, "yB": parechoc_y2},
            ]
        else:
            # Feux à droite → aile arrière à droite
            aile_x1  = int(crop_w * 0.60)
            co_x2    = int(crop_w * 0.75)
            pc_x2    = int(crop_w * 0.85)
            return [
                {"name": "Coffre / hayon",  "xA": 0,       "xB": co_x2,
                 "yA": coffre_y1, "yB": coffre_y2},
                {"name": "Aile arriere",    "xA": aile_x1, "xB": crop_w,
                 "yA": coffre_y1, "yB": parechoc_y2},
                {"name": "Pare-chocs arr.", "xA": 0,       "xB": pc_x2,
                 "yA": parechoc_y1, "yB": parechoc_y2},
            ]

    # --------------------------------------------------
    # CAS 4 : 3/4 ARRIÈRE
    # --------------------------------------------------
    elif view_type == "rear_3q":
        coffre_y2   = int(crop_h * 0.50)
        parechoc_y1 = int(crop_h * 0.60)

        if orientation == "left":
            cut = int(crop_w * 0.42)
            return [
                {"name": "Coffre / hayon",  "xA": cut, "xB": crop_w,
                 "yA": band_y1, "yB": coffre_y2},
                {"name": "Aile arriere",    "xA": 0,   "xB": cut,
                 "yA": band_y1, "yB": band_y2},
                {"name": "Pare-chocs arr.", "xA": cut, "xB": crop_w,
                 "yA": parechoc_y1, "yB": band_y2},
            ]
        else:
            cut = int(crop_w * 0.58)
            return [
                {"name": "Coffre / hayon",  "xA": 0,   "xB": cut,
                 "yA": band_y1, "yB": coffre_y2},
                {"name": "Aile arriere",    "xA": cut, "xB": crop_w,
                 "yA": band_y1, "yB": band_y2},
                {"name": "Pare-chocs arr.", "xA": 0,   "xB": cut,
                 "yA": parechoc_y1, "yB": band_y2},
            ]

    # Fallback
    cut1 = int(crop_w * 0.33)
    cut2 = int(crop_w * 0.67)
    return [
        {"name": "Zone gauche",  "xA": 0,    "xB": cut1,
         "yA": band_y1, "yB": band_y2},
        {"name": "Zone centre",  "xA": cut1, "xB": cut2,
         "yA": band_y1, "yB": band_y2},
        {"name": "Zone droite",  "xA": cut2, "xB": crop_w,
         "yA": band_y1, "yB": band_y2},
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
        # DÉTECTER LA VUE
        # ===============================================
        view_type, orientation, view_info, view_log = detect_view(car_crop)

        # ===============================================
        # ZONES ADAPTÉES À LA VUE
        # ===============================================
        zones = define_zones(view_type, orientation, crop_h, crop_w)

        # ===============================================
        # MASQUE + ESPACES COLORIMÉTRIQUES
        # ===============================================
        hsv_full  = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
        lab_full  = cv2.cvtColor(car_crop, cv2.COLOR_BGR2LAB)
        mask_body = build_body_mask(car_crop, hsv_full)

        # ===============================================
        # RÉFÉRENCE GLOBALE LAB (percentiles 10-90)
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

        # Variabilité naturelle pour normalisation
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

        # Séparateurs entre zones (évite les doublons)
        drawn_x = set()
        step = max(10, int(12 * scale_y))
        dash = max(4,  int(6  * scale_y))
        for zone in zones:
            for cx in [zone["xA"], zone["xB"]]:
                if cx in drawn_x or cx == 0 or cx == crop_w:
                    continue
                drawn_x.add(cx)
                abs_cy = x1 + cx
                yA_d   = y1 + zone["yA"]
                yB_d   = y1 + zone["yB"]
                for dy in range(yA_d, yB_d, step):
                    cv2.line(final_img,
                             (abs_cy, dy),
                             (abs_cy, dy + dash),
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
                # Score normalisé LAB (a,b) sans luminosité L
                da   = abs(zone_color[1] - ref_color[1]) / nat_std_a
                db   = abs(zone_color[2] - ref_color[2]) / nat_std_b
                diff = float(np.sqrt(da**2 + db**2))

                label_score = f"{diff:.1f}"

                if diff < 1:
                    color_rect = (0, 0, 255)
                    verdict    = "Peinture refaite!"
                    detected  += 1
                elif diff > 1 and diff< 1.8:
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
        scores      = [z["score"] for z in results_zones if z["score"] > 0]
        score_raw   = float(np.mean(scores)) if scores else 0.0
        score_100   = min(int(score_raw * 40), 100)

        if score_raw > 1.8:
            result = "Difference importante — repeinture probable"
        elif score_raw > 0.9:
            result = "Legeres variations detectees"
        else:
            result = "Peinture homogene (OK)"

        analysed_name = "analysed_" + filename
        analysed_path = os.path.join(UPLOAD_FOLDER, analysed_name)
        cv2.imwrite(analysed_path, final_img)

        if os.path.exists(resized_path):
            os.remove(resized_path)

        return jsonify({
            "yolo":          yolo_result,
            "score":         score_100,
            "score_raw":     round(score_raw, 2),
            "result":        result,
            "zones":         results_zones,
            "zones_detected":detected,
            "view_type":     view_type,
            "orientation":   orientation,
            "view_log":      view_log,
            "image_size":    {"width": orig_w, "height": orig_h},
            "calibration": {
                "nat_std_a": round(nat_std_a, 1),
                "nat_std_b": round(nat_std_b, 1),
                "ref_L":     round(ref_color[0], 1),
                "ref_a":     round(ref_color[1], 1),
                "ref_b":     round(ref_color[2], 1)
            },
            "image_result":  analysed_name,
            "image_url":     request.host_url + "uploads/" + analysed_name
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
