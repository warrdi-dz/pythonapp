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
        ext  = os.path.splitext(image_path)[1].lower()
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
        with open(image_path, "rb") as f:
            files   = {"image": (os.path.basename(image_path), f, mime)}
            headers = {"Accept": "application/json"}
            r = requests.post(url, files=files, headers=headers, timeout=20)
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
    return jsonify({"status": "OK", "message": "GARAGE PRO V6"})


# =============================================
# COULEUR LAB MÉDIANE dans un POLYGONE
# Utilise LAB pour une meilleure précision
# Exclut les 10% extrêmes (reflets et ombres)
# =============================================
def get_poly_color_lab(lab_img, body_mask, polygon):
    h, w = lab_img.shape[:2]
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [np.array(polygon, dtype=np.int32)], 255)
    combined  = cv2.bitwise_and(poly_mask, body_mask)
    valid     = lab_img[combined > 0]
    if len(valid) < 80:
        return None, 0
    # Exclure 10% extrêmes de luminosité
    L    = valid[:, 0]
    p10  = np.percentile(L, 10)
    p90  = np.percentile(L, 90)
    keep = (L >= p10) & (L <= p90)
    valid = valid[keep]
    if len(valid) < 50:
        return None, 0
    return np.array([
        float(np.median(valid[:, 0])),
        float(np.median(valid[:, 1])),
        float(np.median(valid[:, 2]))
    ]), len(valid)


# =============================================
# MASQUE CARROSSERIE ANTI-OMBRES / REFLETS
# =============================================
def build_body_mask(car_crop, hsv):
    # Trop sombre = vitres, pneus, taches noires
    mask_dark = cv2.inRange(hsv, (0, 0,   0), (180, 255,  45))
    # Reflets blancs très forts
    mask_refl = cv2.inRange(hsv, (0, 0, 218), (180, 255, 255))
    # Ciel
    mask_sky  = cv2.inRange(hsv, (0, 0, 210), (180,  20, 255))
    # Chrome / plastique
    mask_chro = cv2.inRange(hsv, (0, 0,   0), (180,  28, 255))

    exclude   = cv2.bitwise_or(mask_dark, mask_refl)
    exclude   = cv2.bitwise_or(exclude,   mask_sky)
    exclude   = cv2.bitwise_or(exclude,   mask_chro)
    mask_body = cv2.bitwise_not(exclude)

    k         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_CLOSE, k, iterations=2)
    mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_OPEN,  k, iterations=1)

    h_c, w_c = car_crop.shape[:2]

    # Supprimer ombres par forme
    mask_semi           = cv2.inRange(hsv, (0, 0, 35), (180, 255, 130))
    mask_shadow_on_body = cv2.bitwise_and(mask_semi, mask_body)
    ks                  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_shadow_on_body = cv2.morphologyEx(
        mask_shadow_on_body, cv2.MORPH_CLOSE, ks, iterations=3
    )

    contours, _ = cv2.findContours(
        mask_shadow_on_body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    mask_rm         = np.zeros_like(mask_body)
    total_body_area = max(cv2.countNonZero(mask_body), 1)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        ratio_hw   = h / max(w, 1)
        ratio_wh   = w / max(h, 1)
        area_ratio = area / total_body_area
        is_pole    = (ratio_hw > 3.0) and (w < w_c * 0.12)
        is_long    = (ratio_wh > 4.0) and (area_ratio > 0.04)
        touch_bot  = (y + h) > (h_c * 0.88)
        is_round   = (area_ratio < 0.07) and (ratio_hw < 1.6) and (ratio_wh < 1.6)
        if is_pole or is_long or touch_bot or is_round:
            cv2.drawContours(mask_rm, [cnt], -1, 255, -1)

    mask_body = cv2.bitwise_and(mask_body, cv2.bitwise_not(mask_rm))
    mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_CLOSE, k, iterations=1)
    return mask_body


# =========================
# REFINE CROP
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
# DETECTION FEUX
# =========================
def detect_lights(car_crop):
    h, w    = car_crop.shape[:2]
    band_w  = int(w * 0.28)
    feux_y1 = int(h * 0.30)
    feux_y2 = int(h * 0.92)

    left_feux  = car_crop[feux_y1:feux_y2, 0:band_w]
    right_feux = car_crop[feux_y1:feux_y2, w - band_w:w]
    left_hsv   = cv2.cvtColor(left_feux,  cv2.COLOR_BGR2HSV)
    right_hsv  = cv2.cvtColor(right_feux, cv2.COLOR_BGR2HSV)

    def count_red(hsv):
        m1 = cv2.inRange(hsv, (0,   60, 60), (12,  255, 255))
        m2 = cv2.inRange(hsv, (168, 60, 60), (180, 255, 255))
        return int(cv2.countNonZero(cv2.bitwise_or(m1, m2)))

    def count_white(hsv):
        white  = cv2.inRange(hsv, (0,  0,  170), (180, 90,  255))
        yellow = cv2.inRange(hsv, (15, 40, 170), (40,  220, 255))
        return int(cv2.countNonZero(cv2.bitwise_or(white, yellow)))

    rl = count_red(left_hsv);   rr = count_red(right_hsv)
    wl = count_white(left_hsv); wr = count_white(right_hsv)
    ba = band_w * (feux_y2 - feux_y1)

    return {
        "red_left": rl, "red_right": rr,
        "white_left": wl, "white_right": wr,
        "red_tot": rl+rr, "white_tot": wl+wr,
        "band_area": ba,
        "red_left_ratio":  rl/max(ba,1),
        "red_right_ratio": rr/max(ba,1),
        "whi_left_ratio":  wl/max(ba,1),
        "whi_right_ratio": wr/max(ba,1),
    }


# =========================
# DETECTION AVANT/ARRIERE
# =========================
def detect_front_rear(lights):
    log = []
    rl, rr = lights["red_left"],   lights["red_right"]
    wl, wr = lights["white_left"], lights["white_right"]
    red_tot = rl + rr; white_tot = wl + wr

    rear_side  = "left"  if rl > rr*1.25 else ("right" if rr > rl*1.25 else None)
    front_side = "left"  if wl > wr*1.25 else ("right" if wr > wl*1.25 else None)

    if   red_tot > white_tot*1.4 and red_tot > 150:   facing = "rear"
    elif white_tot > red_tot*1.4 and white_tot > 150: facing = "front"
    else:                                               facing = "side"

    if rear_side  and not front_side:
        front_side = "right" if rear_side  == "left" else "left"
    if front_side and not rear_side:
        rear_side  = "right" if front_side == "left" else "left"
    if not rear_side and not front_side:
        rear_side, front_side = "right", "left"
        log.append("Fallback arriere=droite")

    log.append(f"rouge G={rl} D={rr} | blanc G={wl} D={wr}")
    log.append(f"facing={facing} rear={rear_side} front={front_side}")
    return rear_side, front_side, facing, log


# =========================
# ESTIMER L'ANGLE
# =========================
def estimate_angle(lights, crop_w, crop_h, facing):
    rl, rr = lights["red_left"],   lights["red_right"]
    wl, wr = lights["white_left"], lights["white_right"]

    if   facing == "rear":  big = max(rl,rr); sml = min(rl,rr)
    elif facing == "front": big = max(wl,wr); sml = min(wl,wr)
    else:
        big = max(rl+wl, rr+wr); sml = min(rl+wl, rr+wr)

    if big == 0:
        return 45.0

    sym = big / max(sml, 1)

    if   sym >= 8.0: angle = 10.0
    elif sym >= 4.0: angle = 10.0 + (8.0-sym)/4.0 * 30.0
    elif sym >= 2.0: angle = 40.0 + (4.0-sym)/2.0 * 25.0
    elif sym >= 1.3: angle = 65.0 + (2.0-sym)/0.7 * 20.0
    else:            angle = 87.0

    ratio_wh = crop_w / max(crop_h, 1)
    if ratio_wh > 1.6: angle = min(angle, 35.0)
    elif ratio_wh < 0.9: angle = max(angle, 60.0)

    return round(angle, 1)


# =============================================
# SCORE NORMALISÉ LAB
# Utilise les canaux a,b (teinte pure) normalisés
# par la variabilité naturelle de la carrosserie.
# Retourne un score 0-∞ où :
#   < 1.0  → OK
#   1.0-2.0 → Variation suspecte
#   > 2.0  → Peinture refaite
#
# CLEF : on compare la teinte (a,b) pas la luminosité (L)
# pour éviter les faux positifs dus à l'éclairage
# =============================================
def compute_score(zone_color, ref_color, nat_std_a, nat_std_b,
                   zone_tex, ref_tex):
    # Delta teinte normalisé
    da         = abs(zone_color[1] - ref_color[1]) / nat_std_a
    db         = abs(zone_color[2] - ref_color[2]) / nat_std_b
    color_diff = float(np.sqrt(da**2 + db**2))

    # Delta texture normalisé
    tex_diff   = min(abs(zone_tex - ref_tex) / max(ref_tex, 1), 2.0)

    # Score final : 85% couleur + 15% texture
    return round((color_diff * 0.85) + (tex_diff * 0.15), 2)


# =========================
# POLYGONE TRAPEZE PERSPECTIF
# =========================
def make_poly(crop_w, crop_h, xA, xB, top_base, bot_base, persp, tilt_dir):
    drop = 0.08

    def top_y(x):
        f = x / max(1, crop_w)
        return top_base + (persp*drop*f if tilt_dir>0 else persp*drop*(1-f))

    def bot_y(x):
        f = x / max(1, crop_w)
        return bot_base - (persp*drop*f if tilt_dir>0 else persp*drop*(1-f))

    return [
        (xA, int(top_y(xA)*crop_h)),
        (xB, int(top_y(xB)*crop_h)),
        (xB, int(bot_y(xB)*crop_h)),
        (xA, int(bot_y(xA)*crop_h)),
    ]


# =========================
# BUILD ZONES SELON ANGLE
# =========================
def build_zones(crop_w, crop_h, angle, rear_side, front_side, facing, lights):
    rl, rr = lights["red_left"],   lights["red_right"]
    wl, wr = lights["white_left"], lights["white_right"]

    if   facing == "rear":  near_side = "left" if rl>rr else "right"
    elif facing == "front": near_side = "left" if wl>wr else "right"
    else:                   near_side = "left" if (rl+wl)>(rr+wr) else "right"

    far_side = "right" if near_side=="left" else "left"

    if   angle <= 10: persp = 0.0
    elif angle <= 55: persp = 0.6 * min(1.0, (angle-10)/45.0)
    elif angle <= 80: persp = 0.6 * max(0.0, 1.0-(angle-55)/25.0)
    else:             persp = 0.0

    tilt_dir = +1 if far_side=="right" else -1
    top_base = 0.20
    bot_base = 0.88

    is_rear  = (facing=="rear") or (facing=="side" and (rl+rr)>=(wl+wr))
    panel    = "Coffre"     if is_rear else "Capot"
    pc_label = "Pare-ch.AR" if is_rear else "Pare-ch.AV"

    def zone(name, a, b):
        return {"name": name, "poly": make_poly(
            crop_w, crop_h, int(a*crop_w), int(b*crop_w),
            top_base, bot_base, persp, tilt_dir
        )}

    if angle <= 25:
        log = f"PROFIL(angle={angle}) near={near_side}"
        if near_side=="right":
            return ([zone("Aile AV",0.00,0.20), zone("Porte AV",0.20,0.48),
                     zone("Porte AR",0.48,0.78), zone("Aile AR",0.78,1.00)]
                    if is_rear else
                    [zone("Aile AR",0.00,0.20), zone("Porte AR",0.20,0.48),
                     zone("Porte AV",0.48,0.78), zone("Aile AV",0.78,1.00)]), log
        else:
            return ([zone("Aile AR",0.00,0.22), zone("Porte AR",0.22,0.52),
                     zone("Porte AV",0.52,0.80), zone("Aile AV",0.80,1.00)]
                    if is_rear else
                    [zone("Aile AV",0.00,0.22), zone("Porte AV",0.22,0.52),
                     zone("Porte AR",0.52,0.80), zone("Aile AR",0.80,1.00)]), log

    elif angle <= 55:
        log = f"3/4 LEGER(angle={angle}) near={near_side}"
        if near_side=="right":
            return ([zone("Porte AV",0.00,0.22), zone("Porte AR",0.22,0.50),
                     zone("Aile AR",0.50,0.72), zone(pc_label,0.72,1.00)]
                    if is_rear else
                    [zone("Porte AR",0.00,0.22), zone("Porte AV",0.22,0.50),
                     zone("Aile AV",0.50,0.72), zone(pc_label,0.72,1.00)]), log
        else:
            return ([zone(pc_label,0.00,0.28), zone("Aile AR",0.28,0.50),
                     zone("Porte AR",0.50,0.78), zone("Porte AV",0.78,1.00)]
                    if is_rear else
                    [zone(pc_label,0.00,0.28), zone("Aile AV",0.28,0.50),
                     zone("Porte AV",0.50,0.78), zone("Porte AR",0.78,1.00)]), log

    elif angle <= 80:
        log = f"3/4 MARQUE(angle={angle}) near={near_side}"
        if near_side=="right":
            return [zone("Aile AR" if is_rear else "Aile AV",0.42,0.68),
                    zone(panel,0.68,0.85), zone(pc_label,0.85,1.00)], log
        else:
            return [zone(pc_label,0.00,0.15), zone(panel,0.15,0.32),
                    zone("Aile AR" if is_rear else "Aile AV",0.32,0.58)], log

    else:
        log = f"FACE/DOS(angle={angle})"
        if is_rear:
            return [zone("Aile AR G",0.00,0.20), zone(pc_label,0.20,0.55),
                    zone(panel,0.45,0.80), zone("Aile AR D",0.80,1.00)], log
        else:
            return [zone("Aile AV G",0.00,0.20), zone(pc_label,0.20,0.55),
                    zone(panel,0.45,0.80), zone("Aile AV D",0.80,1.00)], log


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
        resized_path = os.path.join(UPLOAD_FOLDER, "resized_" + filename + ".jpg")
        cv2.imwrite(resized_path, img_yolo, [cv2.IMWRITE_JPEG_QUALITY, 92])

        yolo_result = call_yolo(resized_path)
        detections  = yolo_result.get("detections", [])
        cars = [d for d in detections if d.get("class") == 2]
        if not cars:
            return jsonify({"error": "Car not detected"}), 400

        scale_x = orig_w / YOLO_W
        scale_y = orig_h / YOLO_H

        raw_x1 = int(min(d["box"][0] for d in cars) * scale_x)
        raw_y1 = int(min(d["box"][1] for d in cars) * scale_y)
        raw_x2 = int(max(d["box"][2] for d in cars) * scale_x)
        raw_y2 = int(max(d["box"][3] for d in cars) * scale_y)

        pad_x = int(15*scale_x); pad_y = int(10*scale_y)
        thr_x = int(150*scale_x); thr_y = int(80*scale_y)
        x1 = 0      if raw_x1<thr_x            else max(0,      raw_x1-pad_x)
        x2 = orig_w if (orig_w-raw_x2)<thr_x   else min(orig_w, raw_x2+pad_x)
        y1 = 0      if raw_y1<thr_y            else max(0,      raw_y1-pad_y)
        y2 = orig_h if (orig_h-raw_y2)<thr_y   else min(orig_h, raw_y2+pad_y)

        x1, y1, x2, y2 = refine_car_bbox(img_orig, x1, y1, x2, y2)
        car_crop = img_orig[y1:y2, x1:x2]
        if car_crop.size == 0:
            return jsonify({"error": "invalid crop"}), 400
        crop_h, crop_w = car_crop.shape[:2]

        # ===== LUMIÈRES + ORIENTATION + ANGLE =====
        lights                            = detect_lights(car_crop)
        rear_side, front_side, facing, fr_log = detect_front_rear(lights)
        angle                             = estimate_angle(lights, crop_w, crop_h, facing)
        fr_log.append(f"Angle estime: {angle}°")

        # ===== ZONES =====
        zones, zone_decision = build_zones(
            crop_w, crop_h, angle, rear_side, front_side, facing, lights
        )
        fr_log.append(zone_decision)

        # ===== MASQUE + ESPACES COLORIMÉTRIQUES =====
        hsv_full  = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
        lab_full  = cv2.cvtColor(car_crop, cv2.COLOR_BGR2LAB)
        gray_full = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
        mask_body = build_body_mask(car_crop, hsv_full)

        # ===== RÉFÉRENCE GLOBALE LAB (percentiles 10-90) =====
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

        nat_std_a = max(float(np.std(vl_ref[:, 1])), 1.0)
        nat_std_b = max(float(np.std(vl_ref[:, 2])), 1.0)

        # Texture ref sur masque seulement
        lap_full  = cv2.Laplacian(gray_full, cv2.CV_64F)
        ref_tex   = float(np.var(lap_full[mask_body > 0]))
        ref_tex   = max(ref_tex, 1.0)

        # =============================================
        # SEUILS ADAPTATIFS SELON LA COULEUR
        #
        # Le problème central : une voiture NOIRE a
        # une variabilité naturelle plus faible qu'une
        # voiture ARGENT. Les mêmes seuils fixes
        # donnent trop de faux positifs sur les noires.
        #
        # Solution : on ajuste les seuils selon la
        # luminosité de référence (L de LAB) :
        #
        # Voiture sombre (L < 80)  → seuils plus hauts
        #   car les différences de teinte sont comprimées
        # Voiture claire (L > 160) → seuils normaux
        # Voiture argent (L 80-160)→ seuils intermédiaires
        #
        # Seuil ROUGE  : score > thresh_red  → "Peinture refaite!"
        # Seuil ORANGE : score > thresh_susp → "Variation suspecte"
        # =============================================
        med_L = float(np.median(vl_ref[:, 0]))

        if med_L < 70:
            # Voiture très sombre (noir, bleu foncé)
            thresh_red  = 3.5
            thresh_susp = 2.0
        elif med_L < 110:
            # Voiture sombre (gris foncé, bordeaux)
            thresh_red  = 2.8
            thresh_susp = 1.6
        elif med_L < 150:
            # Voiture intermédiaire (gris, rouge)
            thresh_red  = 2.2
            thresh_susp = 1.2
        else:
            # Voiture claire (blanc, beige, argent)
            thresh_red  = 1.8
            thresh_susp = 1.0

        # ===== DESSIN =====
        final_img  = img_orig.copy()
        thick_box  = max(3, int(4*min(scale_x, scale_y)))
        thick_line = max(1, int(1*min(scale_x, scale_y)))
        font_big   = max(0.55, 0.58*min(scale_x, scale_y))
        font_med   = max(0.42, 0.44*min(scale_x, scale_y))
        font_thick = max(2, int(2*min(scale_x, scale_y)))

        cv2.rectangle(final_img, (x1, y1), (x2, y2), (220, 220, 220), thick_line)

        results_zones = []
        detected      = 0

        for idx, zone in enumerate(zones, start=1):
            poly_local  = zone["poly"]
            poly_global = np.array(
                [[x1+p[0], y1+p[1]] for p in poly_local],
                dtype=np.int32
            )

            # Couleur LAB de la zone
            zone_color, px_count = get_poly_color_lab(
                lab_full, mask_body, poly_local
            )

            # Texture de la zone sur masque seulement
            h_c, w_c  = lab_full.shape[:2]
            poly_mask = np.zeros((h_c, w_c), dtype=np.uint8)
            cv2.fillPoly(poly_mask,
                         [np.array(poly_local, dtype=np.int32)], 255)
            comb      = cv2.bitwise_and(poly_mask, mask_body)
            lap_z     = lap_full[comb > 0]
            zone_tex  = float(np.var(lap_z)) if len(lap_z) > 100 else ref_tex

            if zone_color is None:
                color_rect  = (150, 150, 150)
                label_score = "N/A"
                score       = 0.0
                verdict     = "Non analysable"
            else:
                score = compute_score(
                    zone_color, ref_color,
                    nat_std_a, nat_std_b,
                    zone_tex, ref_tex
                )
                label_score = f"{score:.1f}"

                if score > thresh_red:
                    color_rect = (0, 0, 255)
                    verdict    = "Peinture refaite!"
                    detected  += 1
                elif score > thresh_susp:
                    color_rect = (0, 165, 255)
                    verdict    = "Variation suspecte"
                    detected  += 1
                else:
                    color_rect = (0, 210, 0)
                    verdict    = "OK"

            # Remplissage semi-transparent
            overlay = final_img.copy()
            cv2.fillPoly(overlay, [poly_global], color_rect)
            cv2.addWeighted(overlay, 0.22, final_img, 0.78, 0, final_img)

            # Contour
            cv2.polylines(final_img, [poly_global], True, color_rect, thick_box)

            # Cercle numéroté
            cx = int(np.mean(poly_global[:, 0]))
            cy = int(np.mean(poly_global[:, 1]))
            radius = max(18, int(20*min(scale_x, scale_y)))
            cv2.circle(final_img, (cx+2, cy+2), radius, (0, 0, 0), -1)
            cv2.circle(final_img, (cx, cy),   radius, color_rect, -1)
            cv2.circle(final_img, (cx, cy),   radius, (255, 255, 255), 2)
            num_txt = str(idx)
            (tw, th), _ = cv2.getTextSize(
                num_txt, cv2.FONT_HERSHEY_SIMPLEX, font_big*1.3, font_thick+1
            )
            cv2.putText(final_img, num_txt,
                        (cx-tw//2, cy+th//2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_big*1.3, (255, 255, 255), font_thick+1)

            # Étiquette repositionnée si hors image
            top_pt  = poly_global[poly_global[:, 1].argmin()]
            lbl_x   = max(5, int(top_pt[0]))
            lbl_y   = max(20, int(top_pt[1]) - 10)
            lbl_txt = f"{idx}. {zone['name']}  S:{label_score}"
            (lw, lh), _ = cv2.getTextSize(
                lbl_txt, cv2.FONT_HERSHEY_SIMPLEX, font_med, font_thick
            )
            lbl_x = min(lbl_x, orig_w - lw - 10)
            cv2.rectangle(final_img,
                          (lbl_x-4, lbl_y-lh-6),
                          (lbl_x+lw+6, lbl_y+4),
                          (0, 0, 0), -1)
            cv2.putText(final_img, lbl_txt, (lbl_x, lbl_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_med, (255, 255, 255), font_thick)

            results_zones.append({
                "idx":     idx,
                "zone":    zone["name"],
                "score":   score,
                "pixels":  px_count,
                "verdict": verdict,
                "polygon": poly_global.tolist()
            })

        # ===== SCORE GLOBAL =====
        scores      = [z["score"] for z in results_zones if z["score"] > 0]
        score_raw   = float(np.mean(scores)) if scores else 0.0
        score_100   = min(int(score_raw * 30), 100)

        if   score_raw > thresh_red:  result = "Difference importante - repeinture probable"
        elif score_raw > thresh_susp: result = "Legeres variations detectees"
        else:                         result = "Peinture homogene (OK)"

        analysed_name = "analysed_" + filename
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, analysed_name), final_img)
        if os.path.exists(resized_path):
            os.remove(resized_path)

        return jsonify({
            "yolo":            yolo_result,
            "angle_estime":    angle,
            "score":           score_100,
            "score_raw":       round(score_raw, 2),
            "result":          result,
            "zones":           results_zones,
            "zones_detected":  detected,
            "facing":          facing,
            "rear_side":       rear_side,
            "front_side":      front_side,
            "orientation_log": fr_log,
            "lights":          lights,
            "seuils": {
                "rouge":   thresh_red,
                "suspect": thresh_susp,
                "med_L":   round(med_L, 1)
            },
            "calibration": {
                "nat_std_a":   round(nat_std_a, 1),
                "nat_std_b":   round(nat_std_b, 1),
                "ref_L":       round(ref_color[0], 1),
                "ref_a":       round(ref_color[1], 1),
                "ref_b":       round(ref_color[2], 1),
                "ref_texture": round(ref_tex, 1)
            },
            "image_size":   {"width": orig_w, "height": orig_h},
            "image_result": analysed_name,
            "image_url":    request.host_url + "uploads/" + analysed_name
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
