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
# YOLO API CALL  (detection + marque/modele si dispo)
# =========================
def call_yolo(image_path):
    url = "https://warrdi.com/pytho/detect"
    try:
        ext  = os.path.splitext(image_path)[1].lower()
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
        with open(image_path, "rb") as f:
            files   = {"image": (os.path.basename(image_path), f, mime)}
            headers = {"Accept": "application/json"}
            r = requests.post(url, files=files, headers=headers, timeout=25)
        if r.status_code == 200:
            return r.json()
        return {"error": "YOLO failed", "status": r.status_code}
    except Exception as e:
        return {"error": "YOLO exception", "details": str(e)}


def call_car_make_model(image_path):
    """
    Tentative d'appel a un endpoint dedie marque/modele.
    Si l'endpoint n'existe pas, on retourne Unknown sans bloquer.
    L'endpoint attendu doit renvoyer {"make": "...", "model": "...", "confidence": 0.x}
    """
    url = "https://warrdi.com/pytho/car_make_model"
    try:
        ext  = os.path.splitext(image_path)[1].lower()
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
        with open(image_path, "rb") as f:
            files = {"image": (os.path.basename(image_path), f, mime)}
            r = requests.post(url, files=files, timeout=15)
        if r.status_code == 200:
            data = r.json()
            return {
                "make":       data.get("make",  "Unknown"),
                "model":      data.get("model", "Unknown"),
                "confidence": data.get("confidence", 0.0)
            }
    except Exception:
        pass
    return {"make": "Unknown", "model": "Unknown", "confidence": 0.0}


@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/")
def home():
    return jsonify({"status": "OK", "message": "GARAGE PRO V7"})


# =========================
# COULEUR HSV MEDIANE dans un POLYGONE
# =========================
def get_poly_color(hsv_img, body_mask, polygon):
    h, w = hsv_img.shape[:2]
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [np.array(polygon, dtype=np.int32)], 255)
    combined = cv2.bitwise_and(poly_mask, body_mask)
    valid = hsv_img[combined > 0]
    if len(valid) < 80:
        return None, 0, None
    med = np.array([
        float(np.median(valid[:, 0])),
        float(np.median(valid[:, 1])),
        float(np.median(valid[:, 2]))
    ])
    stats = {
        "std_h": float(np.std(valid[:, 0])),
        "std_s": float(np.std(valid[:, 1])),
        "std_v": float(np.std(valid[:, 2])),
    }
    return med, len(valid), stats


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
    PAD = 8
    new_x1 = max(0,            x1 + int(valid_cols[0])  - PAD)
    new_x2 = min(img.shape[1], x1 + int(valid_cols[-1]) + PAD)
    new_y1 = max(0,            y1 + int(valid_rows[0])  - PAD)
    new_y2 = min(img.shape[0], y1 + int(valid_rows[-1]) + PAD)
    if (new_x2 - new_x1) < 100 or (new_y2 - new_y1) < 80:
        return x1, y1, x2, y2
    return new_x1, new_y1, new_x2, new_y2


# =========================
# DETECTION FEUX
#
# CORRECTION CRITIQUE :
# - Bande verticale resserree (40%-80% au lieu de 30%-92%) pour
#   exclure le mur/le ciel/le toit
# - Plage ROUGE elargie en saturation (les feux LED sont satures meme
#   eteints, le rouge sature est tres rare dans une scene exterieure)
# - Plage BLANC/JAUNE plus stricte (S maxi reduit, V mini eleve)
#   pour ne pas confondre un mur beige avec un phare
# =========================
def detect_lights(car_crop):
    h, w = car_crop.shape[:2]
    band_w  = int(w * 0.28)
    # Bande resserree autour de la hauteur des feux uniquement
    feux_y1 = int(h * 0.40)
    feux_y2 = int(h * 0.78)

    left_feux  = car_crop[feux_y1:feux_y2, 0:band_w]
    right_feux = car_crop[feux_y1:feux_y2, w - band_w:w]
    left_hsv   = cv2.cvtColor(left_feux,  cv2.COLOR_BGR2HSV)
    right_hsv  = cv2.cvtColor(right_feux, cv2.COLOR_BGR2HSV)

    def count_red(hsv):
        # Rouge sature uniquement (S>=90, V>=70) — les murs beiges sont S<60
        m1 = cv2.inRange(hsv, (0,   90, 70),  (12,  255, 255))
        m2 = cv2.inRange(hsv, (165, 90, 70),  (180, 255, 255))
        return int(cv2.countNonZero(cv2.bitwise_or(m1, m2)))

    def count_white(hsv):
        # Blanc TRES brillant uniquement (V>=210, S<=60) — exclut murs beiges
        white = cv2.inRange(hsv, (0, 0, 210), (180, 60, 255))
        # Jaune des clignotants : sature ET brillant
        yellow = cv2.inRange(hsv, (18, 130, 190), (35, 255, 255))
        return int(cv2.countNonZero(cv2.bitwise_or(white, yellow)))

    rl = count_red(left_hsv)
    rr = count_red(right_hsv)
    wl = count_white(left_hsv)
    wr = count_white(right_hsv)
    band_area = band_w * (feux_y2 - feux_y1)

    return {
        "red_left":    rl,
        "red_right":   rr,
        "white_left":  wl,
        "white_right": wr,
        "red_tot":     rl + rr,
        "white_tot":   wl + wr,
        "band_area":   band_area,
        "red_left_ratio":  rl / max(band_area, 1),
        "red_right_ratio": rr / max(band_area, 1),
        "whi_left_ratio":  wl / max(band_area, 1),
        "whi_right_ratio": wr / max(band_area, 1),
    }


# =========================
# DETECTION AVANT/ARRIERE
#
# PRIORITE AU ROUGE :
# Le rouge sature est rare dans une scene exterieure. S'il est
# present en quantite significative, c'est forcement un feu arriere.
# Le "blanc" peut etre un mur, un trottoir, du ciel, etc.
# =========================
def detect_front_rear(lights):
    log = []
    rl, rr = lights["red_left"],   lights["red_right"]
    wl, wr = lights["white_left"], lights["white_right"]
    red_tot   = rl + rr
    white_tot = wl + wr
    band_area = lights["band_area"]

    # Seuil minimal pour qu'un feu rouge soit considere "present"
    red_thr   = max(120, int(band_area * 0.003))
    white_thr = max(400, int(band_area * 0.020))   # exigence plus forte

    # ----- PRIORITE 1 : rouge significatif = arriere -----
    if red_tot >= red_thr:
        facing = "rear"
        log.append(f"facing=REAR (rouge significatif tot={red_tot} >= seuil {red_thr})")
    # ----- PRIORITE 2 : blanc tres dominant = avant -----
    elif white_tot >= white_thr and white_tot > red_tot * 3:
        facing = "front"
        log.append(f"facing=FRONT (blanc dominant tot={white_tot})")
    else:
        facing = "side"
        log.append(f"facing=SIDE (red={red_tot} white={white_tot})")

    # Cote du feu rouge dominant
    if rl > rr * 1.25:
        rear_side = "left"
    elif rr > rl * 1.25:
        rear_side = "right"
    else:
        rear_side = None

    # Cote des phares dominants
    if wl > wr * 1.25:
        front_side = "left"
    elif wr > wl * 1.25:
        front_side = "right"
    else:
        front_side = None

    # Reconciliation
    if rear_side and not front_side:
        front_side = "right" if rear_side == "left" else "left"
    if front_side and not rear_side:
        rear_side = "right" if front_side == "left" else "left"
    if not rear_side and not front_side:
        rear_side, front_side = "right", "left"
        log.append("Fallback: arriere=droite")

    log.append(f"rouge G={rl} D={rr} | blanc G={wl} D={wr}")
    log.append(f"rear_side={rear_side} front_side={front_side}")
    return rear_side, front_side, facing, log


# =========================
# ESTIMER L'ANGLE DE VUE
# =========================
def estimate_angle(lights, crop_w, crop_h, facing):
    rl, rr = lights["red_left"],   lights["red_right"]
    wl, wr = lights["white_left"], lights["white_right"]

    if facing == "rear":
        big = max(rl, rr); sml = min(rl, rr)
    elif facing == "front":
        big = max(wl, wr); sml = min(wl, wr)
    else:
        big = max(rl + wl, rr + wr); sml = min(rl + wl, rr + wr)

    if big == 0:
        return 45.0

    sym_ratio = big / max(sml, 1)

    if sym_ratio >= 8.0:
        angle = 10.0
    elif sym_ratio >= 4.0:
        angle = 10.0 + (8.0 - sym_ratio) / 4.0 * 30.0
    elif sym_ratio >= 2.0:
        angle = 40.0 + (4.0 - sym_ratio) / 2.0 * 25.0
    elif sym_ratio >= 1.3:
        angle = 65.0 + (2.0 - sym_ratio) / 0.7 * 20.0
    else:
        angle = 87.0

    ratio_wh = crop_w / max(crop_h, 1)
    if ratio_wh > 1.6:
        angle = min(angle, 35.0)
    elif ratio_wh < 0.9:
        angle = max(angle, 60.0)

    return round(angle, 1)


# =========================
# POLYGONE TRAPEZE PERSPECTIF
# =========================
def make_poly(crop_w, crop_h, xA, xB, top_base, bot_base, persp, tilt_dir):
    drop = 0.08
    def top_y(x):
        f = x / max(1, crop_w)
        return top_base + (persp * drop * f if tilt_dir > 0
                           else persp * drop * (1 - f))
    def bot_y(x):
        f = x / max(1, crop_w)
        return bot_base - (persp * drop * f if tilt_dir > 0
                           else persp * drop * (1 - f))
    return [
        (xA, int(top_y(xA) * crop_h)),
        (xB, int(top_y(xB) * crop_h)),
        (xB, int(bot_y(xB) * crop_h)),
        (xA, int(bot_y(xA) * crop_h)),
    ]


# =========================
# CONSTRUIRE LES ZONES
# =========================
def build_zones(crop_w, crop_h, angle, rear_side, front_side, facing, lights):
    rl, rr = lights["red_left"],   lights["red_right"]
    wl, wr = lights["white_left"], lights["white_right"]

    if facing == "rear":
        near_side = "left" if rl > rr else "right"
    elif facing == "front":
        near_side = "left" if wl > wr else "right"
    else:
        near_side = "left" if (rl+wl) > (rr+wr) else "right"

    far_side = "right" if near_side == "left" else "left"

    if angle <= 10:
        persp = 0.0
    elif angle <= 55:
        persp = 0.6 * min(1.0, (angle - 10) / 45.0)
    elif angle <= 80:
        persp = 0.6 * max(0.0, 1.0 - (angle - 55) / 25.0)
    else:
        persp = 0.0

    tilt_dir = +1 if far_side == "right" else -1
    top_base = 0.20
    bot_base = 0.88

    is_rear  = (facing == "rear") or (
        facing == "side" and (rl + rr) >= (wl + wr)
    )
    panel    = "Coffre"     if is_rear else "Capot"
    pc_label = "Pare-ch.AR" if is_rear else "Pare-ch.AV"

    def zone(name, a, b):
        return {
            "name": name,
            "poly": make_poly(
                crop_w, crop_h,
                int(a * crop_w), int(b * crop_w),
                top_base, bot_base, persp, tilt_dir
            )
        }

    log_label = ""

    # 0-25° : PROFIL
    if angle <= 25:
        log_label = f"PROFIL (angle={angle}°) near={near_side} {'AR' if is_rear else 'AV'}"
        if near_side == "right":
            if is_rear:
                return [
                    zone("Aile AV",  0.00, 0.20),
                    zone("Porte AV", 0.20, 0.48),
                    zone("Porte AR", 0.48, 0.78),
                    zone("Aile AR",  0.78, 1.00),
                ], log_label
            else:
                return [
                    zone("Aile AR",  0.00, 0.20),
                    zone("Porte AR", 0.20, 0.48),
                    zone("Porte AV", 0.48, 0.78),
                    zone("Aile AV",  0.78, 1.00),
                ], log_label
        else:
            if is_rear:
                return [
                    zone("Aile AR",  0.00, 0.22),
                    zone("Porte AR", 0.22, 0.52),
                    zone("Porte AV", 0.52, 0.80),
                    zone("Aile AV",  0.80, 1.00),
                ], log_label
            else:
                return [
                    zone("Aile AV",  0.00, 0.22),
                    zone("Porte AV", 0.22, 0.52),
                    zone("Porte AR", 0.52, 0.80),
                    zone("Aile AR",  0.80, 1.00),
                ], log_label

    # 25-55° : 3/4 LEGER
    elif angle <= 55:
        log_label = f"3/4 LEGER (angle={angle}°) near={near_side} {'AR' if is_rear else 'AV'}"
        if near_side == "right":
            if is_rear:
                return [
                    zone("Porte AV", 0.00, 0.22),
                    zone("Porte AR", 0.22, 0.50),
                    zone("Aile AR",  0.50, 0.72),
                    zone(pc_label,   0.72, 1.00),
                ], log_label
            else:
                return [
                    zone("Porte AR", 0.00, 0.22),
                    zone("Porte AV", 0.22, 0.50),
                    zone("Aile AV",  0.50, 0.72),
                    zone(pc_label,   0.72, 1.00),
                ], log_label
        else:
            if is_rear:
                return [
                    zone(pc_label,   0.00, 0.28),
                    zone("Aile AR",  0.28, 0.50),
                    zone("Porte AR", 0.50, 0.78),
                    zone("Porte AV", 0.78, 1.00),
                ], log_label
            else:
                return [
                    zone(pc_label,   0.00, 0.28),
                    zone("Aile AV",  0.28, 0.50),
                    zone("Porte AV", 0.50, 0.78),
                    zone("Porte AR", 0.78, 1.00),
                ], log_label

    # 55-80° : 3/4 MARQUE
    elif angle <= 80:
        log_label = f"3/4 MARQUE (angle={angle}°) near={near_side} {'AR' if is_rear else 'AV'}"
        if near_side == "right":
            return [
                zone("Aile AR" if is_rear else "Aile AV", 0.42, 0.68),
                zone(panel,                               0.68, 0.85),
                zone(pc_label,                            0.85, 1.00),
            ], log_label
        else:
            return [
                zone(pc_label,                            0.00, 0.15),
                zone(panel,                               0.15, 0.32),
                zone("Aile AR" if is_rear else "Aile AV", 0.32, 0.58),
            ], log_label

    # 80-90° : FACE / DOS
    else:
        log_label = f"FACE/DOS (angle={angle}°) {'AR' if is_rear else 'AV'}"
        if is_rear:
            return [
                zone("Aile AR G", 0.00, 0.20),
                zone(pc_label,    0.20, 0.55),
                zone(panel,       0.45, 0.80),
                zone("Aile AR D", 0.80, 1.00),
            ], log_label
        else:
            return [
                zone("Aile AV G", 0.00, 0.20),
                zone(pc_label,    0.20, 0.55),
                zone(panel,       0.45, 0.80),
                zone("Aile AV D", 0.80, 1.00),
            ], log_label


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

        # ===== MARQUE / MODELE =====
        car_info = call_car_make_model(resized_path)
        # Si YOLO renvoie deja make/model, on les utilise
        if yolo_result.get("make"):
            car_info["make"]  = yolo_result.get("make", car_info["make"])
        if yolo_result.get("model"):
            car_info["model"] = yolo_result.get("model", car_info["model"])

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

        lights = detect_lights(car_crop)
        rear_side, front_side, facing, fr_log = detect_front_rear(lights)
        angle = estimate_angle(lights, crop_w, crop_h, facing)
        fr_log.append(f"Angle estime: {angle}°")

        zones, zone_decision = build_zones(
            crop_w, crop_h, angle, rear_side, front_side, facing, lights
        )
        fr_log.append(f"Decision zones: {zone_decision}")

        hsv_full  = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
        mask_dark = cv2.inRange(hsv_full, (0, 0, 0),   (180, 255, 45))
        mask_sky  = cv2.inRange(hsv_full, (0, 0, 210), (180, 18, 255))
        mask_body = cv2.bitwise_not(cv2.bitwise_or(mask_dark, mask_sky))
        kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_CLOSE, kernel)

        all_valid = hsv_full[mask_body > 0]
        if len(all_valid) < 100:
            return jsonify({"error": "No body pixels found"}), 400

        ref_color = np.array([
            float(np.median(all_valid[:, 0])),
            float(np.median(all_valid[:, 1])),
            float(np.median(all_valid[:, 2]))
        ])

        final_img  = img_orig.copy()
        thick_box  = max(3, int(4 * min(scale_x, scale_y)))
        thick_line = max(1, int(1 * min(scale_x, scale_y)))
        font_big   = max(0.55, 0.58 * min(scale_x, scale_y))
        font_med   = max(0.42, 0.44 * min(scale_x, scale_y))
        font_thick = max(2, int(2 * min(scale_x, scale_y)))

        cv2.rectangle(final_img, (x1, y1), (x2, y2), (220, 220, 220), thick_line)

        # En-tete avec marque/modele + orientation
        header = f"{car_info['make']} {car_info['model']} | {'AR' if facing=='rear' else ('AV' if facing=='front' else 'COTE')} | {angle}°"
        (hw, hh), _ = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, font_med * 1.2, font_thick)
        cv2.rectangle(final_img, (5, 5), (15 + hw, 20 + hh), (0, 0, 0), -1)
        cv2.putText(final_img, header, (10, 15 + hh),
                    cv2.FONT_HERSHEY_SIMPLEX, font_med * 1.2,
                    (255, 255, 255), font_thick)

        results_zones = []
        detected = 0

        for idx, zone in enumerate(zones, start=1):
            poly_local  = zone["poly"]
            poly_global = np.array(
                [[x1 + p[0], y1 + p[1]] for p in poly_local], dtype=np.int32
            )

            zone_color, px_count, stats = get_poly_color(hsv_full, mask_body, poly_local)

            if zone_color is None:
                color_rect, label_score, diff, verdict = (150,150,150), "N/A", 0.0, "Non analysable"
                std_h = std_s = std_v = 0.0
            else:
                # ecart de TEINTE (H) seulement -> plus fiable que la norme HSV
                diff_h = abs(float(zone_color[0]) - float(ref_color[0]))
                diff_h = min(diff_h, 180.0 - diff_h)  # H est circulaire
                diff   = diff_h
                std_h  = stats["std_h"]
                std_s  = stats["std_s"]
                std_v  = stats["std_v"]

                # CRITERES COMBINES :
                #   - diff (mediane H)  : couleur globale
                #   - std_s             : empreinte chimique de la peinture
                #   - std_v             : texture / mastic / grain
                suspect_color   = 2<= diff  <= 4
                suspect_satur   = std_s > 6
                suspect_texture = std_v < 15

                if suspect_color and suspect_satur:
                    color_rect, verdict = (0, 0, 255),   "Peinture refaite!";  detected += 1
                elif suspect_color or (suspect_satur and suspect_texture):
                    color_rect, verdict = (0, 165, 255), "Variation suspecte"; detected += 1
                else:
                    color_rect, verdict = (0, 210, 0),   "OK"
                label_score = f"H{int(diff)}/S{int(std_s)}/V{int(std_v)}"

            overlay = final_img.copy()
            cv2.fillPoly(overlay, [poly_global], color_rect)
            cv2.addWeighted(overlay, 0.22, final_img, 0.78, 0, final_img)
            cv2.polylines(final_img, [poly_global], True, color_rect, thick_box)

            cx = int(np.mean(poly_global[:, 0]))
            cy = int(np.mean(poly_global[:, 1]))
            radius = max(18, int(20 * min(scale_x, scale_y)))
            cv2.circle(final_img, (cx + 2, cy + 2), radius, (0, 0, 0), -1)
            cv2.circle(final_img, (cx, cy), radius, color_rect, -1)
            cv2.circle(final_img, (cx, cy), radius, (255, 255, 255), 2)
            num_txt = str(idx)
            (tw, th), _ = cv2.getTextSize(num_txt, cv2.FONT_HERSHEY_SIMPLEX, font_big * 1.3, font_thick + 1)
            cv2.putText(final_img, num_txt, (cx - tw // 2, cy + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_big * 1.3,
                        (255, 255, 255), font_thick + 1)

            top_pt = poly_global[poly_global[:, 1].argmin()]
            lbl_x  = max(5, int(top_pt[0]))
            lbl_y  = max(20, int(top_pt[1]) - 10)
            label_full = f"{idx}. {zone['name']}  E:{label_score}"
            (lw, lh), _ = cv2.getTextSize(label_full, cv2.FONT_HERSHEY_SIMPLEX, font_med, font_thick)
            lbl_x = min(lbl_x, orig_w - lw - 10)
            cv2.rectangle(final_img, (lbl_x - 4, lbl_y - lh - 6),
                          (lbl_x + lw + 6, lbl_y + 4), (0, 0, 0), -1)
            cv2.putText(final_img, label_full, (lbl_x, lbl_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_med,
                        (255, 255, 255), font_thick)

            results_zones.append({
                "idx": idx, "zone": zone["name"], "diff": round(diff, 1),
                "std_h": round(std_h, 2), "std_s": round(std_s, 2), "std_v": round(std_v, 2),
                "pixels": px_count, "verdict": verdict,
                "polygon": poly_global.tolist()
            })

        diffs       = [z["diff"] for z in results_zones if z["diff"] > 0]
        final_score = min(int(np.mean(diffs)) if diffs else 0, 100)
        if   final_score < 10: result = "Peinture homogene (OK)"
        elif final_score < 28: result = "Legeres variations detectees"
        else:                  result = "Difference importante - repeinture probable"

        analysed_name = "analysed_" + filename
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, analysed_name), final_img)
        if os.path.exists(resized_path):
            os.remove(resized_path)

        return jsonify({
            "yolo":            yolo_result,
            "car":             car_info,
            "angle_estime":    angle,
            "score":           final_score,
            "result":          result,
            "zones":           results_zones,
            "zones_detected":  detected,
            "facing":          facing,
            "rear_side":       rear_side,
            "front_side":      front_side,
            "orientation_log": fr_log,
            "lights":          lights,
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
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
