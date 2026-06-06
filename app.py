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
    return jsonify({"status": "OK", "message": "GARAGE PRO V6 (3D zones)"})


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
        return None, 0
    return np.array([
        float(np.median(valid[:, 0])),
        float(np.median(valid[:, 1])),
        float(np.median(valid[:, 2]))
    ]), len(valid)


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
# =========================
def detect_lights(car_crop):
    h, w = car_crop.shape[:2]
    band_w  = int(w * 0.22)
    feux_y1 = int(h * 0.38)
    feux_y2 = int(h * 0.88)

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

    return {
        "red_left":    count_red(left_hsv),
        "red_right":   count_red(right_hsv),
        "white_left":  count_white(left_hsv),
        "white_right": count_white(right_hsv),
        "band_area":   band_w * (feux_y2 - feux_y1)
    }


# =========================
# DETECTION AVANT/ARRIERE EN PREMIER (basé sur les feux)
#  - rear_side  = cote (left/right) ou se trouve l'arriere
#  - front_side = cote oppose
#  - facing     = "rear" (on voit surtout l'arriere) / "front" / "side"
# =========================
def detect_front_rear(lights):
    log = []
    rl, rr = lights["red_left"], lights["red_right"]
    wl, wr = lights["white_left"], lights["white_right"]
    red_tot   = rl + rr
    white_tot = wl + wr

    # cote du rouge dominant
    if rl > rr * 1.25:
        rear_side = "left"
    elif rr > rl * 1.25:
        rear_side = "right"
    else:
        rear_side = None

    # cote du blanc dominant
    if wl > wr * 1.25:
        front_side = "left"
    elif wr > wl * 1.25:
        front_side = "right"
    else:
        front_side = None

    # Si rouge tres dominant globalement -> on voit l'arriere
    if red_tot > white_tot * 1.4 and red_tot > 200:
        facing = "rear"
    elif white_tot > red_tot * 1.4 and white_tot > 200:
        facing = "front"
    else:
        facing = "side"

    # Réconciliation: si on a un cote pour l'un, l'autre est oppose
    if rear_side and not front_side:
        front_side = "right" if rear_side == "left" else "left"
    if front_side and not rear_side:
        rear_side = "right" if front_side == "left" else "left"
    if not rear_side and not front_side:
        # fallback : on suppose arriere a droite
        rear_side, front_side = "right", "left"
        log.append("Pas de feux clairs -> fallback arriere=droite")

    log.append(f"Feux rouge G={rl} D={rr} | blanc G={wl} D={wr}")
    log.append(f"Total rouge={red_tot} blanc={white_tot} -> facing={facing}")
    log.append(f"rear_side={rear_side} front_side={front_side}")
    return rear_side, front_side, facing, log


# =========================
# CONSTRUCTION DE POLYGONES 3D (trapezes perspective)
#
# On modelise la voiture comme une bande horizontale qui suit la perspective:
#  - top_y et bot_y varient lineairement selon x (la voiture s'eloigne)
#  - le cote "loin" (vers l'avant fuyant) est plus etroit en hauteur
#
# perspective_factor : 0 = pas d'effet 3D, 0.2 = leger, 0.45 = vue 3/4 marquee
# tilt_dir : +1 si le cote droit s'enfonce (top descend a droite),
#            -1 si le cote gauche s'enfonce
# =========================
def make_poly(crop_w, crop_h, xA, xB, top_base, bot_base, perspective, tilt_dir):
    # variation verticale du toit/bas selon position x (en fraction de crop_h)
    def y_at(x_frac, base, drop):
        # base in [0,1], drop = amplitude
        if tilt_dir > 0:
            # cote droit s'enfonce -> top descend a droite, bot remonte a droite
            offset_top = perspective * drop * x_frac          # +y a droite
            offset_bot = -perspective * drop * x_frac
        else:
            # cote gauche s'enfonce
            offset_top = perspective * drop * (1 - x_frac)
            offset_bot = -perspective * drop * (1 - x_frac)
        return base + offset_top, (1 - base) + offset_bot   # not used

    drop = 0.10  # amplitude verticale (fraction de crop_h)

    def top_y(x):
        f = x / max(1, crop_w)
        if tilt_dir > 0:
            return top_base + perspective * drop * f
        else:
            return top_base + perspective * drop * (1 - f)

    def bot_y(x):
        f = x / max(1, crop_w)
        if tilt_dir > 0:
            return bot_base - perspective * drop * f
        else:
            return bot_base - perspective * drop * (1 - f)

    pts = [
        (xA, int(top_y(xA) * crop_h)),
        (xB, int(top_y(xB) * crop_h)),
        (xB, int(bot_y(xB) * crop_h)),
        (xA, int(bot_y(xA) * crop_h)),
    ]
    return pts


# =========================
# DECIDE ZONES par ANGLE
# IMPORTANT: l'orientation (avant/arriere) est decidee AVANT par detect_front_rear
# Ici on utilise rear_side/front_side/facing + angle pour decider quoi scanner
# Les zones sont retournees sous forme de polygones 3D (trapezes)
# =========================
def build_zones(crop_w, crop_h, angle, rear_side, front_side, facing, lights):
    """
    Logique unifiee 0-180 deg, basee sur la MOITIE DU CORPS de la voiture.

    PRIORITE 1 : orientation decidee par les feux
       - grand feu rouge  -> on voit l'ARRIERE (facing = rear)
       - grand feu blanc  -> on voit l'AVANT  (facing = front)
       Le cote (gauche/droite) du feu dominant donne le cote PROCHE camera.

    PRIORITE 2 : l'angle decide quels segments sont visibles
       0-10   : profil pur    -> 4 zones laterales (ailes + portes)
       10-30  : 3/4 leger     -> 4 zones cote dominant (parchoq + aile + malle/capot + porte)
       30-70  : 3/4 marque    -> parchoq + malle/capot + aile
       70-110 : face/arriere  -> parchoq + malle/capot uniquement
       110-150: 3/4 inverse   -> parchoq + aile + porte AR + porte AV (cote dominant)
       150-180: profil inverse-> 4 zones laterales

    rear_side / front_side indiquent ou est l'AR / AV dans l'image (left/right).
    """
    is_rear = (facing == "rear") or (
        facing == "side" and
        (lights["red_left"] + lights["red_right"]) >=
        (lights["white_left"] + lights["white_right"])
    )
    # cote PROCHE camera = cote du feu dominant
    near_side = rear_side if is_rear else front_side
    if near_side is None:
        near_side = "right"

    # tilt_dir : le cote OPPOSE au feu dominant s'enfonce dans la perspective
    tilt_dir = +1 if near_side == "left" else -1

    if angle <= 5 or (85 <= angle <= 95):
        persp = 0.0
    elif angle <= 30:
        persp = 0.5 * (angle / 30.0)
    elif angle <= 60:
        persp = 0.5 + 0.4 * ((angle - 30) / 30.0)
    elif angle <= 85:
        persp = max(0.0, 0.9 - (angle - 60) * 0.03)
    elif angle <= 120:
        persp = 0.5 * ((angle - 95) / 25.0) if angle > 95 else 0.0
    elif angle <= 150:
        persp = 0.5 + 0.4 * ((angle - 120) / 30.0)
    else:
        persp = max(0.0, 0.9 - (angle - 150) * 0.03)

    top_base = 0.22
    bot_base = 0.85
    orient = "AV" if is_rear else "AR"
    panel  = "Malle" if is_rear else "Capot"

    def pack(zones, label):
        out = [{
            "name": n,
            "poly": make_poly(crop_w, crop_h, int(a*crop_w), int(b*crop_w),
                              top_base, bot_base, persp, tilt_dir)
        } for (n, a, b) in zones]
        return out, label

    # =============== 0-10  PROFIL PUR ===============
    if angle <= 150:
        if near_side == "right":
            zones = [
                (f"Aile {'AV' if is_rear else 'AR'}",  0.00, 0.18),
                (f"Porte {'AV' if is_rear else 'AR'}",0.18, 0.50),
                (f"Porte {orient}", 0.50, 0.82),
                (f"Aile {orient}",  0.82, 1.00),
            ]
        else:
            zones = [
                (f"Aile {orient}",  0.00, 0.18),
                (f"Porte {orient}", 0.18, 0.50),
                (f"Porte {'AV' if is_rear else 'AR'}",0.50, 0.82),
                (f"Aile {'AV' if is_rear else 'AR'}",  0.82, 1.00),
            ]
        return pack(zones, f"0-10 profil pur, near={near_side}, facing={orient}")

    # =============== 10-30  3/4 LEGER ===============
    if angle <= 110:
        if near_side == "right":
            zones = [
                (f"Porte {orient}", 0.30, 0.55),
                (panel,             0.55, 0.72),
                (f"Aile {orient}",  0.72, 0.88),
                (f"Pare-choc {orient}", 0.88, 1.00),
            ]
        else:
            zones = [
                (f"Pare-choc {orient}", 0.00, 0.12),
                (f"Aile {orient}",      0.12, 0.28),
                (panel,                 0.28, 0.45),
                (f"Porte {orient}",     0.45, 0.70),
            ]
        return pack(zones, f"10-30 3/4 leger {orient} near={near_side}")

    # =============== 30-70  3/4 MARQUE ===============
    if angle <= 30:
        if near_side == "right":
            zones = [
                (f"Aile {orient}",      0.55, 0.78),
                (panel,                 0.78, 0.92),
                (f"Pare-choc {orient}", 0.92, 1.00),
            ]
        else:
            zones = [
                (f"Pare-choc {orient}", 0.00, 0.08),
                (panel,                 0.08, 0.22),
                (f"Aile {orient}",      0.22, 0.45),
            ]
        return pack(zones, f"30-70 3/4 marque {orient} near={near_side}")

    # =============== 70-110  FACE/ARRIERE PUR ===============
    if angle <= 10:
        zones = [
            (f"Pare-choc {orient}", 0.00, 0.55),
            (panel,                 0.45, 1.00),
        ]
        return pack(zones, f"70-110 {orient} pur -> Pare-choc + {panel}")

    # =============== 110-150  3/4 INVERSE ===============
    # On voit 3/4 mais de l'autre cote : 4 zones cote dominant
    # parchoq + aile {orient} + porte {orient} + porte {opposite}
    if angle >= 150:
        opp = "AV" if is_rear else "AR"
        if near_side == "right":
            zones = [
                (f"Porte {opp}",        0.20, 0.45),
                (f"Porte {orient}",     0.45, 0.65),
                (f"Aile {orient}",      0.65, 0.85),
                (f"Pare-choc {orient}", 0.85, 1.00),
            ]
        else:
            zones = [
                (f"Pare-choc {orient}", 0.00, 0.15),
                (f"Aile {orient}",      0.15, 0.35),
                (f"Porte {orient}",     0.35, 0.55),
                (f"Porte {opp}",        0.55, 0.80),
            ]
        return pack(zones, f"110-150 3/4 inverse {orient} near={near_side}")

    # =============== 150-180  PROFIL INVERSE ===============
    opp = "AV" if is_rear else "AR"
    if near_side == "right":
        zones = [
            (f"Aile {opp}",   0.00, 0.18),
            (f"Porte {opp}",  0.18, 0.50),
            (f"Porte {orient}",0.50, 0.82),
            (f"Aile {orient}",0.82, 1.00),
        ]
    else:
        zones = [
            (f"Aile {orient}", 0.00, 0.18),
            (f"Porte {orient}",0.18, 0.50),
            (f"Porte {opp}",   0.50, 0.82),
            (f"Aile {opp}",    0.82, 1.00),
        ]
    return pack(zones, f"150-180 profil inverse near={near_side}")


# =========================
# ANALYSE
# =========================
@app.route("/analyse", methods=["POST"])
def analyse():
    try:
        if "image" not in request.files:
            return jsonify({"error": "no image"}), 400

        try:
            angle = float(request.form.get("angle", "30"))
        except Exception:
            angle = 30.0
        angle = max(0.0, min(angle, 180.0))

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

        pad_x = int(15 * scale_x); pad_y = int(10 * scale_y)
        thr_x = int(150 * scale_x); thr_y = int(80 * scale_y)
        x1 = 0      if raw_x1 < thr_x            else max(0,     raw_x1 - pad_x)
        x2 = orig_w if (orig_w - raw_x2) < thr_x else min(orig_w, raw_x2 + pad_x)
        y1 = 0      if raw_y1 < thr_y            else max(0,     raw_y1 - pad_y)
        y2 = orig_h if (orig_h - raw_y2) < thr_y else min(orig_h, raw_y2 + pad_y)

        x1, y1, x2, y2 = refine_car_bbox(img_orig, x1, y1, x2, y2)
        car_crop = img_orig[y1:y2, x1:x2]
        if car_crop.size == 0:
            return jsonify({"error": "invalid crop"}), 400
        crop_h, crop_w = car_crop.shape[:2]

        # ===== 1) DETECTION AVANT/ARRIERE EN PREMIER =====
        lights = detect_lights(car_crop)
        rear_side, front_side, facing, fr_log = detect_front_rear(lights)

        # ===== 2) ZONES selon ANGLE + orientation deja decidee =====
        zones, zone_decision = build_zones(crop_w, crop_h, angle,
                                           rear_side, front_side, facing, lights)
        fr_log.append(f"ANGLE={angle} -> {zone_decision}")

        # ===== 3) Masque carrosserie + ref globale =====
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

        # ===== 4) Dessin avec polygones 3D numerotes =====
        final_img = img_orig.copy()
        thick_box   = max(3, int(4 * min(scale_x, scale_y)))
        thick_line  = max(1, int(1 * min(scale_x, scale_y)))
        font_big    = max(0.6, 0.6 * min(scale_x, scale_y))
        font_med    = max(0.5, 0.5 * min(scale_x, scale_y))
        font_thick  = max(2, int(2 * min(scale_x, scale_y)))

        cv2.rectangle(final_img, (x1, y1), (x2, y2), (220, 220, 220), thick_line)

        results_zones = []
        detected = 0

        for idx, zone in enumerate(zones, start=1):
            poly_local = zone["poly"]
            # convertir en coords image globale
            poly_global = np.array(
                [[x1 + p[0], y1 + p[1]] for p in poly_local],
                dtype=np.int32
            )

            zone_color, px_count = get_poly_color(hsv_full, mask_body, poly_local)

            if zone_color is None:
                color_rect  = (150, 150, 150)
                label_score = "N/A"; diff = 0.0
                verdict     = "Non analysable"
            else:
                diff = float(np.linalg.norm(zone_color - ref_color))
                if 14 <= diff < 26:
                    color_rect = (0, 0, 255);   verdict = "Attention peinture refaite!"
                elif diff < 14:
                    color_rect = (0, 165, 255); verdict = "Legere variation suspecte!"; detected += 1
                else:
                    color_rect = (0, 210, 0);   verdict = "OK"; detected += 1
                label_score = str(int(diff))

            # remplissage semi-transparent du polygone
            overlay = final_img.copy()
            cv2.fillPoly(overlay, [poly_global], color_rect)
            cv2.addWeighted(overlay, 0.25, final_img, 0.75, 0, final_img)

            # contour du polygone 3D
            cv2.polylines(final_img, [poly_global], True, color_rect, thick_box)

            # centre du polygone pour le numero
            cx = int(np.mean(poly_global[:, 0]))
            cy = int(np.mean(poly_global[:, 1]))

            # cercle numerote (style 3D : ombre + cercle plein)
            radius = max(18, int(22 * min(scale_x, scale_y)))
            cv2.circle(final_img, (cx + 2, cy + 2), radius, (0, 0, 0), -1)
            cv2.circle(final_img, (cx, cy), radius, color_rect, -1)
            cv2.circle(final_img, (cx, cy), radius, (255, 255, 255), 2)
            num_txt = str(idx)
            (tw, th), _ = cv2.getTextSize(num_txt, cv2.FONT_HERSHEY_SIMPLEX,
                                          font_big * 1.4, font_thick + 1)
            cv2.putText(final_img, num_txt,
                        (cx - tw // 2, cy + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_big * 1.4,
                        (255, 255, 255), font_thick + 1)

            # etiquette au-dessus du polygone
            top_pt = poly_global[poly_global[:, 1].argmin()]
            lbl_x, lbl_y = int(top_pt[0]), max(20, int(top_pt[1]) - 12)
            label_full = f"{idx}. {zone['name']}  E:{label_score}"
            (lw, lh), _ = cv2.getTextSize(label_full, cv2.FONT_HERSHEY_SIMPLEX,
                                          font_med, font_thick)
            cv2.rectangle(final_img,
                          (lbl_x - 4, lbl_y - lh - 6),
                          (lbl_x + lw + 6, lbl_y + 4),
                          (0, 0, 0), -1)
            cv2.putText(final_img, label_full, (lbl_x, lbl_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_med,
                        (255, 255, 255), font_thick)

            results_zones.append({
                "idx": idx,
                "zone": zone["name"],
                "diff": round(diff, 1),
                "pixels": px_count,
                "verdict": verdict,
                "polygon": poly_global.tolist()
            })

        diffs = [z["diff"] for z in results_zones if z["diff"] > 0]
        final_score = min(int(np.mean(diffs)) if diffs else 0, 100)
        if   final_score < 10: result = "Peinture homogene (OK)"
        elif final_score < 28: result = "Legeres variations detectees"
        else:                  result = "Difference importante - repeinture probable"

        analysed_name = "analysed_" + filename
        analysed_path = os.path.join(UPLOAD_FOLDER, analysed_name)
        cv2.imwrite(analysed_path, final_img)
        if os.path.exists(resized_path):
            os.remove(resized_path)

        return jsonify({
            "yolo":            yolo_result,
            "angle":           angle,
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
