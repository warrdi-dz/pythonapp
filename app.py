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

# =========================
# UPLOADS
# =========================
@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# =========================
# HOME
# =========================
@app.route("/")
def home():
    return jsonify({"status": "OK", "message": "GARAGE PRO V4 API"})


# =========================
# AMÉLIORATION 1+2+3
# Couleur LAB + Texture + Brillance
# Retourne (lab_color, std_texture, mean_L)
# pour une zone masquée
# =========================
def get_zone_features(lab_img, gray_img, mask, xA, yA, xB, yB):
    """
    Extrait 3 features sur les pixels valides de la zone :
    - Couleur LAB médiane  (amélioration 1)
    - Texture = std locale  (amélioration 2)
    - Brillance = médiane L (amélioration 3)
    """
    zone_mask = mask[yA:yB, xA:xB]
    zone_lab  = lab_img[yA:yB, xA:xB]
    zone_gray = gray_img[yA:yB, xA:xB]

    valid_lab  = zone_lab[zone_mask > 0]
    valid_gray = zone_gray[zone_mask > 0]

    if len(valid_lab) < 80:
        return None, None, None, 0

    # 1. Couleur LAB médiane
    lab_color = np.array([
        float(np.median(valid_lab[:, 0])),
        float(np.median(valid_lab[:, 1])),
        float(np.median(valid_lab[:, 2]))
    ])

    # 2. Texture : écart-type des pixels gris valides
    texture = float(np.std(valid_gray))

    # 3. Brillance : médiane du canal L (luminosité LAB)
    brillance = float(np.median(valid_lab[:, 0]))

    return lab_color, texture, brillance, len(valid_lab)


# =========================
# AMÉLIORATION 4
# Score combiné
# couleur + texture + brillance
# =========================
def compute_score(zone_lab, zone_tex, zone_bril,
                  ref_lab,  ref_tex,  ref_bril):
    """
    Score combiné pondéré :
    - Delta E LAB (couleur)    : poids 50%
    - Différence texture       : poids 30%
    - Différence brillance     : poids 20%
    Résultat normalisé 0-100
    """
    # Delta E = distance euclidienne dans LAB
    delta_e = float(np.linalg.norm(zone_lab - ref_lab))

    # Différence de texture normalisée
    delta_tex = abs(zone_tex - ref_tex)

    # Différence de brillance
    delta_bril = abs(zone_bril - ref_bril)

    # Score combiné pondéré
    score = (delta_e   * 0.50) + \
            (delta_tex * 0.30) + \
            (delta_bril * 0.20)

    return round(score, 1), round(delta_e, 1), round(delta_tex, 1), round(delta_bril, 1)


# =========================
# AMÉLIORATION 5
# Analyse par bandes horizontales
# Vérifie cohérence interne zone
# vs référence globale
# =========================
def analyse_bands(lab_img, gray_img, mask,
                  xA, yA, xB, yB,
                  ref_lab, ref_tex, ref_bril):
    """
    Divise la zone en 3 bandes horizontales
    (haut / milieu / bas).
    Si les 3 bandes sont cohérentes entre elles
    MAIS différentes de la référence → forte suspicion.
    Retourne un score de suspicion supplémentaire.
    """
    zone_h = yB - yA
    band_h = zone_h // 3

    band_scores  = []
    band_deltas  = []

    for b in range(3):
        byA = yA + b * band_h
        byB = yA + (b + 1) * band_h if b < 2 else yB

        lab_c, tex, bril, count = get_zone_features(
            lab_img, gray_img, mask, xA, byA, xB, byB
        )
        if lab_c is None:
            continue

        sc, de, dt, db = compute_score(
            lab_c, tex, bril, ref_lab, ref_tex, ref_bril
        )
        band_scores.append(sc)
        band_deltas.append(de)

    if len(band_scores) < 2:
        return 0.0, False

    # Cohérence interne : les 3 bandes se ressemblent entre elles
    internal_std = float(np.std(band_scores))

    # Elles sont cohérentes (std faible) MAIS différentes de la ref
    mean_score   = float(np.mean(band_scores))

    # Suspicion forte = cohérente + différente
    is_suspicious = (internal_std < 8.0) and (mean_score > 12.0)

    # Bonus de score si cohérence confirmée
    bonus = 10.0 if is_suspicious else 0.0

    return bonus, is_suspicious


# =========================
# AMÉLIORATION 6
# Détection raccord peinture
# sur les bords des zones
# =========================
def detect_border_discontinuity(lab_img, mask, x_border, yA, yB):
    """
    Analyse une bande verticale de 30px de large
    centrée sur la frontière entre deux zones.
    Un gradient LAB fort = raccord de peinture visible.
    Retourne un score de discontinuité (0-50).
    """
    h, w = lab_img.shape[:2]
    bw   = 30  # largeur de la bande d'analyse

    x1b = max(0, x_border - bw)
    x2b = min(w, x_border + bw)

    if x2b - x1b < 10:
        return 0.0

    band_lab  = lab_img[yA:yB, x1b:x2b].astype(np.float32)
    band_mask = mask[yA:yB, x1b:x2b]

    if band_mask.sum() < 50:
        return 0.0

    # Gradient horizontal sur le canal L
    band_L = band_lab[:, :, 0]
    band_L[band_mask == 0] = np.nan

    # Ignorer les colonnes sans pixels valides
    col_means = np.nanmean(band_L, axis=0)
    valid     = ~np.isnan(col_means)

    if valid.sum() < 4:
        return 0.0

    col_valid = col_means[valid]

    # Gradient = différence entre moitié gauche et droite
    mid        = len(col_valid) // 2
    left_mean  = float(np.nanmean(col_valid[:mid]))
    right_mean = float(np.nanmean(col_valid[mid:]))

    discontinuity = abs(right_mean - left_mean)
    return round(min(discontinuity, 50.0), 1)


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
# DÉTECTER ORIENTATION
# =========================
def detect_car_orientation(car_crop):
    crop_h, crop_w = car_crop.shape[:2]
    log = []

    band_w  = int(crop_w * 0.22)
    feux_y1 = int(crop_h * 0.38)
    feux_y2 = int(crop_h * 0.88)

    left_feux  = car_crop[feux_y1:feux_y2, 0:band_w]
    right_feux = car_crop[feux_y1:feux_y2, crop_w - band_w:crop_w]
    left_hsv   = cv2.cvtColor(left_feux,  cv2.COLOR_BGR2HSV)
    right_hsv  = cv2.cvtColor(right_feux, cv2.COLOR_BGR2HSV)

    def count_red(hsv):
        m1 = cv2.inRange(hsv, (0,   60, 60), (12,  255, 255))
        m2 = cv2.inRange(hsv, (168, 60, 60), (180, 255, 255))
        return cv2.countNonZero(cv2.bitwise_or(m1, m2))

    red_left  = count_red(left_hsv)
    red_right = count_red(right_hsv)
    total_red = red_left + red_right

    if total_red > 150:
        if red_left > red_right * 1.35:
            log.append(f"P1 ROUGE → avant DROITE")
            return "right", log
        elif red_right > red_left * 1.35:
            log.append(f"P1 ROUGE → avant GAUCHE")
            return "left", log
        else:
            log.append("P1 equilibre, passe P2")
    else:
        log.append("P1 insuffisant, passe P2")

    def count_headlight(hsv):
        white  = cv2.inRange(hsv, (0,  0,  170), (180, 90,  255))
        yellow = cv2.inRange(hsv, (15, 40, 170), (40,  220, 255))
        return cv2.countNonZero(cv2.bitwise_or(white, yellow))

    light_left  = count_headlight(left_hsv)
    light_right = count_headlight(right_hsv)
    total_light = light_left + light_right

    if total_light > 100:
        if light_left > light_right * 1.35:
            log.append(f"P2 PHARE → avant GAUCHE")
            return "left", log
        elif light_right > light_left * 1.35:
            log.append(f"P2 PHARE → avant DROITE")
            return "right", log
        else:
            log.append("P2 equilibre, passe P3")
    else:
        log.append("P2 insuffisant, passe P3")

    gray        = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
    vit_y1      = int(crop_h * 0.08)
    vit_y2      = int(crop_h * 0.58)
    vitre       = gray[vit_y1:vit_y2, :]
    dark        = (vitre < 90).astype(np.uint8)
    dark_f      = cv2.GaussianBlur(dark.astype(np.float32), (15, 15), 0)
    mid         = crop_w // 2
    left_glass  = float(dark_f[:, :mid].sum())
    right_glass = float(dark_f[:, mid:].sum())

    if left_glass > right_glass * 1.15:
        log.append("P3 VITRE → avant GAUCHE")
        return "left", log
    elif right_glass > left_glass * 1.15:
        log.append("P3 VITRE → avant DROITE")
        return "right", log

    left_band  = car_crop[feux_y1:feux_y2, 0:band_w]
    right_band = car_crop[feux_y1:feux_y2, crop_w - band_w:crop_w]
    lg = cv2.cvtColor(left_band,  cv2.COLOR_BGR2GRAY)
    rg = cv2.cvtColor(right_band, cv2.COLOR_BGR2GRAY)
    ls = float(np.mean(np.abs(cv2.Sobel(lg, cv2.CV_64F, 1, 1, ksize=3))))
    rs = float(np.mean(np.abs(cv2.Sobel(rg, cv2.CV_64F, 1, 1, ksize=3))))

    if rs > ls * 1.2:
        log.append("FALLBACK Sobel → avant GAUCHE")
        return "left", log
    else:
        log.append("FALLBACK Sobel → avant DROITE")
        return "right", log


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

        # YOLO sur version réduite
        img_yolo     = cv2.resize(img_orig, (YOLO_W, YOLO_H))
        resized_path = os.path.join(UPLOAD_FOLDER, "resized_" + filename)
        cv2.imwrite(resized_path, img_yolo)
        yolo_result  = call_yolo(resized_path)

        detections = yolo_result.get("detections", [])
        cars = [d for d in detections if d.get("class") == 2]
        if not cars:
            return jsonify({"error": "Car not detected"}), 400

        # Rescale coords YOLO → résolution originale
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

        # Orientation
        orientation, orient_log = detect_car_orientation(car_crop)
        if orientation == "left":
            zone_names = ["Avant", "Portes", "Arriere"]
        else:
            zone_names = ["Arriere", "Portes", "Avant"]

        # ===============================================
        # PRÉPARER LES ESPACES COLORIMÉTRIQUES
        # ===============================================
        # LAB pour la couleur (amélioration 1)
        lab_crop  = cv2.cvtColor(car_crop, cv2.COLOR_BGR2LAB)

        # Gris pour texture et brillance (amélioration 2+3)
        gray_crop = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)

        # Masque carrosserie
        hsv_full  = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
        mask_dark = cv2.inRange(hsv_full, (0, 0, 0),   (180, 255, 45))
        mask_sky  = cv2.inRange(hsv_full, (0, 0, 210), (180, 18, 255))
        mask_body = cv2.bitwise_not(cv2.bitwise_or(mask_dark, mask_sky))
        kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_CLOSE, kernel)

        # ===============================================
        # RÉFÉRENCE GLOBALE (toute la carrosserie)
        # ===============================================
        ref_lab, ref_tex, ref_bril, _ = get_zone_features(
            lab_crop, gray_crop, mask_body,
            0, 0, crop_w, crop_h
        )
        if ref_lab is None:
            return jsonify({"error": "No body pixels found"}), 400

        # ===============================================
        # 3 ZONES
        # ===============================================
        band_y1 = int(crop_h * 0.15)
        band_y2 = int(crop_h * 0.80)
        cut1    = int(crop_w * 0.33)
        cut2    = int(crop_w * 0.67)

        zones = [
            {"name": zone_names[0], "xA": 0,    "xB": cut1,   "yA": band_y1, "yB": band_y2},
            {"name": zone_names[1], "xA": cut1, "xB": cut2,   "yA": band_y1, "yB": band_y2},
            {"name": zone_names[2], "xA": cut2, "xB": crop_w, "yA": band_y1, "yB": band_y2},
        ]

        # ===============================================
        # DESSIN
        # ===============================================
        final_img  = img_orig.copy()
        thick_box  = max(3, int(5 * min(scale_x, scale_y)))
        thick_line = max(1, int(1 * min(scale_x, scale_y)))
        fs_big     = max(0.6, 0.6  * min(scale_x, scale_y))
        fs_med     = max(0.5, 0.5  * min(scale_x, scale_y))
        ft_big     = max(2,   int(2 * min(scale_x, scale_y)))
        ov_h       = max(55,  int(55 * scale_y))

        cv2.rectangle(final_img, (x1, y1), (x2, y2), (220, 220, 220), thick_line)

        step = max(10, int(12 * scale_y))
        dash = max(4,  int(6  * scale_y))
        for cut in [cut1, cut2]:
            for dy in range(band_y1, band_y2, step):
                cv2.line(
                    final_img,
                    (x1 + cut, y1 + dy),
                    (x1 + cut, y1 + dy + dash),
                    (255, 255, 255), thick_line
                )

        results_zones = []
        detected = 0

        for zone in zones:
            xA, xB = zone["xA"], zone["xB"]
            yA, yB = zone["yA"], zone["yB"]

            # Features de la zone (amélioration 1+2+3)
            z_lab, z_tex, z_bril, px_count = get_zone_features(
                lab_crop, gray_crop, mask_body, xA, yA, xB, yB
            )

            abs_x1 = x1 + xA
            abs_y1 = y1 + yA
            abs_x2 = x1 + xB
            abs_y2 = y1 + yB

            if z_lab is None:
                color_rect  = (150, 150, 150)
                label_score = "N/A"
                final_score_zone = 0.0
                verdict     = "Non analysable"
                detail      = {}
            else:
                # Score combiné (amélioration 4)
                score_c, d_e, d_tex, d_bril = compute_score(
                    z_lab, z_tex, z_bril,
                    ref_lab, ref_tex, ref_bril
                )

                # Bonus bandes horizontales (amélioration 5)
                bonus, is_susp = analyse_bands(
                    lab_crop, gray_crop, mask_body,
                    xA, yA, xB, yB,
                    ref_lab, ref_tex, ref_bril
                )

                # Discontinuité aux bords (amélioration 6)
                disc_left  = 0.0
                disc_right = 0.0
                if xA > 0:
                    disc_left = detect_border_discontinuity(
                        lab_crop, mask_body, xA, yA, yB
                    )
                if xB < crop_w:
                    disc_right = detect_border_discontinuity(
                        lab_crop, mask_body, xB, yA, yB
                    )
                disc_bonus = min((disc_left + disc_right) / 2.0, 15.0)

                # Score final zone = combiné + bonus bandes + bonus raccord
                final_score_zone = round(score_c + bonus + disc_bonus, 1)

                detail = {
                    "delta_e":      d_e,
                    "delta_tex":    d_tex,
                    "delta_bril":   d_bril,
                    "bonus_bandes": round(bonus, 1),
                    "bonus_raccord":round(disc_bonus, 1),
                    "coherente":    is_susp
                }

                # Seuils verdict
                if final_score_zone >= 20 and final_score_zone < 35:
                    color_rect = (0, 0, 255)
                    verdict    = "Attention peinture refaite!"
                    detected  += 1
                elif final_score_zone < 20:
                    color_rect = (0, 165, 255)
                    verdict    = "Legere variation suspecte!"
                    detected  += 1
                else:
                    color_rect = (0, 210, 0)
                    verdict    = "OK"

                label_score = str(int(final_score_zone))

            # Dessin grand rectangle
            cv2.rectangle(final_img, (abs_x1, abs_y1),
                          (abs_x2, abs_y2), color_rect, thick_box)

            overlay = final_img.copy()
            cv2.rectangle(overlay, (abs_x1, abs_y1),
                          (abs_x2, abs_y1 + ov_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, final_img, 0.5, 0, final_img)

            cv2.putText(final_img, zone["name"],
                        (abs_x1 + 8, abs_y1 + int(ov_h * 0.40)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_big,
                        (255, 255, 255), ft_big)

            cv2.putText(final_img, f"Score: {label_score}",
                        (abs_x1 + 8, abs_y1 + int(ov_h * 0.80)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_med,
                        color_rect, ft_big)

            cv2.putText(final_img, verdict,
                        (abs_x1 + 8, abs_y2 - int(10 * scale_y)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_med,
                        color_rect, ft_big)

            results_zones.append({
                "zone":    zone["name"],
                "score":   final_score_zone,
                "pixels":  px_count,
                "verdict": verdict,
                "detail":  detail
            })

        # Score global
        scores = [z["score"] for z in results_zones if z["score"] > 0]
        global_score = int(np.mean(scores)) if scores else 0
        global_score = min(global_score, 100)

        if global_score < 15:
            result = "Peinture homogene (OK)"
        elif global_score < 30:
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
            "score":           global_score,
            "result":          result,
            "zones":           results_zones,
            "zones_detected":  detected,
            "orientation":     orientation,
            "orientation_log": orient_log,
            "image_size":      {"width": orig_w, "height": orig_h},
            "image_result":    analysed_name,
            "image_url":       request.host_url + "uploads/" + analysed_name
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
