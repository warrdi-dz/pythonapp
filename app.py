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
# Retire vitres, pneus, plastique, reflets
# =============================================
def build_mask(car_crop):
    hsv = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)

    # Trop sombre = vitres, pneus
    mask_dark = cv2.inRange(hsv, (0, 0, 0),   (180, 255, 45))
    # Fond blanc / ciel
    mask_sky  = cv2.inRange(hsv, (0, 0, 215), (180, 18, 255))
    # Plastique, chrome, faible saturation
    mask_chro = cv2.inRange(hsv, (0, 0, 0),   (180, 30, 255))
    # Surexposé = reflets soleil
    mask_over = cv2.inRange(hsv, (0, 0, 230), (180, 255, 255))

    exclude   = cv2.bitwise_or(mask_dark, mask_sky)
    exclude   = cv2.bitwise_or(exclude,   mask_chro)
    exclude   = cv2.bitwise_or(exclude,   mask_over)
    mask_body = cv2.bitwise_not(exclude)

    k         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_CLOSE, k, iterations=2)
    mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_OPEN,  k, iterations=1)

    return mask_body


# =============================================
# FEATURES D'UNE ZONE
# LAB (a,b) + teinte H + texture
# On IGNORE L et V (luminosité) → anti-soleil
# =============================================
def get_zone_features(lab_img, hsv_img, gray_img, mask, xA, yA, xB, yB):
    zm   = mask[yA:yB, xA:xB]
    zl   = lab_img[yA:yB, xA:xB]
    zh   = hsv_img[yA:yB, xA:xB]
    zg   = gray_img[yA:yB, xA:xB]

    vl   = zl[zm > 0]
    vh   = zh[zm > 0]
    vg   = zg[zm > 0]

    if len(vl) < 80:
        return None

    return {
        # Teinte pure LAB sans luminosité
        "a":       float(np.median(vl[:, 1])),
        "b":       float(np.median(vl[:, 2])),
        "a_std":   float(np.std(vl[:, 1])),
        "b_std":   float(np.std(vl[:, 2])),
        # Teinte H et saturation S
        "h":       float(np.median(vh[:, 0])),
        "s":       float(np.median(vh[:, 1])),
        # Texture grain peinture
        "texture": float(np.std(vg)),
        "count":   len(vl)
    }


# =============================================
# SCORE ADAPTATIF
# Compare zone à référence avec seuils
# calibrés sur la variabilité naturelle
# de CETTE voiture spécifique
# =============================================
def compute_adaptive_score(zone_f, ref_f, natural_std_a, natural_std_b):
    """
    natural_std_a / natural_std_b = écart-type naturel
    des canaux a,b sur toute la carrosserie.
    Ça représente la variabilité normale de la voiture
    (ombres, courbures, reflets légers).

    On normalise le delta par cet écart-type naturel :
    - Si delta >> natural_std → vraiment différent
    - Si delta ≈ natural_std  → variation normale

    C'est ce qui rend le score universel pour toute
    couleur de voiture.
    """
    # Delta teinte a,b LAB normalisé
    raw_da = abs(zone_f["a"] - ref_f["a"])
    raw_db = abs(zone_f["b"] - ref_f["b"])

    # Normalisation : combien de "déviations naturelles"
    norm_da = raw_da / max(natural_std_a, 1.0)
    norm_db = raw_db / max(natural_std_b, 1.0)
    delta_ab_norm = float(np.sqrt(norm_da**2 + norm_db**2))

    # Delta teinte H circulaire normalisé
    dh = abs(zone_f["h"] - ref_f["h"])
    if dh > 90:
        dh = 180 - dh
    delta_h_norm = float(dh) / 10.0  # normalisation fixe H

    # Delta saturation
    delta_s = abs(zone_f["s"] - ref_f["s"]) / 20.0

    # Delta texture
    delta_tex = abs(zone_f["texture"] - ref_f["texture"]) / 5.0

    # Score final pondéré (tous normalisés, comparables)
    score = (
        delta_ab_norm * 0.45 +
        delta_h_norm  * 0.25 +
        delta_s       * 0.20 +
        delta_tex     * 0.10
    )

    return {
        "score":       round(score, 2),
        "delta_ab":    round(float(np.sqrt(raw_da**2 + raw_db**2)), 1),
        "delta_h":     round(float(dh), 1),
        "delta_s":     round(abs(zone_f["s"] - ref_f["s"]), 1),
        "delta_tex":   round(abs(zone_f["texture"] - ref_f["texture"]), 1),
    }


# =============================================
# COHÉRENCE INTERNE : 3 BANDES
# Confirme si toute la zone est différente
# ou juste un reflet/ombre ponctuel
# =============================================
def check_coherence(lab, hsv, gray, mask,
                    xA, yA, xB, yB,
                    ref_f, nat_a, nat_b):
    zone_h  = yB - yA
    band_h  = zone_h // 3
    scores  = []

    for b in range(3):
        byA = yA + b * band_h
        byB = yA + (b + 1) * band_h if b < 2 else yB
        f   = get_zone_features(lab, hsv, gray, mask, xA, byA, xB, byB)
        if f is None:
            continue
        r = compute_adaptive_score(f, ref_f, nat_a, nat_b)
        scores.append(r["score"])

    if len(scores) < 2:
        return 0.0, False

    std_s  = float(np.std(scores))
    mean_s = float(np.mean(scores))

    # Cohérente + différente = vraie repeinture
    is_coh = (std_s < 0.4) and (mean_s > 0.5)
    bonus  = 0.5 if is_coh else 0.0
    return bonus, is_coh


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
            log.append("P1 ROUGE → avant DROITE")
            return "right", log
        elif red_right > red_left * 1.35:
            log.append("P1 ROUGE → avant GAUCHE")
            return "left", log
        else:
            log.append("P1 equilibre, passe P2")
    else:
        log.append(f"P1 insuffisant ({total_red}px), passe P2")

    def count_headlight(hsv):
        white  = cv2.inRange(hsv, (0,  0,  170), (180, 90,  255))
        yellow = cv2.inRange(hsv, (15, 40, 170), (40,  220, 255))
        return cv2.countNonZero(cv2.bitwise_or(white, yellow))

    light_left  = count_headlight(left_hsv)
    light_right = count_headlight(right_hsv)
    total_light = light_left + light_right

    if total_light > 100:
        if light_left > light_right * 1.35:
            log.append("P2 PHARE → avant GAUCHE")
            return "left", log
        elif light_right > light_left * 1.35:
            log.append("P2 PHARE → avant DROITE")
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

        orientation, orient_log = detect_car_orientation(car_crop)

        if orientation == "left":
            zone_names = ["Avant", "Portes", "Arriere"]
        else:
            zone_names = ["Arriere", "Portes", "Avant"]

        # ===============================================
        # ESPACES COLORIMÉTRIQUES + MASQUE
        # ===============================================
        hsv_full  = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
        lab_full  = cv2.cvtColor(car_crop, cv2.COLOR_BGR2LAB)
        gray_full = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
        mask_body = build_mask(car_crop)

        # ===============================================
        # RÉFÉRENCE GLOBALE + VARIABILITÉ NATURELLE
        # La variabilité naturelle = std des canaux a,b
        # sur toute la carrosserie = ombres + courbures
        # normales. C'est la "baseline" de cette voiture.
        # ===============================================
        ref_f = get_zone_features(
            lab_full, hsv_full, gray_full, mask_body,
            0, 0, crop_w, crop_h
        )
        if ref_f is None:
            return jsonify({"error": "No body pixels found"}), 400

        # Calcul de la variabilité naturelle sur pixels valides
        vl_all     = lab_full[mask_body > 0]
        natural_std_a = max(float(np.std(vl_all[:, 1])), 1.0)
        natural_std_b = max(float(np.std(vl_all[:, 2])), 1.0)

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
        final_img      = img_orig.copy()
        thick_box      = max(3, int(5 * min(scale_x, scale_y)))
        thick_line     = max(1, int(1 * min(scale_x, scale_y)))
        font_scale_big = max(0.6, 0.6 * min(scale_x, scale_y))
        font_scale_med = max(0.5, 0.5 * min(scale_x, scale_y))
        font_thick_big = max(2,   int(2 * min(scale_x, scale_y)))
        overlay_h      = max(55,  int(55 * scale_y))

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

            z_f = get_zone_features(
                lab_full, hsv_full, gray_full, mask_body,
                xA, yA, xB, yB
            )

            abs_x1 = x1 + xA
            abs_y1 = y1 + yA
            abs_x2 = x1 + xB
            abs_y2 = y1 + yB

            if z_f is None:
                color_rect  = (150, 150, 150)
                verdict     = "Non analysable"
                score_final = 0.0
                label_score = "N/A"
                detail      = {}
            else:
                # Score adaptatif normalisé
                sc = compute_adaptive_score(
                    z_f, ref_f, natural_std_a, natural_std_b
                )

                # Cohérence interne 3 bandes
                bonus_coh, is_coh = check_coherence(
                    lab_full, hsv_full, gray_full, mask_body,
                    xA, yA, xB, yB,
                    ref_f, natural_std_a, natural_std_b
                )

                score_final = round(sc["score"] + bonus_coh, 2)
                label_score = f"{score_final:.1f}"

                detail = {
                    "delta_ab":      sc["delta_ab"],
                    "delta_h":       sc["delta_h"],
                    "delta_s":       sc["delta_s"],
                    "delta_texture": sc["delta_tex"],
                    "bonus_coherence": round(bonus_coh, 2),
                    "coherente":     is_coh,
                    "nat_std_a":     round(natural_std_a, 1),
                    "nat_std_b":     round(natural_std_b, 1)
                }

                # =======================================
                # SEUILS ADAPTATIFS — universels
                # Score normalisé : 1.0 = 1 déviation
                # naturelle de cette voiture
                # < 0.8  → OK (variation normale)
                # 0.8-1.5 → suspect
                # > 1.5  → repeinture probable
                # =======================================
                if score_final > 1.5:
                    color_rect = (0, 0, 255)
                    verdict    = "Peinture refaite probable!"
                    detected  += 1
                elif score_final > 0.8:
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
                "score":   score_final,
                "pixels":  z_f["count"] if z_f else 0,
                "verdict": verdict,
                "detail":  detail
            })

        # ===============================================
        # SCORE GLOBAL
        # ===============================================
        scores = [z["score"] for z in results_zones if z["score"] > 0]
        final_score_raw = float(np.mean(scores)) if scores else 0.0

        if final_score_raw > 1.5:
            result = "Difference importante — repeinture probable"
        elif final_score_raw > 0.8:
            result = "Legeres variations detectees"
        else:
            result = "Peinture homogene (OK)"

        # Score sur 100 pour affichage
        final_score_100 = min(int(final_score_raw * 40), 100)

        analysed_name = "analysed_" + filename
        analysed_path = os.path.join(UPLOAD_FOLDER, analysed_name)
        cv2.imwrite(analysed_path, final_img)

        if os.path.exists(resized_path):
            os.remove(resized_path)

        return jsonify({
            "yolo":            yolo_result,
            "score":           final_score_100,
            "score_raw":       round(final_score_raw, 2),
            "result":          result,
            "zones":           results_zones,
            "zones_detected":  detected,
            "orientation":     orientation,
            "orientation_log": orient_log,
            "image_size":      {"width": orig_w, "height": orig_h},
            "calibration": {
                "natural_std_a": round(natural_std_a, 1),
                "natural_std_b": round(natural_std_b, 1),
                "ref_a":         round(ref_f["a"], 1),
                "ref_b":         round(ref_f["b"], 1),
                "ref_h":         round(ref_f["h"], 1)
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
