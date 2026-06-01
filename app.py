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

TARGET_W = 900
TARGET_H = 500

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
# Trouve les vrais bords de la
# carrosserie dans la bbox YOLO
# pour gérer les photos zoomées
# =========================
def refine_car_bbox(img, x1, y1, x2, y2):
    """
    À partir de la bbox YOLO (qui peut être trop large
    sur une photo zoomée), on cherche les vrais bords
    de la voiture en analysant où se concentrent
    les pixels de carrosserie (non-fond, non-ciel).
    Retourne (rx1, ry1, rx2, ry2) affiné.
    """
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return x1, y1, x2, y2

    ch, cw = crop.shape[:2]
    hsv    = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Masque pixels "voiture" = ni trop sombres, ni fond blanc/ciel
    mask_dark  = cv2.inRange(hsv, (0, 0, 0),   (180, 255, 40))
    mask_sky   = cv2.inRange(hsv, (0, 0, 215), (180, 15, 255))
    mask_valid = cv2.bitwise_not(cv2.bitwise_or(mask_dark, mask_sky))

    # Morphologie pour boucher les trous
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_valid = cv2.morphologyEx(mask_valid, cv2.MORPH_CLOSE, k, iterations=2)

    # Somme des pixels valides par colonne et par ligne
    col_sum = mask_valid.sum(axis=0).astype(float)   # largeur
    row_sum = mask_valid.sum(axis=1).astype(float)   # hauteur

    # Lisser pour ignorer le bruit
    col_sum = np.convolve(col_sum, np.ones(15) / 15, mode='same')
    row_sum = np.convolve(row_sum, np.ones(15) / 15, mode='same')

    # Seuil : colonne/ligne doit avoir au moins 8% du max
    col_thresh = col_sum.max() * 0.08
    row_thresh = row_sum.max() * 0.08

    valid_cols = np.where(col_sum > col_thresh)[0]
    valid_rows = np.where(row_sum > row_thresh)[0]

    if len(valid_cols) < 20 or len(valid_rows) < 20:
        return x1, y1, x2, y2

    # Bords affinés avec petit padding
    PAD   = 8
    new_x1 = max(0,         x1 + int(valid_cols[0])  - PAD)
    new_x2 = min(img.shape[1], x1 + int(valid_cols[-1]) + PAD)
    new_y1 = max(0,         y1 + int(valid_rows[0])  - PAD)
    new_y2 = min(img.shape[0], y1 + int(valid_rows[-1]) + PAD)

    # Sécurité : le crop affiné doit faire au moins 100x80 px
    if (new_x2 - new_x1) < 100 or (new_y2 - new_y1) < 80:
        return x1, y1, x2, y2

    return new_x1, new_y1, new_x2, new_y2


# =========================
# DÉTECTER ORIENTATION
# 3 méthodes + vote
# =========================
def detect_car_orientation(car_crop):
    crop_h, crop_w = car_crop.shape[:2]
    votes_left  = 0
    votes_right = 0

    band_y1 = int(crop_h * 0.30)
    band_y2 = int(crop_h * 0.85)
    band_w  = int(crop_w * 0.22)

    left_zone  = car_crop[band_y1:band_y2, 0:band_w]
    right_zone = car_crop[band_y1:band_y2, crop_w - band_w:crop_w]

    # Méthode 1 : feux rouges
    def count_red(zone):
        hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
        m1  = cv2.inRange(hsv, (0,   70, 70), (10,  255, 255))
        m2  = cv2.inRange(hsv, (170, 70, 70), (180, 255, 255))
        return cv2.countNonZero(cv2.bitwise_or(m1, m2))

    red_left  = count_red(left_zone)
    red_right = count_red(right_zone)
    if red_left > red_right * 1.4:
        votes_right += 1
    elif red_right > red_left * 1.4:
        votes_left += 1

    # Méthode 2 : position pare-brise
    gray      = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
    top_band  = gray[int(crop_h * 0.10):int(crop_h * 0.55), :]
    dark_mask = (top_band < 80).astype(np.uint8)
    col_sums  = dark_mask.sum(axis=0).astype(float)
    col_sums  = np.convolve(col_sums, np.ones(20) / 20, mode='same')
    total     = col_sums.sum()
    if total > 0:
        center_x = float(np.sum(np.arange(len(col_sums)) * col_sums)) / total
        third = crop_w / 3.0
        if center_x < third:
            votes_left += 1
        elif center_x > crop_w - third:
            votes_right += 1
        else:
            left_dark  = col_sums[:crop_w // 2].sum()
            right_dark = col_sums[crop_w // 2:].sum()
            if left_dark > right_dark * 1.3:
                votes_left += 1
            elif right_dark > left_dark * 1.3:
                votes_right += 1

    # Méthode 3 : texture Sobel
    left_gray   = cv2.cvtColor(left_zone,  cv2.COLOR_BGR2GRAY)
    right_gray  = cv2.cvtColor(right_zone, cv2.COLOR_BGR2GRAY)
    left_sobel  = cv2.Sobel(left_gray,  cv2.CV_64F, 1, 1, ksize=3)
    right_sobel = cv2.Sobel(right_gray, cv2.CV_64F, 1, 1, ksize=3)
    left_detail  = float(np.mean(np.abs(left_sobel)))
    right_detail = float(np.mean(np.abs(right_sobel)))
    if right_detail > left_detail * 1.25:
        votes_left += 1
    elif left_detail > right_detail * 1.25:
        votes_right += 1

    return "left" if votes_left >= votes_right else "right"


# =========================
# DÉTECTER RÉTROVISEUR
# =========================
def detect_mirror(car_crop, orientation, mask_body):
    crop_h, crop_w = car_crop.shape[:2]

    if orientation == "left":
        sx1 = int(crop_w * 0.08)
        sx2 = int(crop_w * 0.38)
    else:
        sx1 = int(crop_w * 0.62)
        sx2 = int(crop_w * 0.92)

    sy1 = int(crop_h * 0.18)
    sy2 = int(crop_h * 0.52)

    search = car_crop[sy1:sy2, sx1:sx2].copy()
    if search.size == 0:
        return None

    sh, sw = search.shape[:2]
    gray    = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    grad   = np.sqrt(sobelx**2 + sobely**2)
    grad   = np.uint8(np.clip(grad / grad.max() * 255, 0, 255))

    _, thresh = cv2.threshold(grad, 60, 255, cv2.THRESH_BINARY)
    kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 4))
    closed    = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated   = cv2.dilate(closed, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    zone_area = sh * sw
    min_area  = zone_area * 0.008
    max_area  = zone_area * 0.10
    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue
        ratio = w / h
        if ratio < 0.8 or ratio > 4.0:
            continue
        center_y = y + h / 2
        if center_y > sh * 0.75:
            continue
        compactness = area / (w * h) if w * h > 0 else 0
        top_bonus   = 1.0 - (center_y / sh)
        score       = area * compactness * (1 + top_bonus)
        candidates.append((score, x, y, w, h))

    if not candidates:
        return None

    candidates.sort(key=lambda c: c[0], reverse=True)
    _, bx, by, bw, bh = candidates[0]

    pad = 5
    bx  = max(0, bx - pad)
    by  = max(0, by - pad)
    bw  = min(sw - bx, bw + pad * 2)
    bh  = min(sh - by, bh + pad * 2)

    return (sx1 + bx, sy1 + by, sx1 + bx + bw, sy1 + by + bh)


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

        img = cv2.resize(img_orig, (TARGET_W, TARGET_H))

        resized_path = os.path.join(UPLOAD_FOLDER, "resized_" + filename)
        cv2.imwrite(resized_path, img)

        yolo_result = call_yolo(resized_path)

        img_h, img_w = img.shape[:2]

        detections = yolo_result.get("detections", [])
        cars = [d for d in detections if d.get("class") == 2]
        if not cars:
            return jsonify({"error": "Car not detected"}), 400

        raw_x1 = min(d["box"][0] for d in cars)
        raw_y1 = min(d["box"][1] for d in cars)
        raw_x2 = max(d["box"][2] for d in cars)
        raw_y2 = max(d["box"][3] for d in cars)

        # Bbox YOLO initiale
        x1 = 0      if raw_x1 < 150           else max(0,     raw_x1 - 15)
        x2 = img_w  if (img_w - raw_x2) < 150 else min(img_w, raw_x2 + 15)
        y1 = 0      if raw_y1 < 80            else max(0,     raw_y1 - 10)
        y2 = img_h  if (img_h - raw_y2) < 80  else min(img_h, raw_y2 + 10)

        # ===============================================
        # AFFINAGE DU CROP
        # Corrige les photos trop zoomées où la bbox
        # YOLO déborde sur le fond
        # ===============================================
        x1, y1, x2, y2 = refine_car_bbox(img, x1, y1, x2, y2)

        car_crop = img[y1:y2, x1:x2]
        if car_crop.size == 0:
            return jsonify({"error": "invalid crop"}), 400

        crop_h, crop_w = car_crop.shape[:2]

        # ===============================================
        # ORIENTATION
        # ===============================================
        orientation = detect_car_orientation(car_crop)

        if orientation == "left":
            zone_names = ["Aile avant", "Portes", "Aile arriere"]
        else:
            zone_names = ["Aile arriere", "Portes", "Aile avant"]

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
        # MOYENNE GLOBALE
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
        # RÉTROVISEUR
        # ===============================================
        mirror_box    = detect_mirror(car_crop, orientation, mask_body)
        mirror_result = None

        # ===============================================
        # DESSIN
        # ===============================================
        final_img = img.copy()
        cv2.rectangle(final_img, (x1, y1), (x2, y2), (220, 220, 220), 1)

        for cut in [cut1, cut2]:
            for dy in range(band_y1, band_y2, 12):
                cv2.line(
                    final_img,
                    (x1 + cut, y1 + dy),
                    (x1 + cut, y1 + dy + 6),
                    (255, 255, 255), 1
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
                elif diff < 14:
                    color_rect = (0, 165, 255)
                    verdict    = "Legere variation suspecte!"
                    detected  += 1
                else:
                    color_rect = (0, 210, 0)
                    verdict    = "OK"
                    detected  += 1

                label_score = str(int(diff))

            cv2.rectangle(final_img, (abs_x1, abs_y1),
                          (abs_x2, abs_y2), color_rect, 5)

            overlay = final_img.copy()
            cv2.rectangle(overlay, (abs_x1, abs_y1),
                          (abs_x2, abs_y1 + 55), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, final_img, 0.5, 0, final_img)

            cv2.putText(final_img, zone["name"],
                        (abs_x1 + 8, abs_y1 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(final_img, f"Ecart: {label_score}",
                        (abs_x1 + 8, abs_y1 + 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rect, 2)
            cv2.putText(final_img, verdict,
                        (abs_x1 + 8, abs_y2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_rect, 2)

            results_zones.append({
                "zone":    zone["name"],
                "diff":    round(diff, 1),
                "pixels":  px_count,
                "verdict": verdict
            })

        # ===============================================
        # DESSIN RÉTROVISEUR
        # ===============================================
        if mirror_box is not None:
            mx1, my1, mx2, my2 = mirror_box
            abs_mx1 = x1 + mx1
            abs_my1 = y1 + my1
            abs_mx2 = x1 + mx2
            abs_my2 = y1 + my2

            mirror_color, mirror_px = get_zone_color(
                hsv_full, mask_body, mx1, my1, mx2, my2
            )

            if mirror_color is not None:
                m_diff = float(np.linalg.norm(mirror_color - ref_color))
                if m_diff >= 14 and m_diff < 26:
                    m_color   = (0, 0, 255)
                    m_verdict = "Retro: peinture suspecte!"
                elif m_diff < 14:
                    m_color   = (0, 165, 255)
                    m_verdict = "Retro: variation legere"
                else:
                    m_color   = (0, 210, 0)
                    m_verdict = "Retro: OK"
            else:
                m_diff    = 0.0
                m_color   = (200, 200, 0)
                m_verdict = "Retro: detecte"

            cv2.rectangle(final_img,
                          (abs_mx1, abs_my1),
                          (abs_mx2, abs_my2),
                          (0, 255, 255), 3)

            overlay2 = final_img.copy()
            cv2.rectangle(overlay2,
                          (abs_mx1, abs_my2 + 2),
                          (abs_mx2, abs_my2 + 22),
                          (0, 0, 0), -1)
            cv2.addWeighted(overlay2, 0.55, final_img, 0.45, 0, final_img)

            cv2.putText(final_img, m_verdict,
                        (abs_mx1 + 2, abs_my2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 255, 255), 1)

            mirror_result = {
                "diff":    round(m_diff, 1),
                "verdict": m_verdict,
                "box":     [abs_mx1, abs_my1, abs_mx2, abs_my2]
            }

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

        cv2.putText(
            final_img,
            f"Ref: H={int(ref_color[0])} S={int(ref_color[1])} "
            f"V={int(ref_color[2])}  |  "
            f"Avant: {'GAUCHE' if orientation == 'left' else 'DROITE'}  |  "
            f"Retro: {'OUI' if mirror_box is not None else 'NON DETECTE'}",
            (10, img_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1
        )

        analysed_name = "analysed_" + filename
        analysed_path = os.path.join(UPLOAD_FOLDER, analysed_name)
        cv2.imwrite(analysed_path, final_img)

        if os.path.exists(resized_path):
            os.remove(resized_path)

        return jsonify({
            "yolo":           yolo_result,
            "score":          final_score,
            "result":         result,
            "zones":          results_zones,
            "mirror":         mirror_result,
            "zones_detected": detected,
            "orientation":    orientation,
            "reference_hsv": {
                "H": round(ref_color[0], 1),
                "S": round(ref_color[1], 1),
                "V": round(ref_color[2], 1)
            },
            "image_result":   analysed_name,
            "image_url":      request.host_url + "uploads/" + analysed_name
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
