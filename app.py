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
# DÉTECTER RÉTROVISEUR
# Cherche un petit rectangle
# sombre saillant sur le côté
# de la voiture (zone milieu)
# =========================
def detect_mirror(car_crop, orientation):
    """
    Le rétroviseur est un petit bloc sombre/coloré
    qui dépasse sur le côté de la voiture,
    situé entre 25% et 50% de la largeur du crop
    (côté avant) dans la bande verticale centrale.
    
    Retourne (mx1, my1, mx2, my2) en coords crop
    ou None si non trouvé.
    """
    crop_h, crop_w = car_crop.shape[:2]

    # Zone de recherche selon l'orientation
    # Le rétro est proche de l'avant, dans le tiers avant
    # verticalement : entre 25% et 55% de la hauteur
    if orientation == "left":
        # avant à gauche → rétro dans le tiers gauche
        search_x1 = int(crop_w * 0.10)
        search_x2 = int(crop_w * 0.42)
    else:
        # avant à droite → rétro dans le tiers droit
        search_x1 = int(crop_w * 0.58)
        search_x2 = int(crop_w * 0.90)

    search_y1 = int(crop_h * 0.22)
    search_y2 = int(crop_h * 0.55)

    search_zone = car_crop[search_y1:search_y2, search_x1:search_x2]
    if search_zone.size == 0:
        return None

    # Convertir en niveaux de gris
    gray = cv2.cvtColor(search_zone, cv2.COLOR_BGR2GRAY)

    # Blur léger pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Détection des contours forts (le rétro a des bords nets)
    edges = cv2.Canny(blurred, 40, 120)

    # Dilatation pour regrouper les contours proches
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # Trouver les contours fermés
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Filtrer : le rétro est un petit rectangle
    # Surface entre 0.3% et 5% de la zone de recherche
    zone_area = search_zone.shape[0] * search_zone.shape[1]
    min_area  = zone_area * 0.003
    max_area  = zone_area * 0.05

    best = None
    best_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Le rétro est plus large que haut (ratio)
        ratio = w / h if h > 0 else 0
        if ratio < 0.8 or ratio > 4.0:
            continue

        # Score : privilégier les contours nets et bien formés
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        compactness = (4 * np.pi * area) / (perimeter ** 2)
        score = area * compactness

        if score > best_score:
            best_score = score
            # Repasser en coords du crop complet
            best = (
                search_x1 + x,
                search_y1 + y,
                search_x1 + x + w,
                search_y1 + y + h
            )

    return best


# =========================
# DÉTECTER ORIENTATION
# 3 méthodes combinées + vote
# =========================
def detect_car_orientation(car_crop):
    """
    Méthode 1 : feux arrière rouges
    Méthode 2 : position du pare-brise (grande zone sombre haute)
    Méthode 3 : texture Sobel des bords (arrière = plus de détails)
    Vote majoritaire → "left" ou "right"
    """
    crop_h, crop_w = car_crop.shape[:2]
    votes_left  = 0
    votes_right = 0

    band_y1 = int(crop_h * 0.30)
    band_y2 = int(crop_h * 0.85)
    band_w  = int(crop_w * 0.22)

    left_zone  = car_crop[band_y1:band_y2, 0:band_w]
    right_zone = car_crop[band_y1:band_y2, crop_w - band_w:crop_w]

    # -----------------------------------------------
    # MÉTHODE 1 : feux rouges
    # -----------------------------------------------
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

    # -----------------------------------------------
    # MÉTHODE 2 : position du pare-brise
    # -----------------------------------------------
    gray     = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
    top_band = gray[int(crop_h * 0.10):int(crop_h * 0.55), :]
    dark_mask = (top_band < 80).astype(np.uint8)
    col_sums  = dark_mask.sum(axis=0).astype(float)
    col_sums  = np.convolve(col_sums, np.ones(20) / 20, mode='same')

    total = col_sums.sum()
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

    # -----------------------------------------------
    # MÉTHODE 3 : texture Sobel
    # -----------------------------------------------
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

    # -----------------------------------------------
    # VOTE FINAL
    # -----------------------------------------------
    return "left" if votes_left >= votes_right else "right"


# =========================
# ANALYSE
# =========================
@app.route("/analyse", methods=["POST"])
def analyse():
    try:
        if "image" not in request.files:
            return jsonify({"error": "no image"}), 400

        file = request.files["image"]
        filename = str(int(time.time())) + "_" + secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
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

        x1 = 0      if raw_x1 < 150           else max(0,     raw_x1 - 15)
        x2 = img_w  if (img_w - raw_x2) < 150 else min(img_w, raw_x2 + 15)
        y1 = 0      if raw_y1 < 80            else max(0,     raw_y1 - 10)
        y2 = img_h  if (img_h - raw_y2) < 80  else min(img_h, raw_y2 + 10)

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
        # RÉTROVISEUR
        # ===============================================
        mirror_box = detect_mirror(car_crop, orientation)

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
        # MOYENNE GLOBALE CARROSSERIE
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
        # 3 ZONES CARROSSERIE
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
        final_img = img.copy()

        # Contour total voiture
        cv2.rectangle(final_img, (x1, y1), (x2, y2), (220, 220, 220), 1)

        # Séparateurs pointillés entre zones
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

            # Grand rectangle zone
            cv2.rectangle(final_img, (abs_x1, abs_y1), (abs_x2, abs_y2),
                          color_rect, 5)

            # Fond texte
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
        mirror_result = None

        if mirror_box is not None:
            mx1, my1, mx2, my2 = mirror_box

            # Coords absolues dans final_img
            abs_mx1 = x1 + mx1
            abs_my1 = y1 + my1
            abs_mx2 = x1 + mx2
            abs_my2 = y1 + my2

            # Analyser la couleur du rétroviseur
            mirror_color, mirror_px = get_zone_color(
                hsv_full, mask_body, mx1, my1, mx2, my2
            )

            if mirror_color is not None:
                mirror_diff = float(np.linalg.norm(mirror_color - ref_color))

                if mirror_diff >= 14 and mirror_diff < 26:
                    mirror_color_rect = (0, 0, 255)
                    mirror_verdict    = "Retroviseur: peinture suspecte!"
                elif mirror_diff < 14:
                    mirror_color_rect = (0, 165, 255)
                    mirror_verdict    = "Retroviseur: variation legere"
                else:
                    mirror_color_rect = (0, 210, 0)
                    mirror_verdict    = "Retroviseur: OK"

                mirror_result = {
                    "diff":    round(mirror_diff, 1),
                    "verdict": mirror_verdict,
                    "box":     [abs_mx1, abs_my1, abs_mx2, abs_my2]
                }
            else:
                mirror_color_rect = (200, 200, 0)
                mirror_verdict    = "Retroviseur detecte"
                mirror_result     = {
                    "diff":    0,
                    "verdict": mirror_verdict,
                    "box":     [abs_mx1, abs_my1, abs_mx2, abs_my2]
                }

            # Rectangle rétroviseur (cyan, trait épais)
            cv2.rectangle(final_img,
                          (abs_mx1, abs_my1),
                          (abs_mx2, abs_my2),
                          (255, 255, 0), 3)

            # Label rétroviseur
            cv2.putText(final_img, mirror_verdict,
                        (abs_mx1, abs_my1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 0), 1)

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

        # Info bas de l'image
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
