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
# DÉTECTER SENS VOITURE
# Retourne "left" si l'avant
# est à gauche, "right" sinon
# =========================
def detect_car_orientation(car_crop, detections, x1_car, y1_car):
    """
    Stratégie : chercher les feux avant (blancs/jaunes)
    vs feux arrière (rouges) dans les bandes gauche/droite.
    Si on trouve du rouge à gauche → avant est à droite.
    Si on trouve du rouge à droite → avant est à gauche.
    Par défaut : compare la saturation rouge des deux côtés.
    """
    crop_h, crop_w = car_crop.shape[:2]

    # Bande gauche et droite (20% de la largeur, zone basse = feux)
    band_w  = int(crop_w * 0.20)
    band_y1 = int(crop_h * 0.40)   # moitié basse = zone des feux
    band_y2 = int(crop_h * 0.85)

    left_band  = car_crop[band_y1:band_y2, 0:band_w]
    right_band = car_crop[band_y1:band_y2, crop_w - band_w:crop_w]

    # Convertir en HSV
    left_hsv  = cv2.cvtColor(left_band,  cv2.COLOR_BGR2HSV)
    right_hsv = cv2.cvtColor(right_band, cv2.COLOR_BGR2HSV)

    # Masque rouge : feux arrière sont rouges
    # Rouge en HSV : H dans [0-10] ou [170-180]
    def red_pixel_count(hsv_img):
        mask1 = cv2.inRange(hsv_img, (0,   80, 80), (10,  255, 255))
        mask2 = cv2.inRange(hsv_img, (170, 80, 80), (180, 255, 255))
        return cv2.countNonZero(cv2.bitwise_or(mask1, mask2))

    red_left  = red_pixel_count(left_hsv)
    red_right = red_pixel_count(right_hsv)

    # Plus de rouge à gauche → feux arrière à gauche → avant à droite
    if red_left > red_right * 1.5:
        return "right"   # avant à droite
    elif red_right > red_left * 1.5:
        return "left"    # avant à gauche
    else:
        # Pas de feux rouges clairs → on regarde aussi les détections YOLO
        # class 9 = traffic light, on peut aussi utiliser les coords des feux
        # Fallback : on compare la complexité de texture des deux côtés
        # (l'arrière a plus de détails = feux, pare-choc, plaque)
        left_gray  = cv2.cvtColor(left_band,  cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_band, cv2.COLOR_BGR2GRAY)
        left_std   = float(np.std(left_gray))
        right_std  = float(np.std(right_gray))

        # Plus de texture à droite → arrière à droite → avant à gauche
        if right_std > left_std * 1.2:
            return "left"
        else:
            return "right"


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
        # DÉTECTER L'ORIENTATION DE LA VOITURE
        # ===============================================
        orientation = detect_car_orientation(car_crop, detections, x1, y1)

        # Nommer les zones selon l'orientation détectée
        if orientation == "left":
            # Avant à gauche → ordre normal
            zone_names = ["Aile avant", "Portes", "Aile arriere"]
        else:
            # Avant à droite → ordre inversé
            zone_names = ["Aile arriere", "Portes", "Aile avant"]

        # ===============================================
        # MASQUE CARROSSERIE
        # ===============================================
        hsv_full = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)

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
        # 3 ZONES
        # ===============================================
        band_y1 = int(crop_h * 0.15)
        band_y2 = int(crop_h * 0.80)

        cut1 = int(crop_w * 0.33)
        cut2 = int(crop_w * 0.67)

        zones = [
            {"name": zone_names[0], "xA": 0,    "xB": cut1,   "yA": band_y1, "yB": band_y2},
            {"name": zone_names[1], "xA": cut1, "xB": cut2,   "yA": band_y1, "yB": band_y2},
            {"name": zone_names[2], "xA": cut2, "xB": crop_w, "yA": band_y1, "yB": band_y2},
        ]

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

            cv2.rectangle(final_img, (abs_x1, abs_y1), (abs_x2, abs_y2), color_rect, 5)

            overlay = final_img.copy()
            cv2.rectangle(overlay, (abs_x1, abs_y1), (abs_x2, abs_y1 + 55), (0, 0, 0), -1)
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

        # Référence + orientation détectée en bas
        cv2.putText(
            final_img,
            f"Ref: H={int(ref_color[0])} S={int(ref_color[1])} V={int(ref_color[2])}  |  Avant voiture: {'GAUCHE' if orientation == 'left' else 'DROITE'}",
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
