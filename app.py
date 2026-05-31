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
    return jsonify({
        "status": "OK",
        "message": "GARAGE PRO V4 API"
    })


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

        yolo_result = call_yolo(path)

        img = cv2.imread(path)
        if img is None:
            return jsonify({"error": "image unreadable"}), 400

        img = cv2.resize(img, (900, 500))
        img_h, img_w = img.shape[:2]

        detections = yolo_result.get("detections", [])
        cars = [d for d in detections if d.get("class") == 2]
        if not cars:
            return jsonify({"error": "Car not detected"}), 400

        # ===============================================
        # BOUNDING BOX : ON PREND TOUTE LA LARGEUR
        # de l'image si la voiture touche les bords
        # YOLO rate souvent la partie droite ou gauche
        # ===============================================
        raw_x1 = min(d["box"][0] for d in cars)
        raw_y1 = min(d["box"][1] for d in cars)
        raw_x2 = max(d["box"][2] for d in cars)
        raw_y2 = max(d["box"][3] for d in cars)

        # Si YOLO s'arrête à moins de 150px du bord → on force jusqu'au bord
        x1 = 0         if raw_x1 < 150              else max(0,     raw_x1 - 20)
        x2 = img_w     if (img_w - raw_x2) < 150    else min(img_w, raw_x2 + 20)
        y1 = 0         if raw_y1 < 80               else max(0,     raw_y1 - 15)
        y2 = img_h     if (img_h - raw_y2) < 80     else min(img_h, raw_y2 + 15)

        car_crop = img[y1:y2, x1:x2]
        if car_crop.size == 0:
            return jsonify({"error": "invalid crop"}), 400

        crop_h, crop_w = car_crop.shape[:2]

        # ===============================================
        # MASQUE CARROSSERIE
        # Exclure vitres / roues / fond / ciel
        # ===============================================
        hsv_full = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)

        # Trop sombre → vitres, pneus, joints (V < 45)
        mask_dark = cv2.inRange(hsv_full, (0, 0, 0), (180, 255, 45))

        # Blanc pur désaturé → ciel / fond (S < 18, V > 210)
        mask_sky = cv2.inRange(hsv_full, (0, 0, 210), (180, 18, 255))

        # Carrosserie = tout sauf dark + sky
        mask_body = cv2.bitwise_not(cv2.bitwise_or(mask_dark, mask_sky))

        # Morphologie : boucher les petits trous dans le masque
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_CLOSE, kernel)

        # ===============================================
        # ÉTAPE 1 : MOYENNE GLOBALE CARROSSERIE
        # sur toute la voiture (roue à roue)
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
        # ÉTAPE 2 : 3 ZONES HORIZONTALES
        #
        # On définit la bande carrosserie verticalement :
        # - haut  : on ignore le toit (15% haut)
        # - bas   : on ignore les roues (80% bas)
        #
        # Horizontalement : 3 zones égales sur la largeur
        # pour s'adapter à n'importe quelle voiture
        #
        #  |--- 33% ---|--- 34% ---|--- 33% ---|
        #  aile avant    portes     aile arrière
        # ===============================================

        band_y1 = int(crop_h * 0.15)
        band_y2 = int(crop_h * 0.80)

        cut1 = int(crop_w * 0.33)
        cut2 = int(crop_w * 0.67)

        zones = [
            {
                "name": "Aile avant",
                "xA": 0,     "xB": cut1,
                "yA": band_y1, "yB": band_y2
            },
            {
                "name": "Portes",
                "xA": cut1,  "xB": cut2,
                "yA": band_y1, "yB": band_y2
            },
            {
                "name": "Aile arriere",
                "xA": cut2,  "xB": crop_w,
                "yA": band_y1, "yB": band_y2
            },
        ]

        # ===============================================
        # ÉTAPE 3 : COMPARER CHAQUE ZONE
        # à la moyenne globale
        # ===============================================
        final_img = img.copy()

        # Contour total voiture (blanc fin)
        cv2.rectangle(final_img, (x1, y1), (x2, y2), (220, 220, 220), 1)

        # Ligne de séparation des zones (pointillés blancs)
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

                if diff < 10:
                    color_rect = (0, 210, 0)       # vert  — OK
                    verdict    = "OK"
                elif diff < 28:
                    color_rect = (0, 165, 255)     # orange — légère variation
                    verdict    = "Legere variation"
                    detected  += 1
                else:
                    color_rect = (0, 0, 255)       # rouge — suspect
                    verdict    = "SUSPECT - verifier"
                    detected  += 1

                label_score = str(int(diff))

            # --- Grand rectangle zone ---
            cv2.rectangle(
                final_img,
                (abs_x1, abs_y1),
                (abs_x2, abs_y2),
                color_rect,
                5
            )

            # --- Fond noir semi-transparent pour lisibilité texte ---
            overlay = final_img.copy()
            cv2.rectangle(
                overlay,
                (abs_x1,     abs_y1),
                (abs_x2,     abs_y1 + 52),
                (0, 0, 0), -1
            )
            cv2.addWeighted(overlay, 0.5, final_img, 0.5, 0, final_img)

            # --- Nom de la zone ---
            cv2.putText(
                final_img,
                zone["name"],
                (abs_x1 + 8, abs_y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # --- Écart + verdict ---
            cv2.putText(
                final_img,
                f"Ecart: {label_score}",
                (abs_x1 + 8, abs_y1 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color_rect,
                2
            )

            # --- Verdict en bas du rectangle ---
            cv2.putText(
                final_img,
                verdict,
                (abs_x1 + 8, abs_y2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color_rect,
                2
            )

            results_zones.append({
                "zone":      zone["name"],
                "diff":      round(diff, 1),
                "pixels":    px_count,
                "verdict":   verdict
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

        # Teinte de référence en bas de l'image
        cv2.putText(
            final_img,
            f"Ref globale : H={int(ref_color[0])}  S={int(ref_color[1])}  V={int(ref_color[2])}",
            (10, img_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1
        )

        analysed_name = "analysed_" + filename
        analysed_path = os.path.join(UPLOAD_FOLDER, analysed_name)
        cv2.imwrite(analysed_path, final_img)

        return jsonify({
            "yolo":           yolo_result,
            "score":          final_score,
            "result":         result,
            "zones":          results_zones,
            "zones_detected": detected,
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
