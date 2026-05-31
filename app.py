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

        # =========================
        # YOLO
        # =========================
        yolo_result = call_yolo(path)

        img = cv2.imread(path)
        if img is None:
            return jsonify({"error": "image unreadable"}), 400

        img = cv2.resize(img, (900, 500))
        img_h, img_w = img.shape[:2]

        detections = yolo_result.get("detections", [])

        # --- Toutes les détections voiture (class 2) ---
        cars = [d for d in detections if d.get("class") == 2]

        if not cars:
            return jsonify({"error": "Car not detected"}), 400

        # =========================
        # FUSION DE TOUTES LES
        # BOUNDING BOXES VOITURE
        # =========================
        PADDING = 25

        x1 = max(0,     min(d["box"][0] for d in cars) - PADDING)
        y1 = max(0,     min(d["box"][1] for d in cars) - PADDING)
        x2 = min(img_w, max(d["box"][2] for d in cars) + PADDING)
        y2 = min(img_h, max(d["box"][3] for d in cars) + PADDING)

        # Si la voiture touche les bords, on force jusqu'au bord
        if (img_w - x2) < 60:
            x2 = img_w
        if (img_h - y2) < 60:
            y2 = img_h
        if x1 < 60:
            x1 = 0
        if y1 < 60:
            y1 = 0

        car_crop = img[y1:y2, x1:x2]

        if car_crop.size == 0:
            return jsonify({"error": "invalid crop"}), 400

        # =========================
        # HSV GLOBAL
        # =========================
        car_hsv = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)

        global_color = np.array([
            np.median(car_hsv[:, :, 0]),
            np.median(car_hsv[:, :, 1]),
            np.median(car_hsv[:, :, 2])
        ])

        # =========================
        # GRILLE 4x6 (plus fine)
        # =========================
        rows, cols = 4, 6
        h, w, _ = car_crop.shape

        cell_h = h // rows
        cell_w = w // cols

        final_img = img.copy()

        # Dessiner le contour de la zone analysée
        cv2.rectangle(
            final_img,
            (x1, y1),
            (x2, y2),
            (255, 165, 0),   # orange
            2
        )

        zones_scores = []
        detected = 0

        for i in range(rows):
            for j in range(cols):

                yA, yB = i * cell_h, (i + 1) * cell_h
                xA, xB = j * cell_w, (j + 1) * cell_w

                zone = car_crop[yA:yB, xA:xB]
                if zone.size == 0:
                    continue

                zone_hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)

                zone_color = np.array([
                    np.median(zone_hsv[:, :, 0]),
                    np.median(zone_hsv[:, :, 1]),
                    np.median(zone_hsv[:, :, 2])
                ])

                diff = np.linalg.norm(zone_color - global_color)
                zones_scores.append(diff)

                abs_x1 = x1 + xA
                abs_y1 = y1 + yA
                abs_x2 = x1 + xB
                abs_y2 = y1 + yB

                if diff > 20:
                    # Vert = légère variation, Rouge = forte différence
                    color     = (0, 255, 0) if diff < 45 else (0, 0, 255)
                    thickness = 2           if diff < 45 else 3

                    cv2.rectangle(
                        final_img,
                        (abs_x1, abs_y1),
                        (abs_x2, abs_y2),
                        color,
                        thickness
                    )
                    detected += 1

        final_score = int(np.mean(zones_scores)) if zones_scores else 0
        final_score = min(final_score, 100)

        if final_score < 20:
            result = "Peinture homogène (OK)"
        elif final_score < 40:
            result = "Légères variations"
        elif final_score < 60:
            result = "Zones suspectes"
        else:
            result = "Différence importante"

        analysed_name = "analysed_" + filename
        analysed_path = os.path.join(UPLOAD_FOLDER, analysed_name)

        cv2.imwrite(analysed_path, final_img)

        return jsonify({
            "yolo":           yolo_result,
            "score":          final_score,
            "result":         result,
            "zones_detected": detected,
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
