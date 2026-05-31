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
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # =========================
        # DETECT CAR
        # =========================
        detections = yolo_result.get("detections", [])

        cars = sorted(
            [d for d in detections if d.get("class") == 2],
            key=lambda x: x.get("conf", 0),
            reverse=True
        )

        if not cars:
            return jsonify({"error": "Car not detected"}), 400
        
        x1, y1, x2, y2 = cars[0]["box"]

        h_img, w_img = gray.shape

        pad = 0

        x1 = max(0, x1 + pad)
        y1 = max(0, y1 + pad)
        x2 = max(0, x2 - pad)
        y2 = max(0, y2 - pad)

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

        detections = yolo_result.get("detections", [])

        cars = sorted(
            [d for d in detections if d.get("class") == 2],
            key=lambda x: x.get("conf", 0),
            reverse=True
        )

        if not cars:
            return jsonify({"error": "Car not detected"}), 400

        x1, y1, x2, y2 = cars[0]["box"]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = max(0, x2)
        y2 = max(0, y2)

        # =========================
        # EXTRACTION VOITURE
        # =========================
        car_crop = img[y1:y2, x1:x2]

        if car_crop.size == 0:
            return jsonify({"error": "invalid crop"}), 400

        # =========================
        # HSV (plus stable couleur)
        # =========================
        car_hsv = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)

        h_mean = np.median(car_hsv[:, :, 0])
        s_mean = np.median(car_hsv[:, :, 1])
        v_mean = np.median(car_hsv[:, :, 2])

        global_color = np.array([h_mean, s_mean, v_mean])

        # =========================
        # GRID 3x3 (grandes zones)
        # =========================
        rows = 3
        cols = 3

        h, w, _ = car_crop.shape
        cell_h = h // rows
        cell_w = w // cols

        final_img = img.copy()

        zones_scores = []
        detected = 0

        # =========================
        # ANALYSE ZONES
        # =========================
        for i in range(rows):
            for j in range(cols):

                yA = i * cell_h
                yB = (i + 1) * cell_h
                xA = j * cell_w
                xB = (j + 1) * cell_w

                zone = car_crop[yA:yB, xA:xB]

                if zone.size == 0:
                    continue

                zone_hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)

                zh = np.median(zone_hsv[:, :, 0])
                zs = np.median(zone_hsv[:, :, 1])
                zv = np.median(zone_hsv[:, :, 2])

                zone_color = np.array([zh, zs, zv])

                # différence couleur
                diff = np.linalg.norm(zone_color - global_color)

                score = int(diff)
                zones_scores.append(score)

                # coordonnées image originale
                abs_x1 = x1 + xA
                abs_y1 = y1 + yA
                abs_x2 = x1 + xB
                abs_y2 = y1 + yB

                # =========================
                # LOGIQUE D'ALERTE
                # =========================

                # zone normale
                if diff < 20:
                    continue

                # zone suspecte
                if 20 <= diff < 45:
                    cv2.rectangle(
                        final_img,
                        (abs_x1, abs_y1),
                        (abs_x2, abs_y2),
                        (0, 255, 0),
                        2
                    )
                    detected += 1

                # zone très différente
                elif diff >= 45:
                    cv2.rectangle(
                        final_img,
                        (abs_x1, abs_y1),
                        (abs_x2, abs_y2),
                        (0, 0, 255),
                        3
                    )
                    detected += 1

        # =========================
        # SCORE FINAL
        # =========================
        if len(zones_scores) == 0:
            final_score = 0
        else:
            final_score = int(np.mean(zones_scores))

        final_score = min(final_score, 100)

        if final_score < 20:
            result = "Peinture homogène (OK)"
        elif final_score < 40:
            result = "Légères variations"
        elif final_score < 60:
            result = "Zones suspectes"
        else:
            result = "Différence importante détectée"

        # =========================
        # SAVE IMAGE
        # =========================
        analysed_name = "analysed_" + filename
        analysed_path = os.path.join(UPLOAD_FOLDER, analysed_name)

        cv2.imwrite(analysed_path, final_img)

        return jsonify({
            "yolo": yolo_result,
            "score": final_score,
            "result": result,
            "zones_detected": detected,
            "image_result": analysed_name,
            "image_url": request.host_url + "uploads/" + analysed_name
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500
