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
# YOLO API CALL (REMOTE)
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
        "message": "WARRDI SCAN API + YOLO"
    })


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
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

        x1 = max(0, min(x1, w_img - 1))
        x2 = max(0, min(x2, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        y2 = max(0, min(y2, h_img - 1))

        car_gray = cv2.resize(gray[y1:y2, x1:x2], (600, 300))
        car_color = cv2.resize(img[y1:y2, x1:x2], (600, 300))

        heatmap = np.zeros((300, 600), dtype=np.uint8)

        def score_zone(zone):
            brightness = np.mean(zone)
            texture = cv2.Laplacian(zone, cv2.CV_64F).var()
            color_var = np.std(zone)

            score = 0
            if texture < 60:
                score += 40
            if texture > 250:
                score += 25
            if brightness > 175 or brightness < 65:
                score += 30
            if 80 < brightness < 120 and texture < 90:
                score += 35
            if color_var > 12:
                score += 30

            return score

        zones = {
            "gauche": (0, 200),
            "centre": (200, 400),
            "droite": (400, 600)
        }

        zones_scores = {}
        detected = 0

        for name, (xA, xB) in zones.items():

            zone = car_gray[:, xA:xB]
            s = score_zone(zone)
            zones_scores[name] = int(s)

            if s >= 50:
                detected += 1

            if s >= 70:
                heat_value = 255
                color_rect = (0, 0, 255)
            elif s >= 50:
                heat_value = 160
                color_rect = (0, 165, 255)
            else:
                heat_value = 0
                color_rect = None

            heatmap[:, xA:xB] = heat_value

            if color_rect:
                cv2.rectangle(car_color, (xA, 0), (xB, 300), color_rect, 2)

        final_score = int(np.mean(list(zones_scores.values())))

        if final_score < 20:
            result = "Peinture d'origine (OK)"
        elif final_score < 45:
            result = "Légères retouches"
        elif final_score < 70:
            result = "Peinture suspecte"
        else:
            result = "Voiture probablement repeinte"

        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        final_img = cv2.addWeighted(car_color, 0.7, heatmap_color, 0.3, 0)

        analysed_name = "analysed_" + filename
        analysed_path = os.path.join(UPLOAD_FOLDER, analysed_name)

        cv2.imwrite(analysed_path, final_img)

        return jsonify({
            "yolo": yolo_result,
            "score": final_score,
            "result": result,
            "zones_scores": zones_scores,
            "zones_detected": detected,
            "image_result": analysed_name,
            "image_url": request.host_url + "uploads/" + analysed_name
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500
