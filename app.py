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
        # YOLO CALL
        # =========================
        yolo_result = call_yolo(path)

        # =========================
        # LOAD IMAGE
        # =========================
        img = cv2.imread(path)

        if img is None:
            return jsonify({"error": "image unreadable"}), 400

        img = cv2.resize(img, (900, 500))
        original = img.copy()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # =========================
        # EXTRACT CAR FROM YOLO
        # =========================
        detections = yolo_result.get("detections", [])

        cars = sorted(
            [d for d in detections if d.get("class") == 2],
            key=lambda x: x.get("conf", 0),
            reverse=True
        )

        if not cars:
            return jsonify({
                "error": "Car not detected by YOLO",
                "yolo": yolo_result
            }), 400

        x1, y1, x2, y2 = cars[0]["box"]

        h_img, w_img = gray.shape

        # clamp sécurité
        x1 = max(0, min(x1, w_img - 1))
        x2 = max(0, min(x2, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        y2 = max(0, min(y2, h_img - 1))

        car = gray[y1:y2, x1:x2]

        if car.size == 0:
            return jsonify({"error": "invalid YOLO crop"}), 400

        car = cv2.resize(car, (600, 300))

        # =========================
        # PARTS DETECTION (SIMPLIFIED)
        # =========================

        parts = {
            "capot": car[:150, :],
            "pare_choc_avant": car[0:100, :],
            "pare_choc_arriere": car[200:300, :],
            "porte_gauche": car[:, :200],
            "porte_droite": car[:, 400:],
        }

        def analyze(part):
            brightness = np.mean(part)
            texture = cv2.Laplacian(part, cv2.CV_64F).var()
            color_var = np.std(part)

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

        part_scores = {}
        total_score = 0
        detected_parts = 0

        for name, zone in parts.items():

            s = analyze(zone)
            part_scores[name] = int(s)

            if s >= 50:
                detected_parts += 1
                total_score += s

        final_score = int(min(total_score, 100))

        if final_score < 20:
            result = "Peinture d'origine (OK)"
        elif final_score < 45:
            result = "Légères retouches"
        elif final_score < 70:
            result = "Peinture suspecte"
        else:
            result = "Voiture probablement repeinte"

        analysed_name = "analysed_" + filename
        analysed_path = os.path.join(UPLOAD_FOLDER, analysed_name)

        cv2.imwrite(analysed_path, original)

        return jsonify({
            "yolo": yolo_result,
            "score": final_score,
            "result": result,
            "parts_score": part_scores,
            "detected_parts": detected_parts,
            "image_url": request.host_url + "uploads/" + analysed_name
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
