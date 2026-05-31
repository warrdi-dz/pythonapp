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
        # ANALYSE COULEUR DOMINANTE
        # =========================

        car_crop = img[y1:y2, x1:x2]

        if car_crop.size == 0:
            return jsonify({"error": "invalid crop"}), 400

        h, w, _ = car_crop.shape

        rows = 6
        cols = 8

        cell_h = h // rows
        cell_w = w // cols

        global_b = np.mean(car_crop[:, :, 0])
        global_g = np.mean(car_crop[:, :, 1])
        global_r = np.mean(car_crop[:, :, 2])

        global_color = np.array([
            global_b,
            global_g,
            global_r
        ])

        zones_scores = []
        detected = 0

        # image originale
        final_img = img.copy()

        for i in range(rows):
            for j in range(cols):

                local_y1 = i * cell_h
                local_y2 = (i + 1) * cell_h

                local_x1 = j * cell_w
                local_x2 = (j + 1) * cell_w

                zone = car_crop[
                    local_y1:local_y2,
                    local_x1:local_x2
                ]

                if zone.size == 0:
                    continue

                mean_b = np.mean(zone[:, :, 0])
                mean_g = np.mean(zone[:, :, 1])
                mean_r = np.mean(zone[:, :, 2])

                zone_color = np.array([
                    mean_b,
                    mean_g,
                    mean_r
                ])

                diff = np.linalg.norm(
                    zone_color - global_color
                )

                score = int(diff)

                zones_scores.append(score)

                abs_x1 = x1 + local_x1
                abs_y1 = y1 + local_y1
                abs_x2 = x1 + local_x2
                abs_y2 = y1 + local_y2

                # zone légèrement différente
                if diff > 40:

                    detected += 1

                    cv2.rectangle(
                        final_img,
                        (abs_x1, abs_y1),
                        (abs_x2, abs_y2),
                        (0, 255, 0),
                        2
                    )

                # zone très différente
                if diff > 70:

                    cv2.rectangle(
                        final_img,
                        (abs_x1, abs_y1),
                        (abs_x2, abs_y2),
                        (0, 0, 255),
                        3
                    )

        # =========================
        # SCORE FINAL
        # =========================

        if len(zones_scores) == 0:
            final_score = 0
        else:
            final_score = int(np.max(zones_scores))

        final_score = min(final_score, 100)

        if final_score < 20:
            result = "Peinture d'origine (OK)"
        elif final_score < 40:
            result = "Différence légère détectée"
        elif final_score < 70:
            result = "Peinture suspecte"
        else:
            result = "Zone fortement différente"

        # =========================
        # SAVE IMAGE
        # =========================

        analysed_name = "analysed_" + filename
        analysed_path = os.path.join(
            UPLOAD_FOLDER,
            analysed_name
        )

        cv2.imwrite(
            analysed_path,
            final_img
        )

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

if __name__ == "__main__":
     app.run(host="0.0.0.0", port=5000)
