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
            r = requests.post(url, files=files, timeout=15)

        if r.status_code == 200:
            return r.json()

        return {"error": "YOLO failed", "status": r.status_code}

    except Exception as e:
        return {"error": "YOLO exception", "details": str(e)}


# =========================
# UPLOAD ROUTE
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
        "message": "WARRDI SCAN API"
    })


# =========================
# MAIN ANALYSE
# =========================
@app.route("/analyse", methods=["POST"])
def analyse():

    try:

        # CHECK IMAGE
        if 'image' not in request.files:
            return jsonify({"error": "no image"}), 400

        file = request.files['image']

        filename = str(int(time.time())) + "_" + secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)

        file.save(path)

        # =========================
        # YOLO CALL (REMOTE SERVER)
        # =========================
        yolo_result = call_yolo(path)

        # =========================
        # LOAD IMAGE
        # =========================
        img = cv2.imread(path)

        if img is None:
            return jsonify({"error": "image not readable"}), 400

        img = cv2.resize(img, (900, 500))
        original = img.copy()

        # =========================
        # REFLECTION REMOVAL
        # =========================
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask_reflet = cv2.inRange(
            hsv,
            np.array([0, 0, 220]),
            np.array([180, 60, 255])
        )

        img = cv2.inpaint(img, mask_reflet, 7, cv2.INPAINT_TELEA)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # =========================
        # CAR DETECTION VIA YOLO (BEST METHOD)
        # =========================
        car = None

        try:
            cars = [
                d for d in yolo_result.get("detections", [])
                if d["class"] == 2
            ]

            if cars:
                x1, y1, x2, y2 = cars[0]["box"]

                car = gray[y1:y2, x1:x2]

        except:
            car = None

        # FALLBACK IF YOLO FAIL
        if car is None or car.size == 0:

            edges = cv2.Canny(blur, 70, 140)
            contours, _ = cv2.findContours(
                edges,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if len(contours) == 0:
                return jsonify({"error": "no vehicle detected"}), 400

            car_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(car_contour)

            car = gray[y:y+h, x:x+w]

        car = cv2.resize(car, (600, 300))

        # =========================
        # DIVISION ZONES
        # =========================
        left = car[:, :200]
        center = car[:, 200:400]
        right = car[:, 400:]

        heatmap = np.zeros_like(car)

        score = 0
        zones = 0

        # =========================
        # ANALYSE ZONE FUNCTION
        # =========================
        def analyse_zone(zone, x_offset):

            nonlocal score, zones

            brightness = np.mean(zone)
            texture = cv2.Laplacian(zone, cv2.CV_64F).var()
            color_var = np.std(zone)

            local_score = 0

            if texture < 60:
                local_score += 40
            if texture > 250:
                local_score += 25
            if brightness > 175 or brightness < 65:
                local_score += 30
            if 80 < brightness < 120 and texture < 90:
                local_score += 35
            if color_var > 12:
                local_score += 30

            if local_score >= 50:

                zones += 1
                score += local_score

                h, w = zone.shape

                cv2.rectangle(
                    original,
                    (x_offset, 0),
                    (x_offset + w, h),
                    (0, 0, 255),
                    2
                )

                cv2.rectangle(
                    heatmap,
                    (x_offset, 0),
                    (x_offset + w, h),
                    255,
                    -1
                )

        # =========================
        # APPLY ANALYSIS
        # =========================
        analyse_zone(left, 0)
        analyse_zone(center, 200)
        analyse_zone(right, 400)

        # =========================
        # SYMMETRY CHECK
        # =========================
        left_mean = np.mean(left)
        right_mean = np.mean(right)

        symmetry_diff = abs(left_mean - right_mean)

        if symmetry_diff > 25:
            score += 20
            cv2.putText(
                original,
                "ASYMMETRIE PEINTURE",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # =========================
        # HEATMAP
        # =========================
        heat_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        heat_color = cv2.resize(
            heat_color,
            (original.shape[1], original.shape[0])
        )

        final = cv2.addWeighted(original, 0.85, heat_color, 0.35, 0)

        # =========================
        # FINAL SCORE
        # =========================
        score = int(min(score, 100))

        if score < 20:
            result = "Peinture d'origine (OK)"
        elif score < 45:
            result = "Légères retouches possibles"
        elif score < 70:
            result = "Peinture probablement refaite"
        else:
            result = "Forte suspicion de carrosserie repeinte"

        # =========================
        # SAVE RESULT
        # =========================
        analysed_name = "analysed_" + filename
        analysed_path = os.path.join(UPLOAD_FOLDER, analysed_name)

        cv2.imwrite(analysed_path, final)

        return jsonify({
            "yolo": yolo_result,
            "score": score,
            "result": result,
            "zones_detected": zones,
            "symmetry_diff": float(symmetry_diff),
            "image_result": analysed_name,
            "image_url": request.host_url + "uploads/" + analysed_name
        })

    except Exception as e:

        print(traceback.format_exc())

        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
