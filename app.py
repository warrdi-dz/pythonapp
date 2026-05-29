from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
import time
import traceback
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# HOME
# =========================

@app.route("/")
def home():
    return jsonify({
        "status": "OK",
        "message": "WARRDI STABLE AI SCAN"
    })

# =========================
# SERVE IMAGES
# =========================

@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# =========================
# ANALYSE IA STABLE
# =========================

@app.route("/analyse", methods=["POST"])
def analyse():

    try:

        if 'image' not in request.files:
            return jsonify({"error": "no image"}), 400

        file = request.files['image']

        filename = str(int(time.time())) + "_" + secure_filename(file.filename)

        path = os.path.join(UPLOAD_FOLDER, filename)

        file.save(path)

        img = cv2.imread(path)

        if img is None:
            return jsonify({"error": "image not readable"}), 400

        # =========================
        # REDUCTION MEMOIRE
        # =========================

        img = cv2.resize(img, (800, 450))

        original = img.copy()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # =========================
        # EDGE + TEXTURE MAP
        # =========================

        lap = cv2.Laplacian(blur, cv2.CV_64F)
        texture = np.uint8(np.absolute(lap))

        # =========================
        # REFLECTION ANALYSIS
        # =========================

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]

        blur_v = cv2.GaussianBlur(v, (25, 25), 0)
        diff_v = cv2.absdiff(v, blur_v)

        # =========================
        # COMBINE MAP
        # =========================

        combined = cv2.addWeighted(texture, 0.6, diff_v, 0.4, 0)

        _, thresh = cv2.threshold(combined, 35, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        heatmap = np.zeros(gray.shape, dtype=np.uint8)

        zones = 0
        score = 0

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area < 800:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            roi = gray[y:y+h, x:x+w]

            if roi.size == 0:
                continue

            brightness = np.mean(roi)
            tex = cv2.Laplacian(roi, cv2.CV_64F).var()

            # =========================
            # LOGIQUE PLUS STABLE
            # =========================

            suspicious = False

            if tex < 120:
                suspicious = True

            if brightness > 150:
                suspicious = True

            if 60 < brightness < 90:
                suspicious = True  # peinture possible (zone réfléchissante)

            if suspicious:

                zones += 1
                score += int(min(area / 1000, 20))

                cv2.rectangle(
                    original,
                    (x, y),
                    (x+w, y+h),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    original,
                    "SUSPECT",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

                cv2.rectangle(
                    heatmap,
                    (x, y),
                    (x+w, y+h),
                    255,
                    -1
                )

        # =========================
        # HEATMAP FINAL
        # =========================

        heat_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        final = cv2.addWeighted(original, 0.85, heat_color, 0.35, 0)

        # =========================
        # SCORE FINAL
        # =========================

        score = min(score, 100)

        if score < 20:
            result = "Aucune anomalie importante"
        elif score < 50:
            result = "Quelques zones suspectes"
        elif score < 75:
            result = "Peinture probablement retouchee"
        else:
            result = "Forte suspicion de peinture refaite"

        # =========================
        # SAVE IMAGE
        # =========================

        analysed_name = "analysed_" + filename

        analysed_path = os.path.join(UPLOAD_FOLDER, analysed_name)

        cv2.imwrite(analysed_path, final)

        return jsonify({
            "score": score,
            "result": result,
            "zones_detected": zones,
            "image_result": analysed_name
        })

    except Exception as e:

        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500

# =========================
# START
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
