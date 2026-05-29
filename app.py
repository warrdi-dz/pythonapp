from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import traceback
import time
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return jsonify({
        "status": "OK",
        "message": "WARRDI SCAN AUTO AI"
    })


@app.route("/analyse", methods=["POST"])
def analyse():

    try:

        if 'image' not in request.files:
            return jsonify({
                "error": "no image"
            }), 400

        file = request.files['image']

        filename = str(int(time.time())) + "_" + secure_filename(file.filename)

        path = os.path.join(UPLOAD_FOLDER, filename)

        file.save(path)

        img = cv2.imread(path)

        if img is None:
            return jsonify({
                "error": "image not readable"
            }), 400

        # =========================
        # REDIMENSIONNEMENT
        # =========================

        img = cv2.resize(img, (1200, 700))

        original = img.copy()

        # =========================
        # PREPARATION
        # =========================

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # =========================
        # ANALYSE TEXTURE
        # =========================

        laplacian = cv2.Laplacian(blur, cv2.CV_64F)

        texture_map = np.uint8(np.absolute(laplacian))

        # =========================
        # ANALYSE LUMINOSITE
        # =========================

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        value_channel = hsv[:, :, 2]

        # =========================
        # ANALYSE REFLETS
        # =========================

        reflection = cv2.GaussianBlur(value_channel, (31, 31), 0)

        diff_reflection = cv2.absdiff(value_channel, reflection)

        # =========================
        # SCORE MAP
        # =========================

        combined = cv2.addWeighted(
            texture_map,
            0.6,
            diff_reflection,
            0.4,
            0
        )

        # =========================
        # THRESHOLD
        # =========================

        _, thresh = cv2.threshold(
            combined,
            40,
            255,
            cv2.THRESH_BINARY
        )

        kernel = np.ones((5, 5), np.uint8)

        thresh = cv2.morphologyEx(
            thresh,
            cv2.MORPH_CLOSE,
            kernel
        )

        # =========================
        # CONTOURS
        # =========================

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        suspect_count = 0

        total_area = 0

        heatmap = np.zeros_like(gray)

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area > 300:

                x, y, w, h = cv2.boundingRect(cnt)

                roi_gray = gray[y:y+h, x:x+w]

                brightness = np.mean(roi_gray)

                texture = cv2.Laplacian(
                    roi_gray,
                    cv2.CV_64F
                ).var()

                # =========================
                # DETECTION SUSPECTE
                # =========================

                suspicious = False

                if texture < 180:
                    suspicious = True

                if brightness > 170:
                    suspicious = True

                if suspicious:

                    suspect_count += 1

                    total_area += area

                    # Rectangle rouge
                    cv2.rectangle(
                        original,
                        (x, y),
                        (x+w, y+h),
                        (0, 0, 255),
                        3
                    )

                    # Texte
                    cv2.putText(
                        original,
                        "ZONE SUSPECTE",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                    # Heatmap
                    cv2.drawContours(
                        heatmap,
                        [cnt],
                        -1,
                        255,
                        -1
                    )

        # =========================
        # HEATMAP ROUGE
        # =========================

        heatmap_color = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        final = cv2.addWeighted(
            original,
            0.8,
            heatmap_color,
            0.4,
            0
        )

        # =========================
        # SCORE GLOBAL
        # =========================

        image_area = img.shape[0] * img.shape[1]

        ratio = total_area / image_area

        score = int(min(ratio * 1000, 100))

        if score < 20:
            result = "Aucune anomalie importante"
        elif score < 50:
            result = "Quelques incoherences detectees"
        elif score < 75:
            result = "Peinture probablement refaite"
        else:
            result = "Forte suspicion de peinture refaite"

        # =========================
        # SAUVEGARDE IMAGE
        # =========================

        analysed_name = "analysed_" + filename

        analysed_path = os.path.join(
            UPLOAD_FOLDER,
            analysed_name
        )

        cv2.imwrite(analysed_path, final)

        return jsonify({
            "score": score,
            "result": result,
            "zones_detected": suspect_count,
            "image_result": analysed_name
        })

    except Exception as e:

        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )


