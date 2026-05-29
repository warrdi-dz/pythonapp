from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
from werkzeug.utils import secure_filename

import cv2
import numpy as np
import os
import traceback
import time

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# YOLO MODEL
# =========================

model = YOLO("yolov8n.pt")

# =========================
# HOME
# =========================

@app.route("/")
def home():

    return jsonify({
        "status": "OK",
        "message": "WARRDI AI EXPERT"
    })

# =========================
# SHOW IMAGE
# =========================

@app.route('/uploads/<filename>')
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )

# =========================
# ANALYSE
# =========================

@app.route("/analyse", methods=["POST"])
def analyse():

    try:

        if 'image' not in request.files:

            return jsonify({
                "error": "no image"
            }), 400

        file = request.files['image']

        filename = str(
            int(time.time())
        ) + "_" + secure_filename(file.filename)

        path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        file.save(path)

        img = cv2.imread(path)

        if img is None:

            return jsonify({
                "error": "image not readable"
            }), 400

        original = img.copy()

        # =========================
        # YOLO DETECTION
        # =========================

        results = model(img)

        car_found = False

        total_score = 0

        zones = 0

        heatmap = np.zeros(
            img.shape[:2],
            dtype=np.uint8
        )

        for r in results:

            boxes = r.boxes

            for box in boxes:

                cls = int(box.cls[0])

                # COCO class 2 = car
                if cls == 2:

                    car_found = True

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    car = img[y1:y2, x1:x2]

                    if car.size == 0:
                        continue

                    gray = cv2.cvtColor(
                        car,
                        cv2.COLOR_BGR2GRAY
                    )

                    blur = cv2.GaussianBlur(
                        gray,
                        (5,5),
                        0
                    )

                    laplacian = cv2.Laplacian(
                        blur,
                        cv2.CV_64F
                    )

                    texture_map = np.uint8(
                        np.absolute(laplacian)
                    )

                    _, thresh = cv2.threshold(
                        texture_map,
                        35,
                        255,
                        cv2.THRESH_BINARY
                    )

                    contours, _ = cv2.findContours(
                        thresh,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )

                    for cnt in contours:

                        area = cv2.contourArea(cnt)

                        if area > 200:

                            xx, yy, ww, hh = cv2.boundingRect(cnt)

                            roi = gray[
                                yy:yy+hh,
                                xx:xx+ww
                            ]

                            if roi.size == 0:
                                continue

                            brightness = np.mean(roi)

                            texture = cv2.Laplacian(
                                roi,
                                cv2.CV_64F
                            ).var()

                            suspicious = False

                            if texture < 350:
                                suspicious = True

                            if brightness > 140:
                                suspicious = True

                            if suspicious:

                                zones += 1

                                zone_score = min(
                                    int(area / 200),
                                    25
                                )

                                total_score += zone_score

                                # GLOBAL COORDS
                                gx1 = x1 + xx
                                gy1 = y1 + yy
                                gx2 = gx1 + ww
                                gy2 = gy1 + hh

                                # RED RECTANGLE
                                cv2.rectangle(
                                    original,
                                    (gx1, gy1),
                                    (gx2, gy2),
                                    (0,0,255),
                                    3
                                )

                                cv2.putText(
                                    original,
                                    "Paint Suspect",
                                    (gx1, gy1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0,0,255),
                                    2
                                )

                                # HEATMAP
                                cv2.rectangle(
                                    heatmap,
                                    (gx1, gy1),
                                    (gx2, gy2),
                                    255,
                                    -1
                                )

        if not car_found:

            return jsonify({
                "error": "no car detected"
            }), 400

        # =========================
        # FINAL SCORE
        # =========================

        score = min(total_score, 100)

        if score < 20:

            result = "Aucune anomalie importante"

        elif score < 50:

            result = "Quelques anomalies detectees"

        elif score < 75:

            result = "Peinture probablement refaite"

        else:

            result = "Forte suspicion de peinture refaite"

        # =========================
        # HEATMAP
        # =========================

        heatmap_color = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        final = cv2.addWeighted(
            original,
            0.85,
            heatmap_color,
            0.35,
            0
        )

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
            final
        )

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

    app.run(
        host="0.0.0.0",
        port=5000
    )
