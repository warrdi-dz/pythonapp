from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
from werkzeug.utils import secure_filename

import cv2
import numpy as np
import os
import time
import threading

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = YOLO("yolov8n.pt")

# =========================
# MEMORY STORE (simple)
# =========================

jobs = {}

# =========================
# SERVE FILES
# =========================

@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# =========================
# BACKGROUND PROCESS
# =========================

def process_job(job_id, path, filename):

    try:

        img = cv2.imread(path)
        img = cv2.resize(img, (800, 450))
        original = img.copy()

        results = model(img)

        heatmap = np.zeros(img.shape[:2], dtype=np.uint8)

        zones = 0
        score_total = 0

        for r in results:

            for box in r.boxes:

                if int(box.cls[0]) != 2:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                car = img[y1:y2, x1:x2]

                if car.size == 0:
                    continue

                gray = cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5,5), 0)
                lap = cv2.Laplacian(blur, cv2.CV_64F)

                texture = np.var(lap)

                if texture < 300:

                    zones += 1
                    score_total += 15

                    cv2.rectangle(
                        original,
                        (x1, y1),
                        (x2, y2),
                        (0,0,255),
                        2
                    )

                    cv2.putText(
                        original,
                        "SUSPECT",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,0,255),
                        2
                    )

                    cv2.rectangle(
                        heatmap,
                        (x1, y1),
                        (x2, y2),
                        255,
                        -1
                    )

        heatmap_color = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        final = cv2.addWeighted(original, 0.85, heatmap_color, 0.35, 0)

        analysed_name = "analysed_" + filename

        analysed_path = os.path.join(UPLOAD_FOLDER, analysed_name)

        cv2.imwrite(analysed_path, final)

        score = min(score_total, 100)

        if score < 30:
            result = "Aucune anomalie"
        elif score < 60:
            result = "Suspicion moyenne"
        else:
            result = "Peinture probablement refaite"

        jobs[job_id] = {
            "status": "done",
            "score": score,
            "result": result,
            "zones": zones,
            "image": analysed_name
        }

    except Exception as e:

        jobs[job_id] = {
            "status": "error",
            "error": str(e)
        }

# =========================
# CREATE JOB
# =========================

@app.route("/analyse", methods=["POST"])
def analyse():

    if 'image' not in request.files:
        return jsonify({"error": "no image"}), 400

    file = request.files['image']

    filename = str(int(time.time())) + "_" + secure_filename(file.filename)

    path = os.path.join(UPLOAD_FOLDER, filename)

    file.save(path)

    job_id = str(int(time.time()*1000))

    jobs[job_id] = {
        "status": "processing"
    }

    thread = threading.Thread(
        target=process_job,
        args=(job_id, path, filename)
    )

    thread.start()

    return jsonify({
        "job_id": job_id,
        "status": "processing"
    })

# =========================
# GET RESULT
# =========================

@app.route("/result/<job_id>")
def result(job_id):

    if job_id not in jobs:
        return jsonify({"error": "job not found"}), 404

    return jsonify(jobs[job_id])

# =========================
# START
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
