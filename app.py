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

        car_gray = gray[y1:y2, x1:x2]

        # 🔥 améliore contraste (IMPORTANT)
        car_gray = cv2.equalizeHist(car_gray)

        car_gray = cv2.resize(car_gray, (600, 300))
        car_color = cv2.resize(img[y1:y2, x1:x2], (600, 300))

        # =========================
        # GRID HEATMAP
        # =========================
        h, w = car_gray.shape

        rows = 4
        cols = 6

        cell_h = h // rows
        cell_w = w // cols

        heatmap = np.zeros((h, w), dtype=np.uint8)

        zones_scores = []
        detected = 0

        def score_zone(zone):
            brightness = np.mean(zone)
            texture = cv2.Laplacian(zone, cv2.CV_64F).var()
            color_var = np.std(zone)

            score = 0

# texture
            if texture < 30:
               score += 40
            elif texture < 60:
               score += 25
            elif texture < 100:
             score += 10

          # luminosité
            if brightness < 70 or brightness > 180:
             score += 30

# variation
            if color_var > 15:
             score += 30
            elif color_var > 8:
             score += 15

            return score

        for i in range(rows):
            for j in range(cols):

                yA = i * cell_h
                yB = (i + 1) * cell_h
                xA = j * cell_w
                xB = (j + 1) * cell_w

                zone = car_gray[yA:yB, xA:xB]
                s = score_zone(zone)

                zones_scores.append(s)

                # HEAT COLOR
                if s >= 60:
                     heatmap[yA:yB, xA:xB] = 255
                     color = (0, 0, 255)      # rouge
                elif s >= 30:
                     heatmap[yA:yB, xA:xB] = 180
                     color = (0, 255, 0)      # vert
                else:
                     color = None

                if color:
                    cv2.rectangle(car_color, (xA, yA), (xB, yB), color, 1)

                if s >= 50:
                    detected += 1

        # =========================
        # FINAL SCORE
        # =========================
        mean_score = np.mean(zones_scores)
        max_score = np.max(zones_scores)

        final_score = int(np.mean(zones_scores))
        final_score = min(final_score, 100)

        if final_score < 20:
            result = "Peinture d'origine (OK)"
        elif final_score < 45:
            result = "Légères retouches"
        elif final_score < 70:
            result = "Peinture suspecte"
        else:
            result = "Voiture probablement repeinte"

        # =========================
        # HEATMAP FINAL IMAGE
        # =========================
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        final_img = cv2.addWeighted(car_color, 0.7, heatmap_color, 0.3, 0)

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
