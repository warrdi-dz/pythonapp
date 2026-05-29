from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import traceback
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return jsonify({
        "status": "OK",
        "message": "SCAN AUTO API"
    })


@app.route("/analyse", methods=["POST"])
def analyse():

    try:

        print("FILES:", request.files)
        print("FORM:", request.form)

        if 'image' not in request.files:
            return jsonify({
                "error": "no image",
                "debug_files": str(request.files)
            }), 400

        file = request.files['image']

        filename = secure_filename(file.filename)

        path = os.path.join(UPLOAD_FOLDER, filename)

        file.save(path)

        img = cv2.imread(path)

        if img is None:
            return jsonify({
                "error": "image not readable"
            }), 400

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blur, 50, 150)

        contours_data = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        contours = contours_data[0] if len(contours_data) == 2 else contours_data[1]

        if len(contours) == 0:
            return jsonify({
                "error": "no object detected"
            }), 400

        main_contour = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(main_contour)

        car = gray[y:y+h, x:x+w]

        if car.size == 0:
            return jsonify({
                "error": "empty crop"
            }), 400

        car = cv2.resize(car, (600, 300))

        left_zone = car[:, :300]
        right_zone = car[:, 300:]

        b1 = np.mean(left_zone)
        b2 = np.mean(right_zone)

        diff_brightness = abs(b1 - b2)

        t1 = cv2.Laplacian(left_zone, cv2.CV_64F).var()
        t2 = cv2.Laplacian(right_zone, cv2.CV_64F).var()

        diff_texture = abs(t1 - t2)

        score = 0

        if diff_brightness > 12:
            score += 50

        if diff_texture > 80:
            score += 50

        if score > 60:
            result = "Peinture suspecte détectée"
        else:
            result = "Peinture normale"

        return jsonify({
            "score": int(score),
            "result": result
        })

    except Exception as e:

        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
