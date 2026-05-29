from flask import Flask, request, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/")
def home():
    return jsonify({
        "status": "OK",
        "message": "SCAN AUTO API"
    })

@app.route("/analyse", methods=["POST"])
def analyse():

    if 'image' not in request.files:
        return jsonify({"error": "no image"})

    file = request.files['image']

    path = os.path.join(UPLOAD_FOLDER, file.filename)

    file.save(path)

    # ===== ANALYSE =====

    img = cv2.imread(path)

    if img is None:
        return jsonify({"error": "image not found"})

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return jsonify({"error": "no object detected"})

    main_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(main_contour)

    car = gray[y:y+h, x:x+w]

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

if __name__ == "__main__":
    app.run()
