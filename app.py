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

        if 'image' not in request.files:

            return jsonify({
                "error": "no image"
            }), 400

        file = request.files['image']

        filename = secure_filename(file.filename)

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

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        blur = cv2.GaussianBlur(
            gray,
            (5, 5),
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

        suspect_count = 0

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area > 500:

                x, y, w, h = cv2.boundingRect(cnt)

                roi = gray[y:y+h, x:x+w]

                brightness = np.mean(roi)

                texture = cv2.Laplacian(
                    roi,
                    cv2.CV_64F
                ).var()

                if texture < 40 or brightness > 170:

                    suspect_count += 1

                    cv2.rectangle(
                        original,
                        (x, y),
                        (x + w, y + h),
                        (0, 0, 255),
                        3
                    )

                    cv2.putText(
                        original,
                        "Paint Suspect",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

        score = min(
            suspect_count * 20,
            100
        )

        if score > 50:

            result = "Peinture probablement refaite"

        else:

            result = "Aucune anomalie importante"

        analysed_name = (
            "analysed_" + filename
        )

        analysed_path = os.path.join(
            UPLOAD_FOLDER,
            analysed_name
        )

        cv2.imwrite(
            analysed_path,
            original
        )

        return jsonify({

            "score": score,

            "result": result,

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
