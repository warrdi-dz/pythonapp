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
# =========================
# ANALYSE INTELLIGENTE
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

        yolo_result = call_yolo(path)

        img = cv2.imread(path)
        if img is None:
            return jsonify({"error": "image unreadable"}), 400

        img = cv2.resize(img, (900, 500))
        img_h, img_w = img.shape[:2]

        detections = yolo_result.get("detections", [])
        cars = [d for d in detections if d.get("class") == 2]
        if not cars:
            return jsonify({"error": "Car not detected"}), 400

        # === BOUNDING BOX FUSIONNÉE + PADDING ===
        PADDING = 25
        x1 = max(0,     min(d["box"][0] for d in cars) - PADDING)
        y1 = max(0,     min(d["box"][1] for d in cars) - PADDING)
        x2 = min(img_w, max(d["box"][2] for d in cars) + PADDING)
        y2 = min(img_h, max(d["box"][3] for d in cars) + PADDING)

        if (img_w - x2) < 60: x2 = img_w
        if (img_h - y2) < 60: y2 = img_h
        if x1 < 60: x1 = 0
        if y1 < 60: y1 = 0

        car_crop = img[y1:y2, x1:x2]
        if car_crop.size == 0:
            return jsonify({"error": "invalid crop"}), 400

        crop_h, crop_w = car_crop.shape[:2]

        # =============================================
        # MASQUE : EXCLURE VITRES / ROUES / FOND
        # On travaille uniquement sur la carrosserie
        # =============================================
        hsv_full = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)

        # Masque 1 : exclure les zones très sombres (vitres, pneus, joints)
        # V < 40 = trop sombre = pas de la carrosserie
        mask_dark = cv2.inRange(hsv_full, (0, 0, 0), (180, 255, 50))

        # Masque 2 : exclure les zones très désaturées ET sombres (ciel, fond)
        # S < 15 ET V > 200 = blanc pur = ciel/fond
        mask_sky = cv2.inRange(hsv_full, (0, 0, 200), (180, 15, 255))

        # Masque final carrosserie = tout sauf vitres/roues/fond
        mask_body = cv2.bitwise_not(cv2.bitwise_or(mask_dark, mask_sky))

        # Appliquer le masque : zones exclues = noir
        car_body = car_crop.copy()
        car_body[mask_body == 0] = 0

        # =============================================
        # GRILLE 6x8 FINE SUR LA ZONE CARROSSERIE
        # =============================================
        rows, cols = 6, 8
        cell_h = crop_h // rows
        cell_w = crop_w // cols

        # Calculer la couleur HSV médiane de chaque cellule
        # (uniquement pixels valides = carrosserie)
        cell_colors = []   # liste de (i, j, h_med, s_med, v_med, pixel_count)

        for i in range(rows):
            for j in range(cols):
                yA, yB = i * cell_h, (i + 1) * cell_h
                xA, xB = j * cell_w, (j + 1) * cell_w

                zone_mask  = mask_body[yA:yB, xA:xB]
                zone_hsv   = hsv_full[yA:yB, xA:xB]

                # Pixels valides uniquement
                valid_pixels = zone_hsv[zone_mask > 0]

                if len(valid_pixels) < 50:
                    # Moins de 50 pixels valides = zone vitres/roues, on ignore
                    cell_colors.append((i, j, None, None, None, 0))
                    continue

                h_med = float(np.median(valid_pixels[:, 0]))
                s_med = float(np.median(valid_pixels[:, 1]))
                v_med = float(np.median(valid_pixels[:, 2]))

                cell_colors.append((i, j, h_med, s_med, v_med, len(valid_pixels)))

        # =============================================
        # RÉFÉRENCE : médiane de TOUTES les cellules
        # valides (carrosserie propre)
        # =============================================
        valid_cells = [(c[2], c[3], c[4]) for c in cell_colors if c[2] is not None]

        if not valid_cells:
            return jsonify({"error": "No body pixels found"}), 400

        ref_h = float(np.median([c[0] for c in valid_cells]))
        ref_s = float(np.median([c[1] for c in valid_cells]))
        ref_v = float(np.median([c[2] for c in valid_cells]))
        ref_color = np.array([ref_h, ref_s, ref_v])

        # =============================================
        # DESSIN : colorier chaque cellule selon
        # son écart à la référence carrosserie
        # =============================================
        final_img = img.copy()

        # Contour zone analysée (orange)
        cv2.rectangle(final_img, (x1, y1), (x2, y2), (0, 140, 255), 2)

        zones_scores = []
        detected = 0

        for (i, j, h_med, s_med, v_med, count) in cell_colors:

            yA, yB = i * cell_h, (i + 1) * cell_h
            xA, xB = j * cell_w, (j + 1) * cell_w

            abs_x1 = x1 + xA
            abs_y1 = y1 + yA
            abs_x2 = x1 + xB
            abs_y2 = y1 + yB

            if h_med is None:
                # Zone exclue (vitres/roues) — contour gris fin
                cv2.rectangle(final_img, (abs_x1, abs_y1), (abs_x2, abs_y2),
                              (100, 100, 100), 1)
                continue

            cell_color = np.array([h_med, s_med, v_med])
            diff = np.linalg.norm(cell_color - ref_color)
            zones_scores.append(diff)

            # Seuils affinés
            if diff < 15:
                # OK — pas de rectangle
                pass
            elif diff < 30:
                # Légère variation — vert
                cv2.rectangle(final_img, (abs_x1, abs_y1), (abs_x2, abs_y2),
                              (0, 255, 0), 2)
                detected += 1
            elif diff < 55:
                # Zone suspecte — orange
                cv2.rectangle(final_img, (abs_x1, abs_y1), (abs_x2, abs_y2),
                              (0, 165, 255), 3)
                detected += 1
            else:
                # Forte différence — rouge épais
                cv2.rectangle(final_img, (abs_x1, abs_y1), (abs_x2, abs_y2),
                              (0, 0, 255), 3)
                detected += 1

            # Score affiché dans la cellule
            cv2.putText(
                final_img,
                str(int(diff)),
                (abs_x1 + 4, abs_y1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1
            )

        final_score = int(np.mean(zones_scores)) if zones_scores else 0
        final_score = min(final_score, 100)

        if final_score < 15:
            result = "Peinture homogène (OK)"
        elif final_score < 30:
            result = "Légères variations"
        elif final_score < 55:
            result = "Zones suspectes — vérifier"
        else:
            result = "Différence importante — repeinture probable"

        analysed_name = "analysed_" + filename
        analysed_path = os.path.join(UPLOAD_FOLDER, analysed_name)
        cv2.imwrite(analysed_path, final_img)

        return jsonify({
            "yolo":           yolo_result,
            "score":          final_score,
            "result":         result,
            "zones_detected": detected,
            "image_result":   analysed_name,
            "image_url":      request.host_url + "uploads/" + analysed_name
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500 port=5000)
