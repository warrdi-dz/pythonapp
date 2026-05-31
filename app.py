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
# COULEUR HSV MÉDIANE
# D'UNE ZONE (pixels valides)
# =========================
def get_zone_color(hsv_img, mask, xA, yA, xB, yB):
    zone_mask = mask[yA:yB, xA:xB]
    zone_hsv  = hsv_img[yA:yB, xA:xB]
    valid     = zone_hsv[zone_mask > 0]
    if len(valid) < 80:
        return None
    return np.array([
        float(np.median(valid[:, 0])),
        float(np.median(valid[:, 1])),
        float(np.median(valid[:, 2]))
    ])


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

        # ===========================
        # BOUNDING BOX COMPLÈTE
        # de roue à roue (toute la voiture)
        # ===========================
        PADDING = 10
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

        # ===========================
        # MASQUE CARROSSERIE
        # Exclure vitres / roues / fond
        # ===========================
        hsv_full = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)

        # Trop sombre = vitres, pneus, joints
        mask_dark = cv2.inRange(hsv_full, (0, 0, 0), (180, 255, 45))

        # Blanc pur = ciel / fond
        mask_sky = cv2.inRange(hsv_full, (0, 0, 210), (180, 18, 255))

        # Masque final = carrosserie uniquement
        mask_body = cv2.bitwise_not(cv2.bitwise_or(mask_dark, mask_sky))

        # ===========================
        # ÉTAPE 1 : MOYENNE GLOBALE
        # sur toute la carrosserie
        # ===========================
        all_valid = hsv_full[mask_body > 0]
        if len(all_valid) < 100:
            return jsonify({"error": "No body pixels found"}), 400

        ref_color = np.array([
            float(np.median(all_valid[:, 0])),
            float(np.median(all_valid[:, 1])),
            float(np.median(all_valid[:, 2]))
        ])

        # ===========================
        # ÉTAPE 2 : DÉCOUPAGE EN 3 ZONES
        #
        #  |-- 20% --|-- 40% --|-- 40% --|
        #  aile avant  portes   aile+porte arrière
        #
        # On adapte selon largeur du crop
        # ===========================

        # Limites verticales : on ignore le bas (roues) et le haut (toit/ciel)
        # On prend la bande centrale = carrosserie principale
        zone_y1 = int(crop_h * 0.10)   # 10% depuis le haut
        zone_y2 = int(crop_h * 0.85)   # jusqu'à 85% (avant les roues)

        # Limites horizontales des 3 zones
        cut1 = int(crop_w * 0.22)   # fin aile avant
        cut2 = int(crop_w * 0.62)   # fin portes / début aile arrière

        zones = [
            {
                "name":  "Aile avant",
                "xA": 0,    "xB": cut1,
                "yA": zone_y1, "yB": zone_y2
            },
            {
                "name":  "Portes",
                "xA": cut1, "xB": cut2,
                "yA": zone_y1, "yB": zone_y2
            },
            {
                "name":  "Aile / Porte arrière",
                "xA": cut2, "xB": crop_w,
                "yA": zone_y1, "yB": zone_y2
            },
        ]

        # ===========================
        # ÉTAPE 3 : COMPARER CHAQUE
        # ZONE À LA MOYENNE GLOBALE
        # ===========================
        final_img = img.copy()

        # Dessiner contour total de la voiture (blanc fin)
        cv2.rectangle(final_img, (x1, y1), (x2, y2), (220, 220, 220), 1)

        results_zones = []
        detected = 0

        for zone in zones:
            xA, xB = zone["xA"], zone["xB"]
            yA, yB = zone["yA"], zone["yB"]

            zone_color = get_zone_color(hsv_full, mask_body, xA, yA, xB, yB)

            # Coordonnées absolues sur l'image finale
            abs_x1 = x1 + xA
            abs_y1 = y1 + yA
            abs_x2 = x1 + xB
            abs_y2 = y1 + yB

            if zone_color is None:
                # Pas assez de pixels valides
                color_rect  = (150, 150, 150)
                label_score = "N/A"
                diff        = 0
                verdict     = "Non analysable"
            else:
                diff = float(np.linalg.norm(zone_color - ref_color))

                # Seuils de différence par rapport à la moyenne globale
                if diff < 12:
                    color_rect = (0, 200, 0)       # vert  = OK
                    verdict    = "OK"
                elif diff < 30:
                    color_rect = (0, 165, 255)     # orange = légère variation
                    verdict    = "Légère variation"
                    detected  += 1
                else:
                    color_rect = (0, 0, 255)       # rouge = suspect
                    verdict    = "Suspect — vérifier"
                    detected  += 1

                label_score = str(int(diff))

            # Grand rectangle de zone
            cv2.rectangle(
                final_img,
                (abs_x1, abs_y1),
                (abs_x2, abs_y2),
                color_rect,
                4               # trait épais = bien visible
            )

            # Fond semi-transparent pour le texte
            text_bg_y1 = abs_y1
            text_bg_y2 = abs_y1 + 48
            overlay = final_img.copy()
            cv2.rectangle(overlay, (abs_x1, text_bg_y1),
                          (abs_x2, text_bg_y2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.45, final_img, 0.55, 0, final_img)

            # Nom de la zone
            cv2.putText(
                final_img,
                zone["name"],
                (abs_x1 + 8, abs_y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1
            )

            # Score + verdict
            cv2.putText(
                final_img,
                f"Ecart: {label_score}  {verdict}",
                (abs_x1 + 8, abs_y1 + 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color_rect,
                1
            )

            results_zones.append({
                "zone":    zone["name"],
                "diff":    round(diff, 1),
                "verdict": verdict
            })

        # ===========================
        # SCORE GLOBAL
        # ===========================
        diffs = [z["diff"] for z in results_zones if z["diff"] > 0]
        final_score = int(np.mean(diffs)) if diffs else 0
        final_score = min(final_score, 100)

        if final_score < 12:
            result = "Peinture homogène (OK)"
        elif final_score < 30:
            result = "Légères variations détectées"
        else:
            result = "Différence importante — repeinture probable"

        # Afficher référence globale en bas de l'image
        cv2.putText(
            final_img,
            f"Teinte de reference (H={int(ref_color[0])} S={int(ref_color[1])} V={int(ref_color[2])})",
            (10, img_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1
        )

        analysed_name = "analysed_" + filename
        analysed_path = os.path.join(UPLOAD_FOLDER, analysed_name)
        cv2.imwrite(analysed_path, final_img)

        return jsonify({
            "yolo":           yolo_result,
            "score":          final_score,
            "result":         result,
            "zones":          results_zones,
            "zones_detected": detected,
            "reference_hsv":  {
                "H": round(ref_color[0], 1),
                "S": round(ref_color[1], 1),
                "V": round(ref_color[2], 1)
            },
            "image_result":   analysed_name,
            "image_url":      request.host_url + "uploads/" + analysed_name
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
