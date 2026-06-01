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

TARGET_W = 900
TARGET_H = 500

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
    return jsonify({"status": "OK", "message": "GARAGE PRO V4 API"})

# =========================
# COULEUR HSV MÉDIANE
# =========================
def get_zone_color(hsv_img, mask, xA, yA, xB, yB):
    zone_mask = mask[yA:yB, xA:xB]
    zone_hsv  = hsv_img[yA:yB, xA:xB]
    valid     = zone_hsv[zone_mask > 0]
    if len(valid) < 80:
        return None, 0
    return np.array([
        float(np.median(valid[:, 0])),
        float(np.median(valid[:, 1])),
        float(np.median(valid[:, 2]))
    ]), len(valid)


# =========================
# AFFINER LE CROP
# =========================
def refine_car_bbox(img, x1, y1, x2, y2):
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return x1, y1, x2, y2

    hsv        = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask_dark  = cv2.inRange(hsv, (0, 0, 0),   (180, 255, 40))
    mask_sky   = cv2.inRange(hsv, (0, 0, 215), (180, 15, 255))
    mask_valid = cv2.bitwise_not(cv2.bitwise_or(mask_dark, mask_sky))
    k          = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_valid = cv2.morphologyEx(mask_valid, cv2.MORPH_CLOSE, k, iterations=2)

    col_sum = mask_valid.sum(axis=0).astype(float)
    row_sum = mask_valid.sum(axis=1).astype(float)
    col_sum = np.convolve(col_sum, np.ones(15) / 15, mode='same')
    row_sum = np.convolve(row_sum, np.ones(15) / 15, mode='same')

    col_thresh = col_sum.max() * 0.08
    row_thresh = row_sum.max() * 0.08
    valid_cols = np.where(col_sum > col_thresh)[0]
    valid_rows = np.where(row_sum > row_thresh)[0]

    if len(valid_cols) < 20 or len(valid_rows) < 20:
        return x1, y1, x2, y2

    PAD    = 8
    new_x1 = max(0,            x1 + int(valid_cols[0])  - PAD)
    new_x2 = min(img.shape[1], x1 + int(valid_cols[-1]) + PAD)
    new_y1 = max(0,            y1 + int(valid_rows[0])  - PAD)
    new_y2 = min(img.shape[0], y1 + int(valid_rows[-1]) + PAD)

    if (new_x2 - new_x1) < 100 or (new_y2 - new_y1) < 80:
        return x1, y1, x2, y2

    return new_x1, new_y1, new_x2, new_y2


# =============================================
# DÉTECTER ORIENTATION
#
# PRIORITÉ 1 : feux arrière ROUGES
#              → côté rouge = arrière
# PRIORITÉ 2 : phares avant BLANCS/JAUNES
#              → côté blanc = avant
# PRIORITÉ 3 : pare-brise avant
#              Le pare-brise AVANT est toujours
#              plus grand que le pare-brise arrière
#              On mesure la surface vitrée des
#              deux côtés dans la zone haute
# =============================================
def detect_car_orientation(car_crop):
    crop_h, crop_w = car_crop.shape[:2]
    log = []

    # Bandes gauche/droite (22% de largeur)
    # Zone basse = feux (38% à 88% de hauteur)
    band_w  = int(crop_w * 0.22)
    feux_y1 = int(crop_h * 0.38)
    feux_y2 = int(crop_h * 0.88)

    left_feux  = car_crop[feux_y1:feux_y2, 0:band_w]
    right_feux = car_crop[feux_y1:feux_y2, crop_w - band_w:crop_w]

    left_hsv  = cv2.cvtColor(left_feux,  cv2.COLOR_BGR2HSV)
    right_hsv = cv2.cvtColor(right_feux, cv2.COLOR_BGR2HSV)

    # -----------------------------------------------
    # PRIORITÉ 1 : feux arrière ROUGES
    # Rouge HSV : H=[0-12] ou [168-180], S>60, V>60
    # -----------------------------------------------
    def count_red(hsv):
        m1 = cv2.inRange(hsv, (0,   60, 60), (12,  255, 255))
        m2 = cv2.inRange(hsv, (168, 60, 60), (180, 255, 255))
        return cv2.countNonZero(cv2.bitwise_or(m1, m2))

    red_left  = count_red(left_hsv)
    red_right = count_red(right_hsv)
    total_red = red_left + red_right

    # Seuil minimum : au moins 150 pixels rouges au total
    if total_red > 150:
        if red_left > red_right * 1.35:
            log.append(f"P1 ROUGE: gauche={red_left} droite={red_right} → avant DROITE")
            return "right", log
        elif red_right > red_left * 1.35:
            log.append(f"P1 ROUGE: gauche={red_left} droite={red_right} → avant GAUCHE")
            return "left", log
        else:
            log.append(f"P1 ROUGE: trop equilibre ({red_left}/{red_right}), on passe P2")
    else:
        log.append(f"P1 ROUGE: pas assez de rouge ({total_red}px), on passe P2")

    # -----------------------------------------------
    # PRIORITÉ 2 : phares avant BLANCS/JAUNES
    # Phare avant = zone très lumineuse
    # HSV : V>170 ET (S<90 pour blanc OU H=[15-40] pour jaune)
    # -----------------------------------------------
    def count_headlight(hsv):
        white  = cv2.inRange(hsv, (0,  0,  170), (180, 90,  255))
        yellow = cv2.inRange(hsv, (15, 40, 170), (40,  220, 255))
        return cv2.countNonZero(cv2.bitwise_or(white, yellow))

    light_left  = count_headlight(left_hsv)
    light_right = count_headlight(right_hsv)
    total_light = light_left + light_right

    if total_light > 100:
        if light_left > light_right * 1.35:
            log.append(f"P2 PHARE: gauche={light_left} droite={light_right} → avant GAUCHE")
            return "left", log
        elif light_right > light_left * 1.35:
            log.append(f"P2 PHARE: gauche={light_left} droite={light_right} → avant DROITE")
            return "right", log
        else:
            log.append(f"P2 PHARE: trop equilibre ({light_left}/{light_right}), on passe P3")
    else:
        log.append(f"P2 PHARE: pas assez de lumiere ({total_light}px), on passe P3")

    # -----------------------------------------------
    # PRIORITÉ 3 : taille du pare-brise
    #
    # Le pare-brise AVANT est toujours plus grand
    # que la lunette arrière.
    # On cherche les zones sombres (vitres) dans
    # la bande haute (8% à 55% de hauteur).
    # On divise en deux moitiés gauche/droite
    # et on mesure la surface vitrée de chaque côté.
    # Le côté avec PLUS de surface vitrée = avant
    # -----------------------------------------------
    gray     = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
    vit_y1   = int(crop_h * 0.08)
    vit_y2   = int(crop_h * 0.58)
    vitre    = gray[vit_y1:vit_y2, :]

    # Pixels sombres = vitre (valeur < 90)
    dark = (vitre < 90).astype(np.uint8)

    # Lisser pour ignorer le bruit
    dark_f = cv2.GaussianBlur(dark.astype(np.float32), (15, 15), 0)

    # Surface vitrée côté gauche vs côté droit
    mid          = crop_w // 2
    left_glass   = float(dark_f[:, :mid].sum())
    right_glass  = float(dark_f[:, mid:].sum())

    log.append(f"P3 VITRE: gauche={int(left_glass)} droite={int(right_glass)}")

    if left_glass > right_glass * 1.15:
        log.append("P3 VITRE: plus grand gauche → avant GAUCHE")
        return "left", log
    elif right_glass > left_glass * 1.15:
        log.append("P3 VITRE: plus grand droite → avant DROITE")
        return "right", log

    # -----------------------------------------------
    # FALLBACK : texture Sobel si tout échoue
    # -----------------------------------------------
    left_band  = car_crop[feux_y1:feux_y2, 0:band_w]
    right_band = car_crop[feux_y1:feux_y2, crop_w - band_w:crop_w]

    lg = cv2.cvtColor(left_band,  cv2.COLOR_BGR2GRAY)
    rg = cv2.cvtColor(right_band, cv2.COLOR_BGR2GRAY)
    ls = float(np.mean(np.abs(cv2.Sobel(lg, cv2.CV_64F, 1, 1, ksize=3))))
    rs = float(np.mean(np.abs(cv2.Sobel(rg, cv2.CV_64F, 1, 1, ksize=3))))

    if rs > ls * 1.2:
        log.append(f"FALLBACK Sobel: droite={rs:.1f}>gauche={ls:.1f} → avant GAUCHE")
        return "left", log
    else:
        log.append(f"FALLBACK Sobel: gauche={ls:.1f}>=droite={rs:.1f} → avant DROITE")
        return "right", log


# =========================
# ANALYSE
# =========================
@app.route("/analyse", methods=["POST"])
def analyse():
    try:
        if "image" not in request.files:
            return jsonify({"error": "no image"}), 400

        file     = request.files["image"]
        filename = str(int(time.time())) + "_" + secure_filename(file.filename)
        path     = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        img_orig = cv2.imread(path)
        if img_orig is None:
            return jsonify({"error": "image unreadable"}), 400

        img = cv2.resize(img_orig, (TARGET_W, TARGET_H))

        resized_path = os.path.join(UPLOAD_FOLDER, "resized_" + filename)
        cv2.imwrite(resized_path, img)

        yolo_result = call_yolo(resized_path)

        img_h, img_w = img.shape[:2]

        detections = yolo_result.get("detections", [])
        cars = [d for d in detections if d.get("class") == 2]
        if not cars:
            return jsonify({"error": "Car not detected"}), 400

        raw_x1 = min(d["box"][0] for d in cars)
        raw_y1 = min(d["box"][1] for d in cars)
        raw_x2 = max(d["box"][2] for d in cars)
        raw_y2 = max(d["box"][3] for d in cars)

        x1 = 0      if raw_x1 < 150           else max(0,     raw_x1 - 15)
        x2 = img_w  if (img_w - raw_x2) < 150 else min(img_w, raw_x2 + 15)
        y1 = 0      if raw_y1 < 80            else max(0,     raw_y1 - 10)
        y2 = img_h  if (img_h - raw_y2) < 80  else min(img_h, raw_y2 + 10)

        # Affiner le crop
        x1, y1, x2, y2 = refine_car_bbox(img, x1, y1, x2, y2)

        car_crop = img[y1:y2, x1:x2]
        if car_crop.size == 0:
            return jsonify({"error": "invalid crop"}), 400

        crop_h, crop_w = car_crop.shape[:2]

        # ===============================================
        # ORIENTATION
        # ===============================================
        orientation, orient_log = detect_car_orientation(car_crop)

        if orientation == "left":
            zone_names = ["Aile avant", "Portes", "Aile arriere"]
        else:
            zone_names = ["Aile arriere", "Portes", "Aile avant"]

        # ===============================================
        # MASQUE CARROSSERIE
        # ===============================================
        hsv_full  = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
        mask_dark = cv2.inRange(hsv_full, (0, 0, 0),   (180, 255, 45))
        mask_sky  = cv2.inRange(hsv_full, (0, 0, 210), (180, 18, 255))
        mask_body = cv2.bitwise_not(cv2.bitwise_or(mask_dark, mask_sky))
        kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_CLOSE, kernel)

        # ===============================================
        # MOYENNE GLOBALE
        # ===============================================
        all_valid = hsv_full[mask_body > 0]
        if len(all_valid) < 100:
            return jsonify({"error": "No body pixels found"}), 400

        ref_color = np.array([
            float(np.median(all_valid[:, 0])),
            float(np.median(all_valid[:, 1])),
            float(np.median(all_valid[:, 2]))
        ])

        # ===============================================
        # 3 ZONES
        # ===============================================
        band_y1 = int(crop_h * 0.15)
        band_y2 = int(crop_h * 0.80)
        cut1    = int(crop_w * 0.33)
        cut2    = int(crop_w * 0.67)

        zones = [
            {"name": zone_names[0], "xA": 0,    "xB": cut1,   "yA": band_y1, "yB": band_y2},
            {"name": zone_names[1], "xA": cut1, "xB": cut2,   "yA": band_y1, "yB": band_y2},
            {"name": zone_names[2], "xA": cut2, "xB": crop_w, "yA": band_y1, "yB": band_y2},
        ]

        # ===============================================
        # DESSIN
        # ===============================================
        final_img = img.copy()
        cv2.rectangle(final_img, (x1, y1), (x2, y2), (220, 220, 220), 1)

        for cut in [cut1, cut2]:
            for dy in range(band_y1, band_y2, 12):
                cv2.line(
                    final_img,
                    (x1 + cut, y1 + dy),
                    (x1 + cut, y1 + dy + 6),
                    (255, 255, 255), 1
                )

        results_zones = []
        detected = 0

        for zone in zones:
            xA, xB = zone["xA"], zone["xB"]
            yA, yB = zone["yA"], zone["yB"]

            zone_color, px_count = get_zone_color(
                hsv_full, mask_body, xA, yA, xB, yB
            )

            abs_x1 = x1 + xA
            abs_y1 = y1 + yA
            abs_x2 = x1 + xB
            abs_y2 = y1 + yB

            if zone_color is None:
                color_rect  = (150, 150, 150)
                label_score = "N/A"
                diff        = 0.0
                verdict     = "Non analysable"
            else:
                diff = float(np.linalg.norm(zone_color - ref_color))

                if diff >= 14 and diff < 26:
                    color_rect = (0, 0, 255)
                    verdict    = "Attention peinture refaite!"
                elif diff < 14:
                    color_rect = (0, 165, 255)
                    verdict    = "Legere variation suspecte!"
                    detected  += 1
                else:
                    color_rect = (0, 210, 0)
                    verdict    = "OK"
                    detected  += 1

                label_score = str(int(diff))

            cv2.rectangle(final_img, (abs_x1, abs_y1),
                          (abs_x2, abs_y2), color_rect, 5)

            overlay = final_img.copy()
            cv2.rectangle(overlay, (abs_x1, abs_y1),
                          (abs_x2, abs_y1 + 55), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, final_img, 0.5, 0, final_img)

            cv2.putText(final_img, zone["name"],
                        (abs_x1 + 8, abs_y1 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(final_img, f"Ecart: {label_score}",
                        (abs_x1 + 8, abs_y1 + 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rect, 2)
            cv2.putText(final_img, verdict,
                        (abs_x1 + 8, abs_y2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_rect, 2)

            results_zones.append({
                "zone":    zone["name"],
                "diff":    round(diff, 1),
                "pixels":  px_count,
                "verdict": verdict
            })

        # ===============================================
        # SCORE GLOBAL
        # ===============================================
        diffs = [z["diff"] for z in results_zones if z["diff"] > 0]
        final_score = int(np.mean(diffs)) if diffs else 0
        final_score = min(final_score, 100)

        if final_score < 10:
            result = "Peinture homogene (OK)"
        elif final_score < 28:
            result = "Legeres variations detectees"
        else:
            result = "Difference importante — repeinture probable"

        cv2.putText(
            final_img,
            f"Ref: H={int(ref_color[0])} S={int(ref_color[1])} "
            f"V={int(ref_color[2])}  |  "
            f"Avant: {'GAUCHE' if orientation == 'left' else 'DROITE'}  |  "
            f"Detection: {orient_log[-1][:30]}",
            (10, img_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1
        )

        analysed_name = "analysed_" + filename
        analysed_path = os.path.join(UPLOAD_FOLDER, analysed_name)
        cv2.imwrite(analysed_path, final_img)

        if os.path.exists(resized_path):
            os.remove(resized_path)

        return jsonify({
            "yolo":            yolo_result,
            "score":           final_score,
            "result":          result,
            "zones":           results_zones,
            "zones_detected":  detected,
            "orientation":     orientation,
            "orientation_log": orient_log,
            "reference_hsv": {
                "H": round(ref_color[0], 1),
                "S": round(ref_color[1], 1),
                "V": round(ref_color[2], 1)
            },
            "image_result":    analysed_name,
            "image_url":       request.host_url + "uploads/" + analysed_name
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
