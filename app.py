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

YOLO_W = 900
YOLO_H = 500

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

@app.route("/")
def home():
    return jsonify({"status": "OK", "message": "GARAGE PRO V5 API (angle aware)"})

# =========================
# COULEUR HSV MEDIANE
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
# REFINE CROP
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
    PAD = 8
    new_x1 = max(0,            x1 + int(valid_cols[0])  - PAD)
    new_x2 = min(img.shape[1], x1 + int(valid_cols[-1]) + PAD)
    new_y1 = max(0,            y1 + int(valid_rows[0])  - PAD)
    new_y2 = min(img.shape[0], y1 + int(valid_rows[-1]) + PAD)
    if (new_x2 - new_x1) < 100 or (new_y2 - new_y1) < 80:
        return x1, y1, x2, y2
    return new_x1, new_y1, new_x2, new_y2


# =========================
# DETECTION FEUX (rouge / blanc) gauche & droite
# Retourne dict avec comptages + ratios
# =========================
def detect_lights(car_crop):
    h, w = car_crop.shape[:2]
    band_w  = int(w * 0.22)
    feux_y1 = int(h * 0.38)
    feux_y2 = int(h * 0.88)

    left_feux  = car_crop[feux_y1:feux_y2, 0:band_w]
    right_feux = car_crop[feux_y1:feux_y2, w - band_w:w]
    left_hsv   = cv2.cvtColor(left_feux,  cv2.COLOR_BGR2HSV)
    right_hsv  = cv2.cvtColor(right_feux, cv2.COLOR_BGR2HSV)

    def count_red(hsv):
        m1 = cv2.inRange(hsv, (0,   60, 60), (12,  255, 255))
        m2 = cv2.inRange(hsv, (168, 60, 60), (180, 255, 255))
        return int(cv2.countNonZero(cv2.bitwise_or(m1, m2)))

    def count_white(hsv):
        white  = cv2.inRange(hsv, (0,  0,  170), (180, 90,  255))
        yellow = cv2.inRange(hsv, (15, 40, 170), (40,  220, 255))
        return int(cv2.countNonZero(cv2.bitwise_or(white, yellow)))

    return {
        "red_left":    count_red(left_hsv),
        "red_right":   count_red(right_hsv),
        "white_left":  count_white(left_hsv),
        "white_right": count_white(right_hsv),
        "band_area":   band_w * (feux_y2 - feux_y1)
    }


# =========================
# ORIENTATION (basé sur feux)
# =========================
def detect_car_orientation(car_crop, lights=None):
    log = []
    if lights is None:
        lights = detect_lights(car_crop)

    rl, rr = lights["red_left"], lights["red_right"]
    wl, wr = lights["white_left"], lights["white_right"]

    # Priorité 1 : rouge
    if (rl + rr) > 150:
        if rl > rr * 1.35:
            log.append(f"P1 ROUGE: G={rl} D={rr} -> avant DROITE (rouge=arriere G)")
            return "right", log
        if rr > rl * 1.35:
            log.append(f"P1 ROUGE: G={rl} D={rr} -> avant GAUCHE (rouge=arriere D)")
            return "left", log
        log.append(f"P1 ROUGE equilibre ({rl}/{rr})")

    # Priorité 2 : blanc
    if (wl + wr) > 100:
        if wl > wr * 1.35:
            log.append(f"P2 PHARE: G={wl} D={wr} -> avant GAUCHE")
            return "left", log
        if wr > wl * 1.35:
            log.append(f"P2 PHARE: G={wl} D={wr} -> avant DROITE")
            return "right", log
        log.append(f"P2 PHARE equilibre ({wl}/{wr})")

    # Fallback vitre
    h, w = car_crop.shape[:2]
    gray   = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
    vitre  = gray[int(h*0.08):int(h*0.58), :]
    dark   = (vitre < 90).astype(np.uint8)
    dark_f = cv2.GaussianBlur(dark.astype(np.float32), (15, 15), 0)
    mid    = w // 2
    lg, rg = float(dark_f[:, :mid].sum()), float(dark_f[:, mid:].sum())
    log.append(f"P3 VITRE: G={int(lg)} D={int(rg)}")
    if lg > rg * 1.10:
        return "left", log
    return "right", log


# =========================
# DECIDE ZONES par ANGLE
#
# angle 0-60   : 3/4 arriere -> 4 zones : pare-choc AR, porte AR, porte AV, aile AR
#                (cote "avant" de l'image = arriere de la voiture)
# angle 60-80  : presque profil arriere -> 3 zones : pare-choc AR, aile AR, porte AR
# angle 80-90+ : pile arriere ou pile avant -> 1 zone : pare-choc (AR ou AV)
#                sauf si les DEUX feux (rouge+blanc) visibles -> 4 zones (2 portes + 2 ailes)
# =========================
def build_zones(crop_w, crop_h, angle, orientation, lights):
    band_y1 = int(crop_h * 0.15)
    band_y2 = int(crop_h * 0.80)

    rear_visible  = (lights["red_left"]   + lights["red_right"])   > 150
    front_visible = (lights["white_left"] + lights["white_right"]) > 100

    # ---- 80 - 90+ : vue frontale ou arriere pure ----
    if angle >= 80:
        if rear_visible and front_visible:
            # cas special : on voit les 2 -> 4 zones (2 portes + 2 ailes)
            q = crop_w // 4
            return [
                {"name": "Aile G",  "xA": 0,     "xB": q,       "yA": band_y1, "yB": band_y2},
                {"name": "Porte G", "xA": q,     "xB": 2*q,     "yA": band_y1, "yB": band_y2},
                {"name": "Porte D", "xA": 2*q,   "xB": 3*q,     "yA": band_y1, "yB": band_y2},
                {"name": "Aile D",  "xA": 3*q,   "xB": crop_w,  "yA": band_y1, "yB": band_y2},
            ], "angle>=80, 2 feux visibles -> 4 zones laterales"

        # sinon : pare-choc seulement
        name = "Pare-choc AR" if rear_visible else "Pare-choc AV"
        return [
            {"name": name, "xA": 0, "xB": crop_w, "yA": band_y1, "yB": band_y2},
        ], f"angle>=80 -> 1 zone ({name})"

    # ---- 60 - 80 : 3/4 fortement arriere ----
    if angle >= 60:
        # 3 zones cote arriere : pare-choc AR, aile AR, porte AR
        # on place pare-choc du cote oppose a l'avant
        if orientation == "left":
            # avant a gauche => arriere a droite
            return [
                {"name": "Porte AR",     "xA": 0,             "xB": int(crop_w*0.40), "yA": band_y1, "yB": band_y2},
                {"name": "Aile AR",      "xA": int(crop_w*0.40), "xB": int(crop_w*0.75), "yA": band_y1, "yB": band_y2},
                {"name": "Pare-choc AR", "xA": int(crop_w*0.75), "xB": crop_w,         "yA": band_y1, "yB": band_y2},
            ], "angle 60-80, avant gauche -> 3 zones AR a droite"
        else:
            return [
                {"name": "Pare-choc AR", "xA": 0,             "xB": int(crop_w*0.25), "yA": band_y1, "yB": band_y2},
                {"name": "Aile AR",      "xA": int(crop_w*0.25), "xB": int(crop_w*0.60), "yA": band_y1, "yB": band_y2},
                {"name": "Porte AR",     "xA": int(crop_w*0.60), "xB": crop_w,         "yA": band_y1, "yB": band_y2},
            ], "angle 60-80, avant droit -> 3 zones AR a gauche"

    # ---- 0 - 60 : 3/4 leger -> 4 zones (cote large complet) ----
    # Le cote ou se trouve l'arriere (feu rouge gros) est le cote "large"
    # -> on scanne : pare-choc AR + aile AR + porte AR + porte AV
    # L'aile AVANT du cote oppose est petite (non scannable) donc ignoree
    if orientation == "left":
        # avant a gauche, arriere a droite (gros cote = droite)
        return [
            {"name": "Porte AV",     "xA": 0,                 "xB": int(crop_w*0.28), "yA": band_y1, "yB": band_y2},
            {"name": "Porte AR",     "xA": int(crop_w*0.28),  "xB": int(crop_w*0.55), "yA": band_y1, "yB": band_y2},
            {"name": "Aile AR",      "xA": int(crop_w*0.55),  "xB": int(crop_w*0.80), "yA": band_y1, "yB": band_y2},
            {"name": "Pare-choc AR", "xA": int(crop_w*0.80),  "xB": crop_w,           "yA": band_y1, "yB": band_y2},
        ], "angle 0-60, avant gauche -> 4 zones cote droit"
    else:
        # avant a droite, arriere a gauche (gros cote = gauche)
        return [
            {"name": "Pare-choc AR", "xA": 0,                 "xB": int(crop_w*0.20), "yA": band_y1, "yB": band_y2},
            {"name": "Aile AR",      "xA": int(crop_w*0.20),  "xB": int(crop_w*0.45), "yA": band_y1, "yB": band_y2},
            {"name": "Porte AR",     "xA": int(crop_w*0.45),  "xB": int(crop_w*0.72), "yA": band_y1, "yB": band_y2},
            {"name": "Porte AV",     "xA": int(crop_w*0.72),  "xB": crop_w,           "yA": band_y1, "yB": band_y2},
        ], "angle 0-60, avant droit -> 4 zones cote gauche"


# =========================
# ANALYSE
# =========================
@app.route("/analyse", methods=["POST"])
def analyse():
    try:
        if "image" not in request.files:
            return jsonify({"error": "no image"}), 400

        # angle (degres) : 0 = vue laterale pure, 90 = vue frontale/arriere pure
        try:
            angle = float(request.form.get("angle", "30"))
        except Exception:
            angle = 30.0
        angle = max(0.0, min(angle, 90.0))

        file     = request.files["image"]
        filename = str(int(time.time())) + "_" + secure_filename(file.filename)
        path     = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        img_orig = cv2.imread(path)
        if img_orig is None:
            return jsonify({"error": "image unreadable"}), 400
        orig_h, orig_w = img_orig.shape[:2]

        img_yolo     = cv2.resize(img_orig, (YOLO_W, YOLO_H))
        resized_path = os.path.join(UPLOAD_FOLDER, "resized_" + filename)
        cv2.imwrite(resized_path, img_yolo)

        yolo_result = call_yolo(resized_path)
        detections  = yolo_result.get("detections", [])
        cars = [d for d in detections if d.get("class") == 2]
        if not cars:
            return jsonify({"error": "Car not detected"}), 400

        scale_x = orig_w / YOLO_W
        scale_y = orig_h / YOLO_H

        raw_x1 = int(min(d["box"][0] for d in cars) * scale_x)
        raw_y1 = int(min(d["box"][1] for d in cars) * scale_y)
        raw_x2 = int(max(d["box"][2] for d in cars) * scale_x)
        raw_y2 = int(max(d["box"][3] for d in cars) * scale_y)

        pad_x = int(15 * scale_x); pad_y = int(10 * scale_y)
        thr_x = int(150 * scale_x); thr_y = int(80 * scale_y)
        x1 = 0      if raw_x1 < thr_x            else max(0,     raw_x1 - pad_x)
        x2 = orig_w if (orig_w - raw_x2) < thr_x else min(orig_w, raw_x2 + pad_x)
        y1 = 0      if raw_y1 < thr_y            else max(0,     raw_y1 - pad_y)
        y2 = orig_h if (orig_h - raw_y2) < thr_y else min(orig_h, raw_y2 + pad_y)

        x1, y1, x2, y2 = refine_car_bbox(img_orig, x1, y1, x2, y2)
        car_crop = img_orig[y1:y2, x1:x2]
        if car_crop.size == 0:
            return jsonify({"error": "invalid crop"}), 400
        crop_h, crop_w = car_crop.shape[:2]

        # ---- feux + orientation ----
        lights = detect_lights(car_crop)
        orientation, orient_log = detect_car_orientation(car_crop, lights)

        # ---- zones selon angle ----
        zones, zone_decision = build_zones(crop_w, crop_h, angle, orientation, lights)
        orient_log.append(f"ANGLE={angle} -> {zone_decision}")

        # ---- masque carrosserie + ref globale ----
        hsv_full  = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
        mask_dark = cv2.inRange(hsv_full, (0, 0, 0),   (180, 255, 45))
        mask_sky  = cv2.inRange(hsv_full, (0, 0, 210), (180, 18, 255))
        mask_body = cv2.bitwise_not(cv2.bitwise_or(mask_dark, mask_sky))
        kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_CLOSE, kernel)

        all_valid = hsv_full[mask_body > 0]
        if len(all_valid) < 100:
            return jsonify({"error": "No body pixels found"}), 400
        ref_color = np.array([
            float(np.median(all_valid[:, 0])),
            float(np.median(all_valid[:, 1])),
            float(np.median(all_valid[:, 2]))
        ])

        # ---- dessin ----
        final_img = img_orig.copy()
        thick_box   = max(3, int(5  * min(scale_x, scale_y)))
        thick_line  = max(1, int(1  * min(scale_x, scale_y)))
        font_big    = max(0.6, 0.6  * min(scale_x, scale_y))
        font_med    = max(0.5, 0.5  * min(scale_x, scale_y))
        font_thick  = max(2, int(2  * min(scale_x, scale_y)))
        overlay_h   = max(55, int(55 * scale_y))

        cv2.rectangle(final_img, (x1, y1), (x2, y2), (220, 220, 220), thick_line)

        results_zones = []
        detected = 0

        for zone in zones:
            xA, xB, yA, yB = zone["xA"], zone["xB"], zone["yA"], zone["yB"]
            zone_color, px_count = get_zone_color(hsv_full, mask_body, xA, yA, xB, yB)

            abs_x1 = x1 + xA; abs_y1 = y1 + yA
            abs_x2 = x1 + xB; abs_y2 = y1 + yB

            if zone_color is None:
                color_rect  = (150, 150, 150)
                label_score = "N/A"; diff = 0.0
                verdict     = "Non analysable"
            else:
                diff = float(np.linalg.norm(zone_color - ref_color))
                if 14 <= diff < 26:
                    color_rect = (0, 0, 255);   verdict = "Attention peinture refaite!"
                elif diff < 14:
                    color_rect = (0, 165, 255); verdict = "Legere variation suspecte!"; detected += 1
                else:
                    color_rect = (0, 210, 0);   verdict = "OK"; detected += 1
                label_score = str(int(diff))

            cv2.rectangle(final_img, (abs_x1, abs_y1), (abs_x2, abs_y2), color_rect, thick_box)
            overlay = final_img.copy()
            cv2.rectangle(overlay, (abs_x1, abs_y1), (abs_x2, abs_y1 + overlay_h), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.5, final_img, 0.5, 0, final_img)

            cv2.putText(final_img, zone["name"],
                        (abs_x1 + 8, abs_y1 + int(overlay_h * 0.40)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_big, (255,255,255), font_thick)
            cv2.putText(final_img, f"Ecart: {label_score}",
                        (abs_x1 + 8, abs_y1 + int(overlay_h * 0.80)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_med, color_rect, font_thick)
            cv2.putText(final_img, verdict,
                        (abs_x1 + 8, abs_y2 - int(10 * scale_y)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_med, color_rect, font_thick)

            results_zones.append({
                "zone": zone["name"], "diff": round(diff,1),
                "pixels": px_count, "verdict": verdict
            })

        diffs = [z["diff"] for z in results_zones if z["diff"] > 0]
        final_score = min(int(np.mean(diffs)) if diffs else 0, 100)
        if   final_score < 10: result = "Peinture homogene (OK)"
        elif final_score < 28: result = "Legeres variations detectees"
        else:                  result = "Difference importante - repeinture probable"

        analysed_name = "analysed_" + filename
        analysed_path = os.path.join(UPLOAD_FOLDER, analysed_name)
        cv2.imwrite(analysed_path, final_img)
        if os.path.exists(resized_path):
            os.remove(resized_path)

        return jsonify({
            "yolo":            yolo_result,
            "angle":           angle,
            "score":           final_score,
            "result":          result,
            "zones":           results_zones,
            "zones_detected":  detected,
            "orientation":     orientation,
            "orientation_log": orient_log,
            "lights":          lights,
            "image_size":      {"width": orig_w, "height": orig_h},
            "reference_hsv": {
                "H": round(ref_color[0],1),
                "S": round(ref_color[1],1),
                "V": round(ref_color[2],1)
            },
            "image_result":    analysed_name,
            "image_url":       request.host_url + "uploads/" + analysed_name
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
