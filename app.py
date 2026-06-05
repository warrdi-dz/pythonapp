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
# YOLO API CALL — fix 415
# =========================
import base64


import requests

def call_yolo(image_path):
    url = "https://warrdi.com/pytho/detect"

    try:
        with open(image_path, "rb") as f:
            files = {"image": (os.path.basename(image_path), f, "image/jpeg")}
            r = requests.post(url, files=files, timeout=20)

        print("STATUS:", r.status_code)
        print("TEXT:", r.text[:500])
        print("URL:", url)
      
        print("RESPONSE:", r.text[:500])
        print("REQUEST HEADERS:", r.request.headers)
        print("FINAL URL:", r.url)
        if r.status_code == 200:
            return r.json()

        return {
            "error": "YOLO failed",
            "status": r.status_code,
            "response": r.text
        }

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

@app.route("/detect", methods=["POST"])
def detect():
    print("🔥 YOLO HIT OK")
    return jsonify({"ok": True})
# =============================================
# MASQUE CARROSSERIE ANTI-OMBRES
# =============================================
def build_body_mask(car_crop, hsv):
    mask_dark = cv2.inRange(hsv, (0, 0,   0), (180, 255,  45))
    mask_refl = cv2.inRange(hsv, (0, 0, 218), (180, 255, 255))
    mask_sky  = cv2.inRange(hsv, (0, 0, 210), (180,  20, 255))
    mask_chro = cv2.inRange(hsv, (0, 0,   0), (180,  28, 255))

    exclude   = cv2.bitwise_or(mask_dark, mask_refl)
    exclude   = cv2.bitwise_or(exclude,   mask_sky)
    exclude   = cv2.bitwise_or(exclude,   mask_chro)
    mask_body = cv2.bitwise_not(exclude)

    k         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_CLOSE, k, iterations=2)
    mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_OPEN,  k, iterations=1)

    h_c, w_c = car_crop.shape[:2]

    # Ombres : seuil élargi
    mask_semi           = cv2.inRange(hsv, (0, 0, 35), (180, 255, 130))
    mask_shadow_on_body = cv2.bitwise_and(mask_semi, mask_body)
    ks                  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_shadow_on_body = cv2.morphologyEx(
        mask_shadow_on_body, cv2.MORPH_CLOSE, ks, iterations=3
    )

    contours, _ = cv2.findContours(
        mask_shadow_on_body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    mask_rm         = np.zeros_like(mask_body)
    total_body_area = max(cv2.countNonZero(mask_body), 1)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        x, y, w, h    = cv2.boundingRect(cnt)
        ratio_hw      = h / max(w, 1)
        ratio_wh      = w / max(h, 1)
        area_ratio    = area / total_body_area
        is_pole       = (ratio_hw > 3.0) and (w < w_c * 0.12)
        is_long       = (ratio_wh > 4.0) and (area_ratio > 0.04)
        touch_bot     = (y + h) > (h_c * 0.88)
        if is_pole or is_long or touch_bot:
            cv2.drawContours(mask_rm, [cnt], -1, 255, -1)

    mask_body = cv2.bitwise_and(mask_body, cv2.bitwise_not(mask_rm))
    mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_CLOSE, k, iterations=1)
    return mask_body


# =============================================
# COULEUR LAB MÉDIANE ANTI-REFLETS
# =============================================
def get_zone_color(lab_img, mask, xA, yA, xB, yB):
    zm    = mask[yA:yB, xA:xB]
    zl    = lab_img[yA:yB, xA:xB]
    valid = zl[zm > 0]
    if len(valid) < 80:
        return None, 0
    L    = valid[:, 0]
    p10  = np.percentile(L, 10)
    p90  = np.percentile(L, 90)
    keep = (L >= p10) & (L <= p90)
    valid = valid[keep]
    if len(valid) < 50:
        return None, 0
    return np.array([
        float(np.median(valid[:, 0])),
        float(np.median(valid[:, 1])),
        float(np.median(valid[:, 2]))
    ]), len(valid)


# =============================================
# DÉTECTER LA VUE
# =============================================
def detect_view(car_crop):
    crop_h, crop_w = car_crop.shape[:2]
    hsv  = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
    log  = []

    ratio_wh = crop_w / max(crop_h, 1)

    mr1      = cv2.inRange(hsv, (0,   60, 60), (12,  255, 255))
    mr2      = cv2.inRange(hsv, (168, 60, 60), (180, 255, 255))
    mask_red = cv2.bitwise_or(mr1, mr2)
    red_tot  = cv2.countNonZero(mask_red)
    red_L    = cv2.countNonZero(mask_red[:, :crop_w//2])
    red_R    = cv2.countNonZero(mask_red[:, crop_w//2:])

    mw1       = cv2.inRange(hsv, (0,  0,  195), (180, 50, 255))
    mw2       = cv2.inRange(hsv, (15, 40, 195), (40, 180, 255))
    ph_zone   = cv2.bitwise_or(mw1, mw2)[int(crop_h*0.45):, :]
    white_tot = cv2.countNonZero(ph_zone)
    white_L   = cv2.countNonZero(ph_zone[:, :crop_w//2])
    white_R   = cv2.countNonZero(ph_zone[:, crop_w//2:])

    top       = gray[int(crop_h*0.05):int(crop_h*0.55), :]
    dk        = (top < 75).astype(np.uint8)
    dk_f      = cv2.GaussianBlur(dk.astype(np.float32), (15, 15), 0)
    glass_L   = float(dk_f[:, :crop_w//2].sum())
    glass_R   = float(dk_f[:, crop_w//2:].sum())
    glass_tot = glass_L + glass_R
    has_doors = (glass_tot > 8000) and (ratio_wh > 1.3)

    log.append(f"ratio={ratio_wh:.2f} rouge={red_tot} "
               f"blanc={white_tot} vitres={int(glass_tot)} portes={has_doors}")

    if has_doors and ratio_wh > 1.4:
        log.append("→ SIDE FULL")
        if red_tot > 200:
            if red_L > red_R * 1.4:
                return "side_full", "right", log
            elif red_R > red_L * 1.4:
                return "side_full", "left", log
        if white_tot > 150:
            if white_L > white_R * 1.4:
                return "side_full", "left", log
            elif white_R > white_L * 1.4:
                return "side_full", "right", log
        return ("side_full", "left" if glass_L > glass_R * 1.2 else "right", log)

    if white_tot > 120 and red_tot < 150 and not has_doors:
        log.append("→ FRONT ONLY")
        return "front_only", ("left" if white_L > white_R else "right"), log

    if red_tot > 150 and white_tot < 100 and not has_doors:
        log.append("→ REAR ONLY")
        return "rear_only", ("left" if red_L > red_R else "right"), log

    if red_tot > 100 and ratio_wh < 1.5:
        log.append("→ REAR 3Q")
        return "rear_3q", ("left" if red_L > red_R else "right"), log

    log.append("→ FALLBACK side_full")
    return "side_full", "left", log


# =============================================
# GRILLE FINE 4x3 = 12 ZONES
# =============================================
def define_zones_grid(view_type, orientation, crop_h, crop_w, mask_body):
    y1_band = int(crop_h * 0.10)
    y2_band = int(crop_h * 0.88)
    band_h  = y2_band - y1_band

    cols = 4
    rows = 3
    cw   = crop_w  // cols
    ch   = band_h  // rows

    if view_type == "side_full":
        col_names = (["Aile av.",  "Porte av.", "Porte ar.", "Aile ar."]
                     if orientation == "left" else
                     ["Aile ar.", "Porte ar.", "Porte av.", "Aile av."])
    elif view_type == "front_only":
        col_names = (["Aile av.G", "Capot G",   "Capot D",   "P-ch av."]
                     if orientation == "left" else
                     ["P-ch av.", "Capot G",    "Capot D",   "Aile av.D"])
    elif view_type in ("rear_only", "rear_3q"):
        col_names = (["Aile ar.G", "Coffre G",  "Coffre D",  "Aile ar.D"]
                     if orientation == "left" else
                     ["Aile ar.D", "Coffre D",  "Coffre G",  "Aile ar.G"])
    else:
        col_names = ["Zone 1", "Zone 2", "Zone 3", "Zone 4"]

    row_names = ["Haut", "Milieu", "Bas"]
    zones     = []

    for r in range(rows):
        for c in range(cols):
            xA = c * cw
            xB = (c + 1) * cw if c < cols - 1 else crop_w
            yA = y1_band + r * ch
            yB = y1_band + (r + 1) * ch if r < rows - 1 else y2_band

            zm  = mask_body[yA:yB, xA:xB]
            npx = cv2.countNonZero(zm)
            if npx < 200:
                continue

            zones.append({
                "name": f"{col_names[c]} {row_names[r]}",
                "xA": xA, "xB": xB,
                "yA": yA, "yB": yB,
                "col": c,  "row": r
            })

    if len(zones) < 3:
        y1 = int(crop_h * 0.10)
        y2 = int(crop_h * 0.88)
        c1 = int(crop_w * 0.33)
        c2 = int(crop_w * 0.67)
        if view_type == "side_full" and orientation == "left":
            names = ["Aile avant", "Portes", "Aile arriere"]
        elif view_type == "side_full":
            names = ["Aile arriere", "Portes", "Aile avant"]
        else:
            names = ["Zone gauche", "Zone centre", "Zone droite"]
        return [
            {"name": names[0], "xA": 0,  "xB": c1,    "yA": y1, "yB": y2, "col": 0, "row": 0},
            {"name": names[1], "xA": c1, "xB": c2,    "yA": y1, "yB": y2, "col": 1, "row": 0},
            {"name": names[2], "xA": c2, "xB": crop_w,"yA": y1, "yB": y2, "col": 2, "row": 0},
        ]
    return zones


# =============================================
# CONSOLIDER PAR PIÈCE (colonne)
# =============================================
def consolidate_by_piece(zone_results):
    by_col = {}
    for z in zone_results:
        c = z.get("col", 0)
        if c not in by_col:
            by_col[c] = []
        by_col[c].append(z["score"])
    return {str(c): round(float(np.mean(s)), 2) for c, s in by_col.items()}


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

        orig_h, orig_w = img_orig.shape[:2]

        # Redimensionner pour YOLO et sauvegarder en JPEG
        img_yolo     = cv2.resize(img_orig, (YOLO_W, YOLO_H))
        resized_path = os.path.join(UPLOAD_FOLDER, "resized_" + filename + ".jpg")
        cv2.imwrite(resized_path, img_yolo, [cv2.IMWRITE_JPEG_QUALITY, 92])

        yolo_result = call_yolo(resized_path)

        detections = yolo_result.get("detections", [])
        cars = [d for d in detections if d.get("class") == 2]
        if not cars:
            return jsonify({"error": "Car not detected"}), 400

        scale_x = orig_w / YOLO_W
        scale_y = orig_h / YOLO_H

        raw_x1 = int(min(d["box"][0] for d in cars) * scale_x)
        raw_y1 = int(min(d["box"][1] for d in cars) * scale_y)
        raw_x2 = int(max(d["box"][2] for d in cars) * scale_x)
        raw_y2 = int(max(d["box"][3] for d in cars) * scale_y)

        pad_x = int(15 * scale_x)
        pad_y = int(10 * scale_y)
        thr_x = int(150 * scale_x)
        thr_y = int(80  * scale_y)

        x1 = 0      if raw_x1 < thr_x            else max(0,      raw_x1 - pad_x)
        x2 = orig_w if (orig_w - raw_x2) < thr_x else min(orig_w, raw_x2 + pad_x)
        y1 = 0      if raw_y1 < thr_y            else max(0,      raw_y1 - pad_y)
        y2 = orig_h if (orig_h - raw_y2) < thr_y else min(orig_h, raw_y2 + pad_y)

        x1, y1, x2, y2 = refine_car_bbox(img_orig, x1, y1, x2, y2)

        car_crop = img_orig[y1:y2, x1:x2]
        if car_crop.size == 0:
            return jsonify({"error": "invalid crop"}), 400

        crop_h, crop_w = car_crop.shape[:2]

        # ===============================================
        # VUE + MASQUE + ESPACES COLORIMÉTRIQUES
        # ===============================================
        view_type, orientation, view_log = detect_view(car_crop)

        hsv_full  = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
        lab_full  = cv2.cvtColor(car_crop, cv2.COLOR_BGR2LAB)
        gray_full = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
        mask_body = build_body_mask(car_crop, hsv_full)

        # Grille 4x3
        zones = define_zones_grid(
            view_type, orientation, crop_h, crop_w, mask_body
        )

        # ===============================================
        # RÉFÉRENCE GLOBALE LAB (percentiles 10-90)
        # ===============================================
        vl_all = lab_full[mask_body > 0]
        if len(vl_all) < 100:
            return jsonify({"error": "No body pixels found"}), 400

        L_all  = vl_all[:, 0]
        p10    = np.percentile(L_all, 10)
        p90    = np.percentile(L_all, 90)
        keep   = (L_all >= p10) & (L_all <= p90)
        vl_ref = vl_all[keep]

        ref_color = np.array([
            float(np.median(vl_ref[:, 0])),
            float(np.median(vl_ref[:, 1])),
            float(np.median(vl_ref[:, 2]))
        ])

        nat_std_a = max(float(np.std(vl_ref[:, 1])), 1.0)
        nat_std_b = max(float(np.std(vl_ref[:, 2])), 1.0)

        # Texture ref sur masque carrosserie seulement
        lap_full    = cv2.Laplacian(gray_full, cv2.CV_64F)
        ref_texture = float(np.var(lap_full[mask_body > 0]))
        ref_texture = max(ref_texture, 1.0)

        # ===============================================
        # DESSIN
        # ===============================================
        final_img      = img_orig.copy()
        thick_box      = max(2, int(3 * min(scale_x, scale_y)))
        thick_line     = max(1, int(1 * min(scale_x, scale_y)))
        font_scale_big = max(0.40, 0.42 * min(scale_x, scale_y))
        font_scale_med = max(0.32, 0.34 * min(scale_x, scale_y))
        font_thick     = max(1,    int(1  * min(scale_x, scale_y)))
        overlay_h      = max(40,   int(42 * scale_y))

        cv2.rectangle(final_img, (x1, y1), (x2, y2), (220, 220, 220), thick_line)

        results_zones = []
        detected      = 0

        for zone in zones:
            xA, xB = zone["xA"], zone["xB"]
            yA, yB = zone["yA"], zone["yB"]

            zone_color, px_count = get_zone_color(
                lab_full, mask_body, xA, yA, xB, yB
            )

            # Texture sur masque seulement
            zm_zone  = mask_body[yA:yB, xA:xB]
            gz_zone  = gray_full[yA:yB, xA:xB]
            lap_zone = cv2.Laplacian(gz_zone, cv2.CV_64F)
            lap_v    = lap_zone[zm_zone > 0]
            tex_zone = float(np.var(lap_v)) if len(lap_v) > 100 else ref_texture
            tex_diff = min(abs(tex_zone - ref_texture) / ref_texture, 2.0)

            abs_x1 = x1 + xA
            abs_y1 = y1 + yA
            abs_x2 = x1 + xB
            abs_y2 = y1 + yB

            if zone_color is None:
                color_rect  = (150, 150, 150)
                label_score = "N/A"
                diff        = 0.0
                verdict     = "N/A"
            else:
                da         = abs(zone_color[1] - ref_color[1]) / nat_std_a
                db         = abs(zone_color[2] - ref_color[2]) / nat_std_b
                color_diff = float(np.sqrt(da**2 + db**2))

                # 85% couleur + 15% texture
                diff        = (color_diff * 0.85) + (tex_diff * 0.15)
                label_score = f"{diff:.1f}"

                if diff > 2.0:
                    color_rect = (0, 0, 255)
                    verdict    = "Repeinture!"
                    detected  += 1
                elif diff > 1.0:
                    color_rect = (0, 165, 255)
                    verdict    = "Suspect"
                    detected  += 1
                else:
                    color_rect = (0, 210, 0)
                    verdict    = "OK"

            cv2.rectangle(final_img, (abs_x1, abs_y1),
                          (abs_x2, abs_y2), color_rect, thick_box)

            overlay = final_img.copy()
            cv2.rectangle(overlay, (abs_x1, abs_y1),
                          (abs_x2, abs_y1 + overlay_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.45, final_img, 0.55, 0, final_img)

            short = zone["name"].replace(" Milieu","").replace(" Haut","▲").replace(" Bas","▼")
            cv2.putText(final_img, short,
                        (abs_x1 + 3, abs_y1 + int(overlay_h * 0.45)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale_big, (255, 255, 255), font_thick)

            cv2.putText(final_img, f"{label_score} {verdict}",
                        (abs_x1 + 3, abs_y1 + int(overlay_h * 0.88)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale_med, color_rect, font_thick)

            results_zones.append({
                "zone":    zone["name"],
                "col":     zone.get("col", 0),
                "row":     zone.get("row", 0),
                "score":   round(diff, 2),
                "pixels":  px_count,
                "verdict": verdict
            })

        # ===============================================
        # SCORE GLOBAL
        # ===============================================
        scores    = [z["score"] for z in results_zones if z["score"] > 0]
        score_raw = float(np.mean(scores)) if scores else 0.0
        score_100 = min(int(score_raw * 40), 100)

        if score_raw > 2.0:
            result = "Difference importante — repeinture probable"
        elif score_raw > 1.0:
            result = "Legeres variations detectees"
        else:
            result = "Peinture homogene (OK)"

        piece_scores  = consolidate_by_piece(results_zones)

        analysed_name = "analysed_" + filename
        analysed_path = os.path.join(UPLOAD_FOLDER, analysed_name)
        cv2.imwrite(analysed_path, final_img)

        if os.path.exists(resized_path):
            os.remove(resized_path)

        return jsonify({
            "yolo":           yolo_result,
            "score":          score_100,
            "score_raw":      round(score_raw, 2),
            "result":         result,
            "zones":          results_zones,
            "piece_scores":   piece_scores,
            "zones_detected": detected,
            "view_type":      view_type,
            "orientation":    orientation,
            "view_log":       view_log,
            "image_size":     {"width": orig_w, "height": orig_h},
            "calibration": {
                "nat_std_a":   round(nat_std_a, 1),
                "nat_std_b":   round(nat_std_b, 1),
                "ref_L":       round(ref_color[0], 1),
                "ref_a":       round(ref_color[1], 1),
                "ref_b":       round(ref_color[2], 1),
                "ref_texture": round(ref_texture, 1)
            },
            "image_result":   analysed_name,
            "image_url":      request.host_url + "uploads/" + analysed_name
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
