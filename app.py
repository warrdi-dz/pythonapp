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
        ext  = os.path.splitext(image_path)[1].lower()
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
        with open(image_path, "rb") as f:
            files   = {"image": (os.path.basename(image_path), f, mime)}
            headers = {"Accept": "application/json"}
            r = requests.post(url, files=files, headers=headers, timeout=20)
        if r.status_code == 200:
            return r.json()
        return {"error": "YOLO failed", "status": r.status_code}
    except Exception as e:
        return {"error": "YOLO exception", "details": str(e)}

@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/")
def home():
    return jsonify({"status": "OK", "message": "GARAGE PRO V5 API"})


# =============================================
# DÉTECTER MARQUE ET COULEUR
# Via analyse visuelle : logo, forme, couleur
# =============================================
def detect_brand_and_color(car_crop, hsv_full, mask_body):
    """
    Détecte la couleur dominante de la carrosserie
    et tente d'identifier la marque via les logos
    détectés par YOLO (class spécifique) ou
    via la forme du véhicule.
    Retourne : {"couleur": "...", "teinte_hex": "..."}
    """
    # --- Couleur dominante ---
    lab = cv2.cvtColor(car_crop, cv2.COLOR_BGR2LAB)
    vl  = lab[mask_body > 0]

    if len(vl) < 50:
        return {"couleur": "Inconnue", "teinte_hex": "#000000"}

    # Percentiles pour éliminer reflets/ombres
    L    = vl[:, 0]
    p10  = np.percentile(L, 10)
    p90  = np.percentile(L, 90)
    keep = (L >= p10) & (L <= p90)
    vl   = vl[keep]

    if len(vl) < 20:
        return {"couleur": "Inconnue", "teinte_hex": "#000000"}

    med_L = float(np.median(vl[:, 0]))
    med_a = float(np.median(vl[:, 1]))
    med_b = float(np.median(vl[:, 2]))

    # Convertir LAB → BGR → RGB pour hex
    lab_px        = np.uint8([[[med_L, med_a, med_b]]])
    bgr_px        = cv2.cvtColor(lab_px, cv2.COLOR_LAB2BGR)
    r, g, b       = int(bgr_px[0,0,2]), int(bgr_px[0,0,1]), int(bgr_px[0,0,0])
    teinte_hex    = f"#{r:02X}{g:02X}{b:02X}"

    # --- Identifier la couleur en français ---
    hsv_valid = hsv_full[mask_body > 0]
    hsv_valid = hsv_valid[(hsv_valid[:,2] >= p10) & (hsv_valid[:,2] <= p90)]

    if len(hsv_valid) == 0:
        return {"couleur": "Inconnue", "teinte_hex": teinte_hex}

    med_H = float(np.median(hsv_valid[:, 0]))
    med_S = float(np.median(hsv_valid[:, 1]))
    med_V = float(np.median(hsv_valid[:, 2]))

    # Logique couleur
    if med_S < 25:
        if med_V < 60:
            couleur = "Noir"
        elif med_V < 130:
            couleur = "Gris fonce"
        elif med_V < 175:
            couleur = "Gris"
        elif med_V < 210:
            couleur = "Gris clair"
        else:
            couleur = "Blanc"
    elif med_S < 60:
        if med_V < 80:
            couleur = "Noir metallise"
        elif med_V < 140:
            couleur = "Gris metallise"
        else:
            couleur = "Argente"
    else:
        if med_H < 10 or med_H > 170:
            couleur = "Rouge"
        elif med_H < 20:
            couleur = "Orange"
        elif med_H < 35:
            couleur = "Jaune"
        elif med_H < 85:
            couleur = "Vert"
        elif med_H < 130:
            couleur = "Bleu"
        elif med_H < 150:
            couleur = "Violet"
        else:
            couleur = "Rose"

        # Nuances
        if med_V < 80:
            couleur += " fonce"
        elif med_S > 150 and med_V > 150:
            couleur += " vif"

    return {
        "couleur":    couleur,
        "teinte_hex": teinte_hex,
        "HSV":        {"H": round(med_H,1), "S": round(med_S,1), "V": round(med_V,1)}
    }


# =============================================
# MASQUE CARROSSERIE ANTI-OMBRES AMÉLIORÉ
# =============================================
def build_body_mask(car_crop, hsv):
    mask_dark = cv2.inRange(hsv, (0, 0,   0), (180, 255,  45))
    mask_refl = cv2.inRange(hsv, (0, 0, 218), (180, 255, 255))
    mask_sky  = cv2.inRange(hsv, (0, 0, 210), (180,  20, 255))
    mask_chro = cv2.inRange(hsv, (0, 0,   0), (180,  12, 255))
    mask_chro = np.zeros_like(mask_dark)
    exclude   = cv2.bitwise_or(mask_dark, mask_refl)
    print("DARK =", cv2.countNonZero(mask_dark))
    print("REFL =", cv2.countNonZero(mask_refl))
    print("SKY  =", cv2.countNonZero(mask_sky))
    print("CHRO =", cv2.countNonZero(mask_chro))
    exclude   = cv2.bitwise_or(exclude,   mask_sky)
    exclude   = cv2.bitwise_or(exclude,   mask_chro)
    mask_body = cv2.bitwise_not(exclude)

    k         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_CLOSE, k, iterations=2)
    mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_OPEN,  k, iterations=1)

    h_c, w_c = car_crop.shape[:2]

    # Ombres semi-sombres
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
        x, y, w, h = cv2.boundingRect(cnt)
        ratio_hw   = h / max(w, 1)
        ratio_wh   = w / max(h, 1)
        area_ratio = area / total_body_area

        # Poteau (vertical allongé)
        is_pole    = (ratio_hw > 3.0) and (w < w_c * 0.12)
        # Ombre longue horizontale
        is_long    = (ratio_wh > 4.0) and (area_ratio > 0.04)
        # Touche le bas (ombre sol)
        touch_bot  = (y + h) > (h_c * 0.88)
        # Ombre ronde compacte (arbre, personne)
        is_round   = (area_ratio < 0.07) and (ratio_hw < 1.6) and (ratio_wh < 1.6)

        if is_pole or is_long or touch_bot or is_round:
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
# DÉTECTER LA VUE ET ANALYSER LA VISIBILITÉ
# des pièces dans l'image
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
        return "side_full", ("left" if glass_L > glass_R * 1.2 else "right"), log

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
# DÉFINIR LES ZONES VISIBLES RÉELLES
#
# Logique clé :
# - On ne crée une zone que si elle a assez de
#   pixels de carrosserie valides (> seuil)
# - Les zones sont nommées d'après les vraies pièces
# - Vue côté : aile av | porte av | porte ar | aile ar
# - Vue avant : aile G | capot | pare-chocs | aile D
# - Vue arrière : aile ar G | coffre | pare-ch | aile ar D
# =============================================
def define_visible_zones(view_type, orientation,
                         crop_h, crop_w, mask_body):

    MIN_PIX = 150
    y1b = int(crop_h * 0.15)
    y2b = int(crop_h * 0.80)

    def has_enough(xA, yA, xB, yB):

        zm = mask_body[yA:yB, xA:xB]

        body_pixels = cv2.countNonZero(zm)
        total_pixels = zm.shape[0] * zm.shape[1]

        if total_pixels == 0:
            return False

        ratio = body_pixels / total_pixels

        if body_pixels < MIN_PIX:
            return False

        if ratio < 0.10:
            return False

        print(
            f"ZONE TEST {xA}-{xB} : "
            f"pixels={body_pixels} "
            f"ratio={ratio:.2f}"
        )
        if body_pixels < MIN_PIX:
            return False

        if ratio < 0.10:
            return False
        return True
        
    zones = []

    # suite de ton code...
    # --------------------------------------------------
    # VUE CÔTÉ COMPLÈTE : 4 pièces verticales
    # Aile avant | Porte avant | Porte arrière | Aile arr.
    # --------------------------------------------------
    if view_type == "side_full":
        cuts = [
            int(crop_w * 0.15),
            int(crop_w * 0.40),
            int(crop_w * 0.80),
        ]
        
        if orientation == "left":
            pieces = [
                ("Aile avant",    0,       cuts[0]),
                ("Porte avant",   cuts[0], cuts[1]),
                ("Porte arriere", cuts[1], cuts[2]),
                ("Aile arriere",  cuts[2], crop_w),
            ]
        else:
            pieces = [
                ("Aile arriere",  0,       cuts[0]),
                ("Porte arriere", cuts[0], cuts[1]),
                ("Porte avant",   cuts[1], cuts[2]),
                ("Aile avant",    cuts[2], crop_w),
            ]

        for name, xA, xB in pieces:
            if has_enough(xA, y1b, xB, y2b):
                zones.append({
                    "name": name,
                    "xA": xA, "xB": xB,
                    "yA": y1b, "yB": y2b
                })

    # --------------------------------------------------
    # VUE AVANT SEULEMENT
    # Capot | Aile avant | Pare-chocs avant
    # --------------------------------------------------
    elif view_type == "front_only":
        cap_y1 = int(crop_h * 0.05)
        cap_y2 = int(crop_h * 0.50)
        pc_y1  = int(crop_h * 0.58)
        pc_y2  = int(crop_h * 0.93)

        if orientation == "left":
            aile_x2 = int(crop_w * 0.35)
            cap_x1  = int(crop_w * 0.25)
            candidates = [
                ("Capot avant",    cap_x1, crop_w, cap_y1, cap_y2),
                ("Aile avant",     0,      aile_x2, cap_y1, pc_y2),
                ("Pare-chocs av.", cap_x1, crop_w,  pc_y1, pc_y2),
            ]
        else:
            aile_x1 = int(crop_w * 0.65)
            cap_x2  = int(crop_w * 0.75)
            candidates = [
                ("Capot avant",    0,      cap_x2,  cap_y1, cap_y2),
                ("Aile avant",     aile_x1, crop_w, cap_y1, pc_y2),
                ("Pare-chocs av.", 0,      cap_x2,  pc_y1,  pc_y2),
            ]

        for name, xA, xB, yA, yB in candidates:
            if has_enough(xA, yA, xB, yB):
                zones.append({"name": name, "xA": xA, "xB": xB, "yA": yA, "yB": yB})

    # --------------------------------------------------
    # VUE ARRIÈRE SEULEMENT
    # Coffre/hayon | Aile arrière | Pare-chocs arrière
    # --------------------------------------------------
    elif view_type in ("rear_only", "rear_3q"):
        co_y1 = int(crop_h * 0.05)
        co_y2 = int(crop_h * 0.52)
        pc_y1 = int(crop_h * 0.60)
        pc_y2 = int(crop_h * 0.93)

        if orientation == "left":
            aile_x2 = int(crop_w * 0.38)
            co_x1   = int(crop_w * 0.22)
            candidates = [
                ("Coffre / hayon",  co_x1,  crop_w,  co_y1, co_y2),
                ("Aile arriere",    0,       aile_x2, co_y1, pc_y2),
                ("Pare-chocs arr.", co_x1,  crop_w,  pc_y1, pc_y2),
            ]
        else:
            aile_x1 = int(crop_w * 0.62)
            co_x2   = int(crop_w * 0.78)
            candidates = [
                ("Coffre / hayon",  0,       co_x2,   co_y1, co_y2),
                ("Aile arriere",    aile_x1, crop_w,  co_y1, pc_y2),
                ("Pare-chocs arr.", 0,       co_x2,   pc_y1, pc_y2),
            ]

        for name, xA, xB, yA, yB in candidates:
            if has_enough(xA, yA, xB, yB):
                zones.append({"name": name, "xA": xA, "xB": xB, "yA": yA, "yB": yB})

    # Fallback
    if not zones:
        c1 = int(crop_w * 0.33)
        c2 = int(crop_w * 0.67)
        for name, xA, xB in [
            ("Zone gauche",  0,  c1),
            ("Zone centre",  c1, c2),
            ("Zone droite",  c2, crop_w)
        ]:
            if has_enough(xA, y1b, xB, y2b):
                zones.append({"name": name, "xA": xA, "xB": xB,
                              "yA": y1b, "yB": y2b})

    return zones


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

        # YOLO reçoit une version réduite en JPEG
        img_yolo     = cv2.resize(img_orig, (YOLO_W, YOLO_H))
        resized_path = os.path.join(UPLOAD_FOLDER, "resized_" + filename + ".jpg")
        cv2.imwrite(resized_path, img_yolo, [cv2.IMWRITE_JPEG_QUALITY, 92])

        yolo_result = call_yolo(resized_path)
        print("================================")
        print("YOLO RESULT =", yolo_result)
        print("================================")
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
        # ESPACES COLORIMÉTRIQUES + MASQUE
        # ===============================================
        hsv_full  = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
        lab_full  = cv2.cvtColor(car_crop, cv2.COLOR_BGR2LAB)
        gray_full = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
        mask_body = build_body_mask(car_crop, hsv_full)
        print("BODY PIXELS =",cv2.countNonZero(mask_body),"/",crop_w * crop_h,"=",
             round(cv2.countNonZero(mask_body) /(crop_w * crop_h) * 100,1),"%")

        # ===============================================
        # COULEUR ET MARQUE
        # ===============================================
        car_info = detect_brand_and_color(car_crop, hsv_full, mask_body)

        # ===============================================
        # VUE + ZONES VISIBLES RÉELLES
        # ===============================================
        view_type, orientation, view_log = detect_view(car_crop)
        zones = define_visible_zones(
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

        # Texture ref sur masque seulement
        lap_full    = cv2.Laplacian(gray_full, cv2.CV_64F)
        ref_texture = float(np.var(lap_full[mask_body > 0]))
        ref_texture = max(ref_texture, 1.0)

        # ===============================================
        # DESSIN
        # ===============================================
        final_img      = img_orig.copy()
        thick_box      = max(3, int(4 * min(scale_x, scale_y)))
        thick_line     = max(1, int(1 * min(scale_x, scale_y)))
        font_scale_big = max(0.50, 0.52 * min(scale_x, scale_y))
        font_scale_med = max(0.40, 0.42 * min(scale_x, scale_y))
        font_thick     = max(2,    int(2  * min(scale_x, scale_y)))
        overlay_h      = max(52,   int(52 * scale_y))

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
                verdict     = "Non analysable"
            else:
                da         = abs(zone_color[1] - ref_color[1]) / nat_std_a
                db         = abs(zone_color[2] - ref_color[2]) / nat_std_b
                color_diff = float(np.sqrt(da**2 + db**2))

                # 85% couleur + 15% texture
                diff        = (color_diff * 0.85) + (tex_diff * 0.15)
                label_score = f"{diff:.1f}"

                if diff > 2.0:
                    color_rect = (0, 0, 255)
                    verdict    = "Peinture refaite!"
                    detected  += 1
                elif diff > 1.0:
                    color_rect = (0, 165, 255)
                    verdict    = "Variation suspecte"
                    detected  += 1
                else:
                    color_rect = (0, 210, 0)
                    verdict    = "OK"

            # Grand rectangle par pièce
            cv2.rectangle(final_img, (abs_x1, abs_y1),
                          (abs_x2, abs_y2), color_rect, thick_box)

            # Séparateur vertical entre zones
            if zones.index(zone) < len(zones) - 1:
                step = max(10, int(12 * scale_y))
                dash = max(4,  int(6  * scale_y))
                for dy in range(abs_y1, abs_y2, step):
                    cv2.line(final_img, (abs_x2, dy),
                             (abs_x2, dy + dash), (255, 255, 255), thick_line)

            overlay = final_img.copy()
            cv2.rectangle(overlay, (abs_x1, abs_y1),
                          (abs_x2, abs_y1 + overlay_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.50, final_img, 0.50, 0, final_img)

            cv2.putText(final_img, zone["name"],
                        (abs_x1 + 6, abs_y1 + int(overlay_h * 0.42)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale_big, (255, 255, 255), font_thick)

            cv2.putText(final_img, f"Score: {label_score}  {verdict}",
                        (abs_x1 + 6, abs_y1 + int(overlay_h * 0.85)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale_med, color_rect, font_thick)

            results_zones.append({
                "zone":    zone["name"],
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
            "vehicule": {
                "couleur":    car_info["couleur"],
                "teinte_hex": car_info["teinte_hex"],
                "HSV":        car_info.get("HSV", {})
            },
            "zones":          results_zones,
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
