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

@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/")
def home():
    return jsonify({"status": "OK", "message": "GARAGE PRO V4 API"})


# =============================================
# MASQUE CARROSSERIE ANTI-OMBRES
#
# Le problème des ombres :
# Une ombre sur carrosserie = zone sombre MAIS
# avec la MÊME teinte que la carrosserie autour.
# Un poteau / arbre / sol projette une ombre
# qui a une forme géométrique ou organique.
#
# Solution : après avoir créé le masque de base,
# on détecte les régions sombres CONNEXES et on
# regarde si leur forme est suspecte (trop allongée,
# trop petite, bord droit = ombre de poteau).
# On les exclut du masque.
# =============================================
def build_body_mask(car_crop, hsv):

    # --- Exclusions de base ---
    # Trop sombre = vitres, pneus
    mask_dark = cv2.inRange(hsv, (0, 0,   0), (180, 255,  45))
    # Reflets blancs forts = soleil
    mask_refl = cv2.inRange(hsv, (0, 0, 218), (180, 255, 255))
    # Ciel / fond blanc
    mask_sky  = cv2.inRange(hsv, (0, 0, 210), (180,  20, 255))
    # Chrome / plastique (faible saturation)
    mask_chro = cv2.inRange(hsv, (0, 0,   0), (180,  28, 255))

    exclude   = cv2.bitwise_or(mask_dark, mask_refl)
    exclude   = cv2.bitwise_or(exclude,   mask_sky)
    exclude   = cv2.bitwise_or(exclude,   mask_chro)
    mask_body = cv2.bitwise_not(exclude)

    k         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_CLOSE, k, iterations=2)
    mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_OPEN,  k, iterations=1)

    # -----------------------------------------------
    # DÉTECTION ET SUPPRESSION DES OMBRES
    #
    # Une ombre sur carrosserie = zone où V est
    # nettement plus bas que ses voisins MAIS
    # H et S restent similaires à la carrosserie.
    #
    # On détecte les zones "semi-sombres" (V entre
    # 45 et 110) qui sont incluses dans le masque
    # carrosserie, puis on analyse leur forme :
    # - ratio largeur/hauteur extrême → poteau
    # - surface trop petite → bruit
    # - forme très allongée → ombre d'arbre
    # -----------------------------------------------
    gray = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
    h_c, w_c = car_crop.shape[:2]

    # Zones semi-sombres sur carrosserie
    mask_semi = cv2.inRange(hsv, (0, 0, 45), (180, 255, 115))
    # Garder seulement celles qui sont dans la carrosserie
    mask_shadow_on_body = cv2.bitwise_and(mask_semi, mask_body)

    # Morphologie pour regrouper les zones proches
    ks = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_shadow_on_body = cv2.morphologyEx(
        mask_shadow_on_body, cv2.MORPH_CLOSE, ks, iterations=3
    )

    # Trouver les contours des zones d'ombre
    contours, _ = cv2.findContours(
        mask_shadow_on_body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    mask_shadows_to_remove = np.zeros_like(mask_body)
    total_body_area = max(cv2.countNonZero(mask_body), 1)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue  # trop petit = bruit

        x, y, w, h = cv2.boundingRect(cnt)

        # Ratio : ombre de poteau = très allongée verticalement
        ratio_hw = h / max(w, 1)
        ratio_wh = w / max(h, 1)

        # Surface relative à la carrosserie
        area_ratio = area / total_body_area

        # Ombre de poteau : très haute et étroite
        is_pole_shadow = (ratio_hw > 3.0) and (w < w_c * 0.12)

        # Ombre d'arbre / bâtiment : grande zone très allongée
        is_long_shadow = (ratio_wh > 4.0) and (area_ratio > 0.04)

        # Ombre ronde compacte (arbre, personne)
        # qui occupe moins de 8% de la carrosserie
        is_small_round = (area_ratio < 0.06) and (ratio_hw < 1.8) and (ratio_wh < 1.8)

        # Ombre qui touche le bord inférieur du crop
        # (ombre du sol qui remonte)
        touches_bottom = (y + h) > (h_c * 0.88)

        if is_pole_shadow or is_long_shadow or touches_bottom:
            # Supprimer cette zone du masque carrosserie
            cv2.drawContours(
                mask_shadows_to_remove, [cnt], -1, 255, -1
            )

    # Appliquer la suppression des ombres
    mask_body = cv2.bitwise_and(
        mask_body,
        cv2.bitwise_not(mask_shadows_to_remove)
    )

    # Morphologie finale pour lisser
    mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_CLOSE, k, iterations=1)

    return mask_body


# =============================================
# COULEUR LAB MÉDIANE ANTI-REFLETS
# Exclut les 10% extrêmes de luminosité
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

    # Feux arrière rouges
    mr1      = cv2.inRange(hsv, (0,   60, 60), (12,  255, 255))
    mr2      = cv2.inRange(hsv, (168, 60, 60), (180, 255, 255))
    mask_red = cv2.bitwise_or(mr1, mr2)
    red_tot  = cv2.countNonZero(mask_red)
    red_L    = cv2.countNonZero(mask_red[:, :crop_w//2])
    red_R    = cv2.countNonZero(mask_red[:, crop_w//2:])

    # Phares avant (seulement dans le tiers bas)
    mw1       = cv2.inRange(hsv, (0,  0,  195), (180, 50, 255))
    mw2       = cv2.inRange(hsv, (15, 40, 195), (40, 180, 255))
    ph_zone   = cv2.bitwise_or(mw1, mw2)[int(crop_h*0.45):, :]
    white_tot = cv2.countNonZero(ph_zone)
    white_L   = cv2.countNonZero(ph_zone[:, :crop_w//2])
    white_R   = cv2.countNonZero(ph_zone[:, crop_w//2:])

    # Vitres (partie haute)
    top       = gray[int(crop_h*0.05):int(crop_h*0.55), :]
    dk        = (top < 75).astype(np.uint8)
    dk_f      = cv2.GaussianBlur(dk.astype(np.float32), (15, 15), 0)
    glass_L   = float(dk_f[:, :crop_w//2].sum())
    glass_R   = float(dk_f[:, crop_w//2:].sum())
    glass_tot = glass_L + glass_R

    has_doors = (glass_tot > 8000) and (ratio_wh > 1.3)

    log.append(f"ratio={ratio_wh:.2f} rouge={red_tot} "
               f"blanc={white_tot} vitres={int(glass_tot)} "
               f"portes={has_doors}")

    # CAS 1 : Côté complet avec portes
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
        if glass_L > glass_R * 1.2:
            return "side_full", "left", log
        return "side_full", "right", log

    # CAS 2 : Avant seulement
    if white_tot > 120 and red_tot < 150 and not has_doors:
        log.append("→ FRONT ONLY")
        return "front_only", ("left" if white_L > white_R else "right"), log

    # CAS 3 : Arrière seulement
    if red_tot > 150 and white_tot < 100 and not has_doors:
        log.append("→ REAR ONLY")
        return "rear_only", ("left" if red_L > red_R else "right"), log

    # CAS 4 : 3/4 arrière
    if red_tot > 100 and ratio_wh < 1.5:
        log.append("→ REAR 3Q")
        return "rear_3q", ("left" if red_L > red_R else "right"), log

    log.append("→ FALLBACK side_full")
    return "side_full", "left", log


# =============================================
# DÉFINIR LES ZONES SELON LA VUE
# =============================================
def define_zones(view_type, orientation, crop_h, crop_w):
    band_y1 = int(crop_h * 0.08)
    band_y2 = int(crop_h * 0.90)

    if view_type == "side_full":
        cut1 = int(crop_w * 0.25)
        cut2 = int(crop_w * 0.65)
        if orientation == "left":
            return [
                {"name": "Aile avant",   "xA": 0,    "xB": cut1,   "yA": band_y1, "yB": band_y2},
                {"name": "Portes",       "xA": cut1, "xB": cut2,   "yA": band_y1, "yB": band_y2},
                {"name": "Aile arriere", "xA": cut2, "xB": crop_w, "yA": band_y1, "yB": band_y2},
            ]
        else:
            return [
                {"name": "Aile arriere", "xA": 0,    "xB": cut1,   "yA": band_y1, "yB": band_y2},
                {"name": "Portes",       "xA": cut1, "xB": cut2,   "yA": band_y1, "yB": band_y2},
                {"name": "Aile avant",   "xA": cut2, "xB": crop_w, "yA": band_y1, "yB": band_y2},
            ]

    elif view_type == "front_only":
        cap_y1 = int(crop_h * 0.08)
        cap_y2 = int(crop_h * 0.52)
        pc_y1  = int(crop_h * 0.60)
        pc_y2  = int(crop_h * 0.92)
        if orientation == "left":
            ax2 = int(crop_w * 0.38)
            cx1 = int(crop_w * 0.30)
            return [
                {"name": "Capot avant",    "xA": cx1, "xB": crop_w, "yA": cap_y1, "yB": cap_y2},
                {"name": "Aile avant",     "xA": 0,   "xB": ax2,    "yA": cap_y1, "yB": pc_y2},
                {"name": "Pare-chocs av.", "xA": cx1, "xB": crop_w, "yA": pc_y1,  "yB": pc_y2},
            ]
        else:
            ax1 = int(crop_w * 0.62)
            cx2 = int(crop_w * 0.70)
            return [
                {"name": "Capot avant",    "xA": 0,   "xB": cx2,    "yA": cap_y1, "yB": cap_y2},
                {"name": "Aile avant",     "xA": ax1, "xB": crop_w, "yA": cap_y1, "yB": pc_y2},
                {"name": "Pare-chocs av.", "xA": 0,   "xB": cx2,    "yA": pc_y1,  "yB": pc_y2},
            ]

    elif view_type == "rear_only":
        co_y1 = int(crop_h * 0.08)
        co_y2 = int(crop_h * 0.52)
        pc_y1 = int(crop_h * 0.62)
        pc_y2 = int(crop_h * 0.92)
        if orientation == "left":
            ax2 = int(crop_w * 0.40)
            cx1 = int(crop_w * 0.25)
            return [
                {"name": "Coffre / hayon",  "xA": cx1, "xB": crop_w, "yA": co_y1, "yB": co_y2},
                {"name": "Aile arriere",    "xA": 0,   "xB": ax2,    "yA": co_y1, "yB": pc_y2},
                {"name": "Pare-chocs arr.", "xA": cx1, "xB": crop_w, "yA": pc_y1, "yB": pc_y2},
            ]
        else:
            ax1 = int(crop_w * 0.60)
            cx2 = int(crop_w * 0.75)
            return [
                {"name": "Coffre / hayon",  "xA": 0,   "xB": cx2,    "yA": co_y1, "yB": co_y2},
                {"name": "Aile arriere",    "xA": ax1, "xB": crop_w, "yA": co_y1, "yB": pc_y2},
                {"name": "Pare-chocs arr.", "xA": 0,   "xB": cx2,    "yA": pc_y1, "yB": pc_y2},
            ]

    elif view_type == "rear_3q":
        co_y2 = int(crop_h * 0.50)
        pc_y1 = int(crop_h * 0.60)
        if orientation == "left":
            cut = int(crop_w * 0.42)
            return [
                {"name": "Coffre / hayon",  "xA": cut, "xB": crop_w, "yA": band_y1, "yB": co_y2},
                {"name": "Aile arriere",    "xA": 0,   "xB": cut,    "yA": band_y1, "yB": band_y2},
                {"name": "Pare-chocs arr.", "xA": cut, "xB": crop_w, "yA": pc_y1,   "yB": band_y2},
            ]
        else:
            cut = int(crop_w * 0.58)
            return [
                {"name": "Coffre / hayon",  "xA": 0,   "xB": cut,    "yA": band_y1, "yB": co_y2},
                {"name": "Aile arriere",    "xA": cut, "xB": crop_w, "yA": band_y1, "yB": band_y2},
                {"name": "Pare-chocs arr.", "xA": 0,   "xB": cut,    "yA": pc_y1,   "yB": band_y2},
            ]

    # Fallback
    cut1 = int(crop_w * 0.33)
    cut2 = int(crop_w * 0.67)
    return [
        {"name": "Zone gauche",  "xA": 0,    "xB": cut1,   "yA": band_y1, "yB": band_y2},
        {"name": "Zone centre",  "xA": cut1, "xB": cut2,   "yA": band_y1, "yB": band_y2},
        {"name": "Zone droite",  "xA": cut2, "xB": crop_w, "yA": band_y1, "yB": band_y2},
    ]


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

        img_yolo     = cv2.resize(img_orig, (YOLO_W, YOLO_H))
        resized_path = os.path.join(UPLOAD_FOLDER, "resized_" + filename)
        cv2.imwrite(resized_path, img_yolo)
        yolo_result  = call_yolo(resized_path)

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
        # VUE + ZONES
        # ===============================================
        view_type, orientation, view_log = detect_view(car_crop)
        zones = define_zones(view_type, orientation, crop_h, crop_w)

        # ===============================================
        # MASQUE ANTI-OMBRES + ESPACES COLORIMÉTRIQUES
        # ===============================================
        hsv_full  = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
        lab_full  = cv2.cvtColor(car_crop, cv2.COLOR_BGR2LAB)
        mask_body = build_body_mask(car_crop, hsv_full)

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
        gray_full = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
        ref_texture = cv2.Laplacian(gray_full,cv2.CV_64F).var()
        nat_std_a = max(float(np.std(vl_ref[:, 1])), 1.0)
        nat_std_b = max(float(np.std(vl_ref[:, 2])), 1.0)

        # ===============================================
        # DESSIN
        # ===============================================
        final_img      = img_orig.copy()
        thick_box      = max(3, int(5 * min(scale_x, scale_y)))
        thick_line     = max(1, int(1 * min(scale_x, scale_y)))
        font_scale_big = max(0.55, 0.55 * min(scale_x, scale_y))
        font_scale_med = max(0.45, 0.45 * min(scale_x, scale_y))
        font_thick_big = max(2,    int(2  * min(scale_x, scale_y)))
        overlay_h      = max(55,   int(55 * scale_y))

        cv2.rectangle(final_img, (x1, y1), (x2, y2), (220, 220, 220), thick_line)

        drawn_x = set()
        step = max(10, int(12 * scale_y))
        dash = max(4,  int(6  * scale_y))
        for zone in zones:
            for cx in [zone["xA"], zone["xB"]]:
                if cx in drawn_x or cx == 0 or cx == crop_w:
                    continue
                drawn_x.add(cx)
                for dy in range(y1 + zone["yA"], y1 + zone["yB"], step):
                    cv2.line(final_img,
                             (x1 + cx, dy),
                             (x1 + cx, dy + dash),
                             (255, 255, 255), thick_line)

        results_zones = []
        detected = 0

        for zone in zones:
            xA, xB = zone["xA"], zone["xB"]
            yA, yB = zone["yA"], zone["yB"]

            zone_color, px_count = get_zone_color(
                lab_full, mask_body, xA, yA, xB, yB
            )
            gray_zone = gray_full[yA:yB, xA:xB]
            texture_score = cv2.Laplacian(gray_zone,cv2.CV_64F).var()
            texture_diff = abs(texture_score - ref_texture) / max(ref_texture, 1)
            texture_diff = min(texture_diff, 2.0)
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
                da   = abs(zone_color[1] - ref_color[1]) / nat_std_a
                db   = abs(zone_color[2] - ref_color[2]) / nat_std_b
                color_diff = float(np.sqrt(da**2 + db**2))
                diff = (color_diff * 0.8) + (texture_diff * 0.2)
                label_score = f"{diff:.1f}"

                if diff > 2.2:
                    color_rect = (0, 0, 255)
                    verdict    = "Peinture refaite!"
                    detected  += 1
                elif diff > 1.2:
                    color_rect = (0, 165, 255)
                    verdict    = "Variation suspecte"
                    detected  += 1
                else:
                    color_rect = (0, 210, 0)
                    verdict    = "OK"

            cv2.rectangle(final_img, (abs_x1, abs_y1),
                          (abs_x2, abs_y2), color_rect, thick_box)

            overlay = final_img.copy()
            cv2.rectangle(overlay, (abs_x1, abs_y1),
                          (abs_x2, abs_y1 + overlay_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, final_img, 0.5, 0, final_img)

            cv2.putText(final_img, zone["name"],
                        (abs_x1 + 8, abs_y1 + int(overlay_h * 0.40)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale_big, (255, 255, 255), font_thick_big)

            cv2.putText(final_img, f"Score: {label_score}",
                        (abs_x1 + 8, abs_y1 + int(overlay_h * 0.80)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale_med, color_rect, font_thick_big)

            cv2.putText(final_img, verdict,
                        (abs_x1 + 8, abs_y2 - int(10 * scale_y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale_med, color_rect, font_thick_big)

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

        if score_raw > 1.8:
            result = "Difference importante — repeinture probable"
        elif score_raw > 0.9:
            result = "Legeres variations detectees"
        else:
            result = "Peinture homogene (OK)"

        analysed_name = "analysed_" + filename
        analysed_path = os.path.join(UPLOAD_FOLDER, analysed_name)
        cv2.imwrite(analysed_path, final_img)

        if os.path.exists(resized_path):
            os.remove(resized_path)

        return jsonify({
            "yolo":          yolo_result,
            "score":         score_100,
            "score_raw":     round(score_raw, 2),
            "result":        result,
            "zones":         results_zones,
            "zones_detected":detected,
            "view_type":     view_type,
            "orientation":   orientation,
            "view_log":      view_log,
            "image_size":    {"width": orig_w, "height": orig_h},
            "calibration": {
                "nat_std_a": round(nat_std_a, 1),
                "nat_std_b": round(nat_std_b, 1),
                "ref_L":     round(ref_color[0], 1),
                "ref_a":     round(ref_color[1], 1),
                "ref_b":     round(ref_color[2], 1)
            },
            "image_result":  analysed_name,
            "image_url":     request.host_url + "uploads/" + analysed_name
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
