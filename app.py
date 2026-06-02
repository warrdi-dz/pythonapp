from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
import time
import traceback
import requests
from werkzeug.utils import secure_filename
from scipy import stats

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
    return jsonify({"status": "OK", "message": "GARAGE PRO V5 API"})


# =============================================
# MASQUE CARROSSERIE STRICT
# Retire : vitres, roues, plastique, reflets,
# ciel, sol, pixels surexposés
# =============================================
def build_body_mask(car_crop):
    hsv  = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
    lab  = cv2.cvtColor(car_crop, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)

    # Trop sombre = vitres, pneus, joints
    mask_dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, 40))

    # Surexposé = reflets soleil (V > 225)
    mask_bright = cv2.inRange(hsv, (0, 0, 225), (180, 255, 255))

    # Faible saturation = chrome, plastique, calandre
    mask_chrome = cv2.inRange(hsv, (0, 0, 0), (180, 35, 255))

    # Fond blanc/ciel
    mask_sky = cv2.inRange(hsv, (0, 0, 210), (180, 15, 255))

    # Combiner toutes les exclusions
    mask_exclude = cv2.bitwise_or(mask_dark,   mask_bright)
    mask_exclude = cv2.bitwise_or(mask_exclude, mask_chrome)
    mask_exclude = cv2.bitwise_or(mask_exclude, mask_sky)
    mask_body    = cv2.bitwise_not(mask_exclude)

    # Morphologie pour nettoyer
    k         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_CLOSE,  k, iterations=2)
    mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_OPEN,   k, iterations=1)

    return mask_body, hsv, lab, gray


# =============================================
# VRAIE TEXTURE : LBP LOCAL BINARY PATTERN
# Beaucoup plus précis que np.std
# Mesure le motif local de chaque pixel
# =============================================
def compute_lbp(gray_img):
    """
    LBP = Local Binary Pattern
    Pour chaque pixel, compare ses 8 voisins.
    Génère un code binaire = signature de texture locale.
    Une peinture refaite aura une distribution LBP différente.
    """
    h, w    = gray_img.shape
    lbp_img = np.zeros((h, w), dtype=np.uint8)

    # 8 directions : N, NE, E, SE, S, SW, W, NW
    offsets = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]

    gray_f = gray_img.astype(np.float32)

    for bit, (dy, dx) in enumerate(offsets):
        shifted = np.roll(np.roll(gray_f, dy, axis=0), dx, axis=1)
        lbp_img |= ((gray_f >= shifted).astype(np.uint8) << bit)

    return lbp_img


# =============================================
# FEATURES COMPLÈTES D'UNE ZONE
# Couleur LAB (a,b) + teinte H + texture LBP
# + gradient surface + distribution couleur
# =============================================
def extract_zone_features(lab, hsv, gray, lbp, mask, xA, yA, xB, yB):
    """
    Extrait un vecteur de features robuste pour une zone.
    Ignore la luminosité (L et V) pour éviter l'effet soleil.
    """
    zm  = mask[yA:yB, xA:xB]
    zl  = lab[yA:yB, xA:xB]
    zh  = hsv[yA:yB, xA:xB]
    zg  = gray[yA:yB, xA:xB]
    zlbp= lbp[yA:yB, xA:xB]

    valid_idx = zm > 0
    vl  = zl[valid_idx]    # pixels LAB valides
    vh  = zh[valid_idx]    # pixels HSV valides
    vg  = zg[valid_idx]    # pixels gris valides
    vlbp= zlbp[valid_idx]  # pixels LBP valides

    if len(vl) < 100:
        return None

    # --- COULEUR : canaux a,b de LAB (sans L) ---
    a_med = float(np.median(vl[:, 1]))
    b_med = float(np.median(vl[:, 2]))
    a_std = float(np.std(vl[:, 1]))
    b_std = float(np.std(vl[:, 2]))

    # --- TEINTE H de HSV (circulaire, sans V) ---
    h_vals = vl[:, 0].astype(np.float32)  # H de HSV
    h_med  = float(np.median(vh[:, 0]))
    s_med  = float(np.median(vh[:, 1]))   # saturation

    # --- TEXTURE LBP : histogramme normalisé ---
    lbp_hist, _ = np.histogram(vlbp, bins=16, range=(0, 255))
    lbp_hist     = lbp_hist.astype(np.float32)
    total        = lbp_hist.sum()
    if total > 0:
        lbp_hist /= total

    # --- GRADIENT : mesure la régularité de surface ---
    # Un reflet isolé → gradient fort localement
    # Une repeinture → gradient uniforme sur toute la zone
    gz_zone = zg.astype(np.float32)
    sobel_x = cv2.Sobel(gz_zone, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gz_zone, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x**2 + sobel_y**2)

    # Gradient uniquement sur pixels valides
    grad_valid = gradient[valid_idx]
    grad_med   = float(np.median(grad_valid))
    grad_std   = float(np.std(grad_valid))

    # --- DISTRIBUTION COULEUR : percentiles ---
    # Capture mieux qu'une médiane seule
    a_p25 = float(np.percentile(vl[:, 1], 25))
    a_p75 = float(np.percentile(vl[:, 1], 75))
    b_p25 = float(np.percentile(vl[:, 2], 25))
    b_p75 = float(np.percentile(vl[:, 2], 75))

    return {
        # Couleur teinte pure (sans luminosité)
        "a_med":    a_med,
        "b_med":    b_med,
        "a_std":    a_std,
        "b_std":    b_std,
        "h_med":    h_med,
        "s_med":    s_med,
        # Texture LBP (vrai grain de peinture)
        "lbp_hist": lbp_hist,
        # Gradient (régularité de surface)
        "grad_med": grad_med,
        "grad_std": grad_std,
        # Distribution percentiles
        "a_p25":    a_p25,
        "a_p75":    a_p75,
        "b_p25":    b_p25,
        "b_p75":    b_p75,
        "count":    int(len(vl))
    }


# =============================================
# SCORE COMPLET DE DIFFÉRENCE
# Compare une zone à la référence globale
# sur TOUS les axes : couleur, texture, gradient
# =============================================
def compare_zone_to_ref(zone_f, ref_f):
    """
    Score multi-axes de différence de peinture.

    Axes comparés (luminosité EXCLUE) :
    1. Delta teinte LAB (a,b)          → poids 35%
    2. Distance texture LBP            → poids 25%
    3. Delta teinte H circulaire       → poids 20%
    4. Delta gradient (régularité)     → poids 10%
    5. Delta distribution (percentiles)→ poids 10%
    """

    # 1. Delta teinte LAB (a,b) — sans L
    delta_ab = float(np.sqrt(
        (zone_f["a_med"] - ref_f["a_med"])**2 +
        (zone_f["b_med"] - ref_f["b_med"])**2
    ))

    # Bonus si les distributions sont aussi différentes
    delta_ab_dist = float(np.sqrt(
        ((zone_f["a_p75"] - zone_f["a_p25"]) -
         (ref_f["a_p75"]  - ref_f["a_p25"]))**2 +
        ((zone_f["b_p75"] - zone_f["b_p25"]) -
         (ref_f["b_p75"]  - ref_f["b_p25"]))**2
    ))
    delta_couleur = delta_ab + delta_ab_dist * 0.3

    # 2. Distance texture LBP (chi-squared)
    # Mesure si les motifs de surface sont différents
    lbp_z = zone_f["lbp_hist"]
    lbp_r = ref_f["lbp_hist"]
    eps   = 1e-10
    chi2  = float(np.sum(
        (lbp_z - lbp_r)**2 / (lbp_r + eps)
    ))
    delta_texture = min(chi2 * 10.0, 50.0)

    # 3. Delta teinte H circulaire
    dh = abs(zone_f["h_med"] - ref_f["h_med"])
    if dh > 90:
        dh = 180 - dh
    delta_h = float(dh)

    # 4. Delta gradient (régularité de surface)
    delta_grad = abs(zone_f["grad_med"] - ref_f["grad_med"])

    # 5. Delta saturation
    delta_s = abs(zone_f["s_med"] - ref_f["s_med"])

    # Score final pondéré
    score = (
        delta_couleur * 0.35 +
        delta_texture * 0.25 +
        delta_h       * 0.20 +
        delta_grad    * 0.10 +
        delta_s       * 0.10
    )

    return {
        "score":          round(score, 1),
        "delta_couleur":  round(delta_couleur, 1),
        "delta_texture":  round(delta_texture, 1),
        "delta_h":        round(delta_h, 1),
        "delta_grad":     round(delta_grad, 1),
        "delta_s":        round(delta_s, 1)
    }


# =============================================
# COHÉRENCE INTERNE : 3 BANDES
# Si les 3 bandes ont même écart → vraie repeinture
# Si seulement 1 bande → reflet ou ombre
# =============================================
def check_coherence(lab, hsv, gray, lbp, mask,
                    xA, yA, xB, yB, ref_f):
    zone_h  = yB - yA
    band_h  = zone_h // 3
    scores  = []

    for b in range(3):
        byA = yA + b * band_h
        byB = yA + (b + 1) * band_h if b < 2 else yB
        f   = extract_zone_features(
            lab, hsv, gray, lbp, mask, xA, byA, xB, byB
        )
        if f is None:
            continue
        r = compare_zone_to_ref(f, ref_f)
        scores.append(r["score"])

    if len(scores) < 2:
        return 0.0, False

    std_bandes = float(np.std(scores))
    mean_score = float(np.mean(scores))

    # Cohérente = toutes les bandes sont également différentes
    is_coherent   = (std_bandes < 10.0) and (mean_score > 8.0)
    bonus         = 12.0 if is_coherent else 0.0

    return bonus, is_coherent


# =============================================
# RACCORD DE PEINTURE AUX BORDS
# Détecte une discontinuité de teinte (a,b)
# exactement à la frontière entre zones
# =============================================
def detect_raccord(lab, mask, x_border, yA, yB):
    h, w  = lab.shape[:2]
    bw    = 50
    x1b   = max(0, x_border - bw)
    x2b   = min(w, x_border + bw)

    if x2b - x1b < 20:
        return 0.0

    band_a = lab[yA:yB, x1b:x2b, 1].astype(np.float32)
    band_b = lab[yA:yB, x1b:x2b, 2].astype(np.float32)
    bm     = mask[yA:yB, x1b:x2b]

    band_a[bm == 0] = np.nan
    band_b[bm == 0] = np.nan

    ca = np.nanmean(band_a, axis=0)
    cb = np.nanmean(band_b, axis=0)
    ok = ~(np.isnan(ca) | np.isnan(cb))

    if ok.sum() < 6:
        return 0.0

    mid = ok.sum() // 2
    idx = np.where(ok)[0]
    l_idx = idx[:mid]
    r_idx = idx[mid:]

    da = abs(float(np.nanmean(ca[l_idx])) - float(np.nanmean(ca[r_idx])))
    db = abs(float(np.nanmean(cb[l_idx])) - float(np.nanmean(cb[r_idx])))

    return round(min(float(np.sqrt(da**2 + db**2)), 20.0), 1)


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
# DÉTECTER ORIENTATION
# =========================
def detect_car_orientation(car_crop):
    crop_h, crop_w = car_crop.shape[:2]
    log = []

    band_w  = int(crop_w * 0.22)
    feux_y1 = int(crop_h * 0.38)
    feux_y2 = int(crop_h * 0.88)

    left_feux  = car_crop[feux_y1:feux_y2, 0:band_w]
    right_feux = car_crop[feux_y1:feux_y2, crop_w - band_w:crop_w]
    left_hsv   = cv2.cvtColor(left_feux,  cv2.COLOR_BGR2HSV)
    right_hsv  = cv2.cvtColor(right_feux, cv2.COLOR_BGR2HSV)

    def count_red(hsv):
        m1 = cv2.inRange(hsv, (0,   60, 60), (12,  255, 255))
        m2 = cv2.inRange(hsv, (168, 60, 60), (180, 255, 255))
        return cv2.countNonZero(cv2.bitwise_or(m1, m2))

    red_left  = count_red(left_hsv)
    red_right = count_red(right_hsv)
    total_red = red_left + red_right

    if total_red > 150:
        if red_left > red_right * 1.35:
            log.append("P1 ROUGE → avant DROITE")
            return "right", log
        elif red_right > red_left * 1.35:
            log.append("P1 ROUGE → avant GAUCHE")
            return "left", log
        else:
            log.append("P1 equilibre, passe P2")
    else:
        log.append(f"P1 insuffisant ({total_red}px), passe P2")

    def count_headlight(hsv):
        white  = cv2.inRange(hsv, (0,  0,  170), (180, 90,  255))
        yellow = cv2.inRange(hsv, (15, 40, 170), (40,  220, 255))
        return cv2.countNonZero(cv2.bitwise_or(white, yellow))

    light_left  = count_headlight(left_hsv)
    light_right = count_headlight(right_hsv)
    total_light = light_left + light_right

    if total_light > 100:
        if light_left > light_right * 1.35:
            log.append("P2 PHARE → avant GAUCHE")
            return "left", log
        elif light_right > light_left * 1.35:
            log.append("P2 PHARE → avant DROITE")
            return "right", log
        else:
            log.append("P2 equilibre, passe P3")
    else:
        log.append("P2 insuffisant, passe P3")

    gray        = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
    vit_y1      = int(crop_h * 0.08)
    vit_y2      = int(crop_h * 0.58)
    vitre       = gray[vit_y1:vit_y2, :]
    dark        = (vitre < 90).astype(np.uint8)
    dark_f      = cv2.GaussianBlur(dark.astype(np.float32), (15, 15), 0)
    mid         = crop_w // 2
    left_glass  = float(dark_f[:, :mid].sum())
    right_glass = float(dark_f[:, mid:].sum())

    if left_glass > right_glass * 1.15:
        log.append("P3 VITRE → avant GAUCHE")
        return "left", log
    elif right_glass > left_glass * 1.15:
        log.append("P3 VITRE → avant DROITE")
        return "right", log

    left_band  = car_crop[feux_y1:feux_y2, 0:band_w]
    right_band = car_crop[feux_y1:feux_y2, crop_w - band_w:crop_w]
    lg = cv2.cvtColor(left_band,  cv2.COLOR_BGR2GRAY)
    rg = cv2.cvtColor(right_band, cv2.COLOR_BGR2GRAY)
    ls = float(np.mean(np.abs(cv2.Sobel(lg, cv2.CV_64F, 1, 1, ksize=3))))
    rs = float(np.mean(np.abs(cv2.Sobel(rg, cv2.CV_64F, 1, 1, ksize=3))))

    if rs > ls * 1.2:
        log.append("FALLBACK Sobel → avant GAUCHE")
        return "left", log
    else:
        log.append("FALLBACK Sobel → avant DROITE")
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

        orientation, orient_log = detect_car_orientation(car_crop)

        # ===============================================
        # PRÉPARER TOUS LES ESPACES + MASQUE
        # ===============================================
        mask_body, hsv_full, lab_full, gray_full = build_body_mask(car_crop)

        # Calculer LBP une seule fois sur toute l'image
        lbp_full = compute_lbp(gray_full)

        # ===============================================
        # RÉFÉRENCE GLOBALE (toute la carrosserie)
        # ===============================================
        ref_f = extract_zone_features(
            lab_full, hsv_full, gray_full, lbp_full,
            mask_body, 0, 0, crop_w, crop_h
        )
        if ref_f is None:
            return jsonify({"error": "No body pixels found"}), 400

        # ===============================================
        # 5 ZONES
        # ===============================================
        band_y1 = int(crop_h * 0.15)
        band_y2 = int(crop_h * 0.80)

        cut1 = int(crop_w * 0.20)
        cut2 = int(crop_w * 0.40)
        cut3 = int(crop_w * 0.60)
        cut4 = int(crop_w * 0.80)

        if orientation == "left":
            zone_names = [
                "Aile avant",
                "Porte avant",
                "Porte arriere",
                "Aile arriere",
                "Pare-chocs arr"
            ]
        else:
            zone_names = [
                "Pare-chocs arr",
                "Aile arriere",
                "Porte arriere",
                "Porte avant",
                "Aile avant"
            ]

        zones = [
            {"name": zone_names[0], "xA": 0,    "xB": cut1,   "yA": band_y1, "yB": band_y2},
            {"name": zone_names[1], "xA": cut1, "xB": cut2,   "yA": band_y1, "yB": band_y2},
            {"name": zone_names[2], "xA": cut2, "xB": cut3,   "yA": band_y1, "yB": band_y2},
            {"name": zone_names[3], "xA": cut3, "xB": cut4,   "yA": band_y1, "yB": band_y2},
            {"name": zone_names[4], "xA": cut4, "xB": crop_w, "yA": band_y1, "yB": band_y2},
        ]

        # ===============================================
        # DESSIN
        # ===============================================
        final_img      = img_orig.copy()
        thick_box      = max(3, int(5 * min(scale_x, scale_y)))
        thick_line     = max(1, int(1 * min(scale_x, scale_y)))
        font_scale_big = max(0.5, 0.55 * min(scale_x, scale_y))
        font_scale_med = max(0.4, 0.44 * min(scale_x, scale_y))
        font_thick     = max(2,   int(2  * min(scale_x, scale_y)))
        overlay_h      = max(55,  int(55 * scale_y))

        cv2.rectangle(final_img, (x1, y1), (x2, y2), (220, 220, 220), thick_line)

        step = max(10, int(12 * scale_y))
        dash = max(4,  int(6  * scale_y))
        for cut in [cut1, cut2, cut3, cut4]:
            for dy in range(band_y1, band_y2, step):
                cv2.line(
                    final_img,
                    (x1 + cut, y1 + dy),
                    (x1 + cut, y1 + dy + dash),
                    (255, 255, 255), thick_line
                )

        results_zones = []
        detected = 0

        for zone in zones:
            xA, xB = zone["xA"], zone["xB"]
            yA, yB = zone["yA"], zone["yB"]

            z_f = extract_zone_features(
                lab_full, hsv_full, gray_full, lbp_full,
                mask_body, xA, yA, xB, yB
            )

            abs_x1 = x1 + xA
            abs_y1 = y1 + yA
            abs_x2 = x1 + xB
            abs_y2 = y1 + yB

            if z_f is None:
                color_rect  = (150, 150, 150)
                label_score = "N/A"
                score_final = 0.0
                verdict     = "Non analysable"
                detail      = {}
            else:
                # Score multi-axes
                sc = compare_zone_to_ref(z_f, ref_f)

                # Cohérence interne 3 bandes
                bonus_coh, is_coh = check_coherence(
                    lab_full, hsv_full, gray_full, lbp_full,
                    mask_body, xA, yA, xB, yB, ref_f
                )

                # Raccord aux bords
                disc_l = detect_raccord(lab_full, mask_body, xA, yA, yB) if xA > 0      else 0.0
                disc_r = detect_raccord(lab_full, mask_body, xB, yA, yB) if xB < crop_w else 0.0
                bonus_raccord = round(min((disc_l + disc_r) / 2.0, 20.0), 1)

                score_final = round(sc["score"] + bonus_coh + bonus_raccord, 1)
                label_score = str(int(score_final))

                detail = {
                    "delta_couleur":   sc["delta_couleur"],
                    "delta_texture":   sc["delta_texture"],
                    "delta_h":         sc["delta_h"],
                    "delta_gradient":  sc["delta_grad"],
                    "delta_s":         sc["delta_s"],
                    "bonus_coherence": round(bonus_coh, 1),
                    "bonus_raccord":   bonus_raccord,
                    "zone_coherente":  is_coh,
                    "pixels_valides":  z_f["count"]
                }

                # Seuils de verdict
                if score_final < 18:
                    color_rect = (0, 210, 0)
                    verdict    = "OK - Peinture homogene"
                elif score_final < 35:
                    color_rect = (0, 165, 255)
                    verdict    = "Variation suspecte"
                    detected  += 1
                else:
                    color_rect = (0, 0, 255)
                    verdict    = "Peinture refaite!"
                    detected  += 1

            cv2.rectangle(final_img, (abs_x1, abs_y1),
                          (abs_x2, abs_y2), color_rect, thick_box)

            overlay = final_img.copy()
            cv2.rectangle(overlay, (abs_x1, abs_y1),
                          (abs_x2, abs_y1 + overlay_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, final_img, 0.5, 0, final_img)

            cv2.putText(final_img, zone["name"],
                        (abs_x1 + 6, abs_y1 + int(overlay_h * 0.40)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale_big, (255, 255, 255), font_thick)

            cv2.putText(final_img, f"Score: {label_score}",
                        (abs_x1 + 6, abs_y1 + int(overlay_h * 0.80)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale_med, color_rect, font_thick)

            cv2.putText(final_img, verdict,
                        (abs_x1 + 6, abs_y2 - int(10 * scale_y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale_med, color_rect, font_thick)

            results_zones.append({
                "zone":    zone["name"],
                "score":   score_final,
                "verdict": verdict,
                "detail":  detail
            })

        # Score global
        scores = [z["score"] for z in results_zones if z["score"] > 0]
        final_score = int(np.mean(scores)) if scores else 0
        final_score = min(final_score, 100)

        if final_score < 18:
            result = "Peinture homogene (OK)"
        elif final_score < 35:
            result = "Variations suspectes — verifier"
        else:
            result = "Repeinture probable detectee"

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
            "image_size":      {"width": orig_w, "height": orig_h},
            "reference": {
                "a":           round(ref_f["a_med"], 1),
                "b":           round(ref_f["b_med"], 1),
                "h":           round(ref_f["h_med"], 1),
                "s":           round(ref_f["s_med"], 1),
                "grad":        round(ref_f["grad_med"], 1),
                "pixels":      ref_f["count"]
            },
            "image_result":    analysed_name,
            "image_url":       request.host_url + "uploads/" + analysed_name
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
