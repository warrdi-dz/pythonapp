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
        is_pole    = (ratio_hw > 3.0) and (w < w_c * 0.12)
        is_long    = (ratio_wh > 4.0) and (area_ratio > 0.04)
        touch_bot  = (y + h) > (h_c * 0.88)
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
# ESTIMER L'ANGLE DE PRISE DE VUE
#
# L'angle est estimé par le rapport entre
# les feux/phares visibles d'un côté vs l'autre
# et par le ratio largeur/hauteur du crop.
#
# angle ≈ 0°  = vue de côté (profil)
# angle ≈ 90° = vue de face ou de derrière
#
# On mesure l'asymétrie des éléments visibles :
# - feux arrière rouges : présence et taille
# - phares avant blancs : présence et taille
# - vitres latérales : longueur = indique vue côté
#
# Retourne :
#   angle_deg    : 0-90 estimé
#   side_visible : "right" ou "left" (côté arrière visible)
#   red_R, red_L : pixels rouges droite/gauche
#   whi_R, whi_L : pixels blancs droite/gauche
# =============================================
def estimate_view_angle(car_crop):
    crop_h, crop_w = car_crop.shape[:2]
    hsv  = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)

    # --- Feux arrière rouges ---
    mr1      = cv2.inRange(hsv, (0,   60, 60), (12,  255, 255))
    mr2      = cv2.inRange(hsv, (168, 60, 60), (180, 255, 255))
    mask_red = cv2.bitwise_or(mr1, mr2)
    red_L    = cv2.countNonZero(mask_red[:, :crop_w//2])
    red_R    = cv2.countNonZero(mask_red[:, crop_w//2:])
    red_tot  = red_L + red_R

    # --- Phares avant blancs/jaunes (bande basse) ---
    mw1       = cv2.inRange(hsv, (0,  0,  195), (180, 50, 255))
    mw2       = cv2.inRange(hsv, (15, 40, 195), (40, 180, 255))
    ph        = cv2.bitwise_or(mw1, mw2)[int(crop_h*0.45):, :]
    whi_L     = cv2.countNonZero(ph[:, :crop_w//2])
    whi_R     = cv2.countNonZero(ph[:, crop_w//2:])
    whi_tot   = whi_L + whi_R

    # --- Vitres latérales (zones sombres hautes) ---
    top       = gray[int(crop_h*0.05):int(crop_h*0.55), :]
    dk        = (top < 75).astype(np.uint8)
    dk_f      = cv2.GaussianBlur(dk.astype(np.float32), (15, 15), 0)
    glass_L   = float(dk_f[:, :crop_w//2].sum())
    glass_R   = float(dk_f[:, crop_w//2:].sum())
    glass_tot = glass_L + glass_R

    ratio_wh  = crop_w / max(crop_h, 1)

    # -----------------------------------------------
    # ESTIMATION DE L'ANGLE
    #
    # Logique :
    # - Vitres longues des deux côtés + ratio > 1.4
    #   → vue de côté → angle ~ 0-30°
    #
    # - Un seul côté a des feux ou phares très dominants
    #   ET l'autre côté a aussi des éléments mais moins
    #   → angle intermédiaire 30-60°
    #
    # - Les deux côtés ont des feux/phares équilibrés
    #   ET ratio ~ 1 (carré)
    #   → vue de face/derrière → angle 80-90°
    #
    # L'asymétrie des feux mesure l'angle :
    # asymétrie = max/min des pixels rouges ou blancs
    # asymétrie élevée = angle faible (on voit surtout un côté)
    # asymétrie faible = angle élevé (les deux côtés visibles)
    # -----------------------------------------------

    # Asymétrie rouge (feux arrière)
    if red_tot > 200:
        max_red  = max(red_L, red_R)
        min_red  = max(min(red_L, red_R), 1)
        asym_red = max_red / min_red
    else:
        asym_red = 1.0

    # Asymétrie phares avant
    if whi_tot > 150:
        max_whi  = max(whi_L, whi_R)
        min_whi  = max(min(whi_L, whi_R), 1)
        asym_whi = max_whi / min_whi
    else:
        asym_whi = 1.0

    # Asymétrie vitres
    if glass_tot > 2000:
        max_gl  = max(glass_L, glass_R)
        min_gl  = max(min(glass_L, glass_R), 1)
        asym_gl = max_gl / min_gl
    else:
        asym_gl = 1.0

    # Score global d'asymétrie (plus haut = plus de côté)
    asym_score = max(asym_red, asym_whi, asym_gl)

    # Conversion asymétrie → angle
    # asym > 5.0 → très de côté → 0-20°
    # asym 3-5   → 3/4 proche du côté → 20-45°
    # asym 1.5-3 → 3/4 → 45-70°
    # asym < 1.5 → face/derrière → 70-90°
    if asym_score > 5.0:
        angle_deg = 15
    elif asym_score > 3.0:
        angle_deg = 32
    elif asym_score > 1.8:
        angle_deg = 55
    elif asym_score > 1.3:
        angle_deg = 72
    else:
        angle_deg = 85

    # Correction par ratio largeur/hauteur
    # Vue de côté → ratio élevé
    if ratio_wh > 1.6 and glass_tot > 5000:
        angle_deg = min(angle_deg, 25)
    elif ratio_wh < 1.1:
        angle_deg = max(angle_deg, 75)

    # Déterminer quel côté arrière est visible
    # (côté avec le plus de rouge = côté arrière visible)
    if red_tot > 200:
        side_rear = "right" if red_R > red_L else "left"
    elif whi_tot > 150:
        # Côté avec phares = côté avant visible
        # Donc côté opposé = côté arrière
        side_rear = "left" if whi_R > whi_L else "right"
    else:
        side_rear = "right"  # défaut

    return {
        "angle_deg":  angle_deg,
        "side_rear":  side_rear,
        "asym_score": round(asym_score, 2),
        "red_L": red_L, "red_R": red_R, "red_tot": red_tot,
        "whi_L": whi_L, "whi_R": whi_R, "whi_tot": whi_tot,
        "glass_L": int(glass_L), "glass_R": int(glass_R),
        "ratio_wh": round(ratio_wh, 2)
    }


# =============================================
# DÉFINIR LES ZONES AVEC TAILLES PROPORTIONNELLES
# À L'ANGLE DE VUE
#
# Principe :
# angle 0-30° (côté)   : 4 zones égales
# angle 30-60° (3/4)   : zones proportionnelles
#                         côté arrière visible = grand
#                         côté avant = petit
# angle 60-80° (3/4 fort) : 3 zones (parechoc, aile, coffre)
#                            côté opposé non scannable
# angle 80-90° (face/dos) : 1-2 zones selon feux visibles
#
# Les zones sont en pixels absolus dans le crop.
# MIN_PIX = seuil minimum de pixels carrosserie valides.
# =============================================
def define_angle_zones(angle_info, crop_h, crop_w, mask_body):
    """
    Crée les zones en fonction de l'angle estimé.
    Chaque zone a une taille proportionnelle à sa visibilité.
    Seules les zones avec assez de pixels valides sont créées.
    """
    MIN_PIX   = 250
    angle_deg = angle_info["angle_deg"]
    side_rear = angle_info["side_rear"]   # côté où est l'arrière
    red_R     = angle_info["red_R"]
    red_L     = angle_info["red_L"]
    whi_R     = angle_info["whi_R"]
    whi_L     = angle_info["whi_L"]

    # side_front = côté opposé à l'arrière
    side_front = "left" if side_rear == "right" else "right"

    y_top = int(crop_h * 0.08)
    y_bot = int(crop_h * 0.90)

    def valid_zone(xA, yA, xB, yB):
        zm = mask_body[yA:yB, xA:xB]
        return cv2.countNonZero(zm) >= MIN_PIX

    def make_zone(name, xA, xB, yA=None, yB=None):
        yA = yA if yA is not None else y_top
        yB = yB if yB is not None else y_bot
        if valid_zone(xA, yA, xB, yB):
            return {"name": name, "xA": xA, "xB": xB, "yA": yA, "yB": yB}
        return None

    zones     = []
    log_angle = []

    # -----------------------------------------------
    # ANGLE 0-30° : VUE DE CÔTÉ QUASI-PARFAITE
    # 4 zones de tailles égales
    # Aile avant | Porte avant | Porte arrière | Aile arr.
    # -----------------------------------------------
    if angle_deg <= 30:
        log_angle.append(f"angle={angle_deg}° → VUE COTE (4 zones egales)")
        c1 = int(crop_w * 0.20)
        c2 = int(crop_w * 0.45)
        c3 = int(crop_w * 0.70)

        if side_rear == "left":
            pieces = [
                ("Aile arriere",  0,  c1),
                ("Porte arriere", c1, c2),
                ("Porte avant",   c2, c3),
                ("Aile avant",    c3, crop_w),
            ]
        else:
            pieces = [
                ("Aile avant",    0,  c1),
                ("Porte avant",   c1, c2),
                ("Porte arriere", c2, c3),
                ("Aile arriere",  c3, crop_w),
            ]

        for name, xA, xB in pieces:
            z = make_zone(name, xA, xB)
            if z:
                zones.append(z)

    # -----------------------------------------------
    # ANGLE 30-60° : VUE 3/4 PROCHE DU CÔTÉ
    #
    # Côté arrière visible (grand) :
    #   - Pare-chocs arr. (grand) → 25% de la largeur
    #   - Aile arrière (grand)    → 20%
    #   - Porte arrière (moyen)   → 20%
    # Côté avant (réduit) :
    #   - Porte avant (moyen)     → 20%
    #   - Aile avant (petit)      → 15%
    # -----------------------------------------------
    elif angle_deg <= 60:
        log_angle.append(f"angle={angle_deg}° → VUE 3/4 (zones proportionnelles)")

        # Proportions selon visibilité
        # côté arrière = 65% de la largeur
        # côté avant   = 35%
        if side_rear == "right":
            # Arrière à droite
            c_pa = int(crop_w * 0.15)   # fin aile avant (petit)
            c_pav= int(crop_w * 0.35)   # fin porte avant
            c_par= int(crop_w * 0.55)   # fin porte arrière
            c_aa = int(crop_w * 0.75)   # fin aile arrière
            # Pare-chocs arr. = dernier tiers à droite
            pieces = [
                ("Aile avant",    0,    c_pa,  True),
                ("Porte avant",   c_pa, c_pav, True),
                ("Porte arriere", c_pav,c_par, True),
                ("Aile arriere",  c_par,c_aa,  True),
                ("Pare-chocs ar.",c_aa, crop_w,True),
            ]
        else:
            # Arrière à gauche
            c_pc = int(crop_w * 0.25)   # fin pare-chocs arr.
            c_aa = int(crop_w * 0.45)   # fin aile arrière
            c_par= int(crop_w * 0.65)   # fin porte arrière
            c_pav= int(crop_w * 0.85)   # fin porte avant
            pieces = [
                ("Pare-chocs ar.",0,    c_pc,  True),
                ("Aile arriere",  c_pc, c_aa,  True),
                ("Porte arriere", c_aa, c_par, True),
                ("Porte avant",   c_par,c_pav, True),
                ("Aile avant",    c_pav,crop_w,True),
            ]

        for item in pieces:
            name, xA, xB, _ = item
            z = make_zone(name, xA, xB)
            if z:
                zones.append(z)

    # -----------------------------------------------
    # ANGLE 60-80° : VUE 3/4 FORTE
    #
    # Côté arrière (très visible) :
    #   - Pare-chocs arr. (très grand) → 35%
    #   - Coffre/hayon    (grand)       → 25%
    #   - Aile arrière    (grand)       → 25%
    # Côté avant (peu visible) :
    #   - Porte avant (petit)           → 15%
    # L'aile avant du côté opposé n'est pas scannable.
    # -----------------------------------------------
    elif angle_deg <= 80:
        log_angle.append(f"angle={angle_deg}° → VUE 3/4 FORTE (3-4 zones arriere)")

        co_y1 = int(crop_h * 0.05)
        co_y2 = int(crop_h * 0.52)
        pc_y1 = int(crop_h * 0.55)
        pc_y2 = int(crop_h * 0.93)

        if side_rear == "right":
            # Arrière fortement visible à droite
            c_pav = int(crop_w * 0.18)   # porte avant (petit, gauche)
            c_aa  = int(crop_w * 0.42)   # aile arrière
            c_co  = int(crop_w * 0.68)   # coffre/hayon

            zones_candidates = [
                make_zone("Porte avant",    0,     c_pav),
                make_zone("Aile arriere",   c_pav, c_aa),
                make_zone("Coffre / hayon", c_aa,  c_co,  co_y1, co_y2),
                make_zone("Pare-chocs ar.", c_aa,  crop_w,pc_y1, pc_y2),
            ]
        else:
            # Arrière fortement visible à gauche
            c_co  = int(crop_w * 0.32)   # fin coffre
            c_aa  = int(crop_w * 0.58)   # fin aile arrière
            c_pav = int(crop_w * 0.82)   # fin porte avant (petit)

            zones_candidates = [
                make_zone("Pare-chocs ar.", 0,     c_co,  pc_y1, pc_y2),
                make_zone("Coffre / hayon", 0,     c_co,  co_y1, co_y2),
                make_zone("Aile arriere",   c_co,  c_aa),
                make_zone("Porte avant",    c_aa,  c_pav),
            ]

        for z in zones_candidates:
            if z:
                zones.append(z)

    # -----------------------------------------------
    # ANGLE 80-90° : VUE DE FACE OU DE DERRIÈRE
    #
    # Si feux rouges dominants → vue arrière :
    #   Pare-chocs arr. (bas) + Coffre (haut) + 2 ailes
    #
    # Si phares blancs dominants → vue avant :
    #   Pare-chocs av. (bas) + Capot (haut) + 2 ailes
    #
    # Si les deux visibles → vue 3/4 face-côté :
    #   2 portes + 2 ailes (côté de chaque)
    # -----------------------------------------------
    else:
        log_angle.append(f"angle={angle_deg}° → VUE FACE/DOS")

        co_y1 = int(crop_h * 0.05)
        co_y2 = int(crop_h * 0.52)
        pc_y1 = int(crop_h * 0.55)
        pc_y2 = int(crop_h * 0.93)
        c1    = int(crop_w * 0.22)
        c2    = int(crop_w * 0.78)

        has_rear  = angle_info["red_tot"] > 200
        has_front = angle_info["whi_tot"] > 150
        both      = has_rear and has_front

        if both:
            # Vue 3/4 avec les deux côtés
            log_angle.append("face+dos → 2 portes + 2 ailes")
            zones_candidates = [
                make_zone("Aile gauche",   0,  c1),
                make_zone("Porte gauche",  0,  c1,  co_y1, co_y2),
                make_zone("Capot / Coffre",c1, c2,  co_y1, co_y2),
                make_zone("Pare-chocs",    c1, c2,  pc_y1, pc_y2),
                make_zone("Porte droite",  c2, crop_w, co_y1, co_y2),
                make_zone("Aile droite",   c2, crop_w),
            ]
        elif has_rear:
            log_angle.append("vue arriere → coffre + parechoc + 2 ailes arr.")
            zones_candidates = [
                make_zone("Aile arr. G",   0,  c1),
                make_zone("Coffre / hayon",c1, c2,  co_y1, co_y2),
                make_zone("Pare-chocs ar.",c1, c2,  pc_y1, pc_y2),
                make_zone("Aile arr. D",   c2, crop_w),
            ]
        else:
            # Vue avant par défaut
            log_angle.append("vue avant → capot + parechoc + 2 ailes av.")
            zones_candidates = [
                make_zone("Aile av. G",    0,  c1),
                make_zone("Capot avant",   c1, c2,  co_y1, co_y2),
                make_zone("Pare-chocs av.",c1, c2,  pc_y1, pc_y2),
                make_zone("Aile av. D",    c2, crop_w),
            ]

        for z in zones_candidates:
            if z:
                zones.append(z)

    # Fallback si aucune zone créée
    if not zones:
        log_angle.append("FALLBACK: 3 zones egales")
        c1 = int(crop_w * 0.33)
        c2 = int(crop_w * 0.67)
        for name, xA, xB in [
            ("Zone gauche",  0,  c1),
            ("Zone centre",  c1, c2),
            ("Zone droite",  c2, crop_w)
        ]:
            z = make_zone(name, xA, xB)
            if z:
                zones.append(z)

    return zones, log_angle


# =============================================
# DÉTECTER COULEUR DOMINANTE
# =============================================
def detect_car_color(car_crop, hsv_full, mask_body):
    lab   = cv2.cvtColor(car_crop, cv2.COLOR_BGR2LAB)
    vl    = lab[mask_body > 0]
    if len(vl) < 50:
        return {"couleur": "Inconnue", "teinte_hex": "#888888"}

    L    = vl[:, 0]
    p10  = np.percentile(L, 10)
    p90  = np.percentile(L, 90)
    keep = (L >= p10) & (L <= p90)
    vl   = vl[keep]
    if len(vl) < 20:
        return {"couleur": "Inconnue", "teinte_hex": "#888888"}

    mL = float(np.median(vl[:, 0]))
    ma = float(np.median(vl[:, 1]))
    mb = float(np.median(vl[:, 2]))

    lab_px  = np.uint8([[[mL, ma, mb]]])
    bgr_px  = cv2.cvtColor(lab_px, cv2.COLOR_LAB2BGR)
    r, g, b = int(bgr_px[0,0,2]), int(bgr_px[0,0,1]), int(bgr_px[0,0,0])
    hex_col = f"#{r:02X}{g:02X}{b:02X}"

    hsv_v   = hsv_full[mask_body > 0]
    hsv_v   = hsv_v[(hsv_v[:,2] >= p10) & (hsv_v[:,2] <= p90)]
    if len(hsv_v) == 0:
        return {"couleur": "Inconnue", "teinte_hex": hex_col}

    mH = float(np.median(hsv_v[:, 0]))
    mS = float(np.median(hsv_v[:, 1]))
    mV = float(np.median(hsv_v[:, 2]))

    if mS < 25:
        if   mV < 55:  couleur = "Noir"
        elif mV < 120: couleur = "Gris fonce"
        elif mV < 165: couleur = "Gris"
        elif mV < 205: couleur = "Gris clair"
        else:          couleur = "Blanc"
    elif mS < 60:
        if   mV < 80:  couleur = "Noir metallise"
        elif mV < 140: couleur = "Gris metallise"
        else:          couleur = "Argente"
    else:
        if   mH < 10 or mH > 170: couleur = "Rouge"
        elif mH < 20:  couleur = "Orange"
        elif mH < 35:  couleur = "Jaune"
        elif mH < 85:  couleur = "Vert"
        elif mH < 130: couleur = "Bleu"
        elif mH < 150: couleur = "Violet"
        else:          couleur = "Rose"
        if mV < 80:    couleur += " fonce"
        elif mS > 150 and mV > 150: couleur += " vif"

    return {"couleur": couleur, "teinte_hex": hex_col,
            "HSV": {"H": round(mH,1), "S": round(mS,1), "V": round(mV,1)}}


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
        # ESPACES COLORIMÉTRIQUES + MASQUE
        # ===============================================
        hsv_full  = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
        lab_full  = cv2.cvtColor(car_crop, cv2.COLOR_BGR2LAB)
        gray_full = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
        mask_body = build_body_mask(car_crop, hsv_full)

        # ===============================================
        # COULEUR + ANGLE + ZONES
        # ===============================================
        car_color  = detect_car_color(car_crop, hsv_full, mask_body)
        angle_info = estimate_view_angle(car_crop)
        zones, angle_log = define_angle_zones(
            angle_info, crop_h, crop_w, mask_body
        )

        # ===============================================
        # RÉFÉRENCE GLOBALE LAB
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
        font_scale_med = max(0.38, 0.40 * min(scale_x, scale_y))
        font_thick     = max(2,    int(2  * min(scale_x, scale_y)))
        overlay_h      = max(50,   int(50 * scale_y))

        cv2.rectangle(final_img, (x1, y1), (x2, y2), (220, 220, 220), thick_line)

        results_zones = []
        detected      = 0

        for i, zone in enumerate(zones):
            xA, xB = zone["xA"], zone["xB"]
            yA, yB = zone["yA"], zone["yB"]

            zone_color, px_count = get_zone_color(
                lab_full, mask_body, xA, yA, xB, yB
            )

            # Texture sur masque
            zm_zone  = mask_body[yA:yB, xA:xB]
            gz_zone  = gray_full[yA:yB, xA:xB]
            lap_z    = cv2.Laplacian(gz_zone, cv2.CV_64F)
            lap_v    = lap_z[zm_zone > 0]
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
                diff       = (color_diff * 0.85) + (tex_diff * 0.15)
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

            cv2.rectangle(final_img, (abs_x1, abs_y1),
                          (abs_x2, abs_y2), color_rect, thick_box)

            # Séparateur avec la zone suivante
            if i < len(zones) - 1:
                step = max(10, int(12 * scale_y))
                dash = max(4,  int(6  * scale_y))
                for dy in range(abs_y1, abs_y2, step):
                    cv2.line(final_img, (abs_x2, dy),
                             (abs_x2, dy + dash),
                             (255, 255, 255), thick_line)

            overlay = final_img.copy()
            cv2.rectangle(overlay, (abs_x1, abs_y1),
                          (abs_x2, abs_y1 + overlay_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.50, final_img, 0.50, 0, final_img)

            cv2.putText(final_img, zone["name"],
                        (abs_x1 + 5, abs_y1 + int(overlay_h * 0.42)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale_big, (255, 255, 255), font_thick)

            cv2.putText(final_img, f"{label_score} {verdict}",
                        (abs_x1 + 5, abs_y1 + int(overlay_h * 0.85)),
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
            "vehicule":       car_color,
            "angle":          angle_info,
            "angle_log":      angle_log,
            "zones":          results_zones,
            "zones_detected": detected,
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
