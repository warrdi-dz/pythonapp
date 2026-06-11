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

def call_yolo(image_path):
    url = "https://warrdi.com/pytho/detect"
    try:
        ext  = os.path.splitext(image_path)[1].lower()
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
        with open(image_path, "rb") as f:
            files   = {"image": (os.path.basename(image_path), f, mime)}
            headers = {"Accept": "application/json"}
            r = requests.post(url, files=files, headers=headers, timeout=25)
        if r.status_code == 200:
            return r.json()
        return {"error": "YOLO failed", "status": r.status_code}
    except Exception as e:
        return {"error": "YOLO exception", "details": str(e)}

def call_car_make_model(image_path):
    url = "https://warrdi.com/pytho/car_make_model"
    try:
        ext  = os.path.splitext(image_path)[1].lower()
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
        with open(image_path, "rb") as f:
            files = {"image": (os.path.basename(image_path), f, mime)}
            r = requests.post(url, files=files, timeout=15)
        if r.status_code == 200:
            data = r.json()
            return {
                "make":       data.get("make",  "Unknown"),
                "model":      data.get("model", "Unknown"),
                "confidence": data.get("confidence", 0.0)
            }
    except Exception:
        pass
    return {"make": "Unknown", "model": "Unknown", "confidence": 0.0}

@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/")
def home():
    return jsonify({"status": "OK", "message": "GARAGE PRO V8 - comparaison voisins"})


# =============================================
# MASQUE CARROSSERIE STRICT
# Filtre par teinte pour exclure le fond coloré
# =============================================
def build_body_mask(car_crop, hsv_full):
    h_c, w_c = car_crop.shape[:2]

    # Masque de base
    mask_dark = cv2.inRange(hsv_full, (0, 0,   0), (180, 255,  50))
    mask_refl = cv2.inRange(hsv_full, (0, 0, 215), (180, 255, 255))
    mask_sky  = cv2.inRange(hsv_full, (0, 0, 210), (180,  20, 255))
    exclude   = cv2.bitwise_or(mask_dark, mask_refl)
    exclude   = cv2.bitwise_or(exclude,   mask_sky)
    mask_base = cv2.bitwise_not(exclude)

    k         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_base = cv2.morphologyEx(mask_base, cv2.MORPH_CLOSE, k, iterations=2)

    # Référence rapide sur le masque de base
    valid = hsv_full[mask_base > 0]
    if len(valid) < 100:
        return mask_base

    # Filtre teinte autour de la carrosserie
    ref_H = float(np.median(valid[:, 0]))
    ref_S = float(np.median(valid[:, 1]))
    ref_V = float(np.median(valid[:, 2]))

    if ref_S < 25:
        # Blanc / gris / argent → filtrer sur V
        diff_V     = np.abs(hsv_full[:, :, 2].astype(np.float32) - ref_V)
        mask_color = (diff_V < 50).astype(np.uint8) * 255
    else:
        # Voiture colorée → filtrer sur H ±28°
        H_ch       = hsv_full[:, :, 0].astype(np.float32)
        dH         = np.abs(H_ch - ref_H)
        dH         = np.minimum(dH, 180.0 - dH)
        mask_color = (dH < 28).astype(np.uint8) * 255

    mask_strict = cv2.bitwise_and(mask_base, mask_color)
    mask_strict = cv2.morphologyEx(mask_strict, cv2.MORPH_CLOSE, k, iterations=2)
    mask_strict = cv2.morphologyEx(mask_strict, cv2.MORPH_OPEN,  k, iterations=1)

    # Supprimer ombres géométriques
    mask_semi = cv2.inRange(hsv_full, (0, 0, 35), (180, 255, 120))
    msb       = cv2.bitwise_and(mask_semi, mask_strict)
    ks        = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    msb       = cv2.morphologyEx(msb, cv2.MORPH_CLOSE, ks, iterations=3)
    contours, _ = cv2.findContours(msb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_rm     = np.zeros_like(mask_strict)
    tba         = max(cv2.countNonZero(mask_strict), 1)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        rh = h / max(w, 1); rw = w / max(h, 1); ar = area / tba
        if ((rh > 3.0 and w < w_c*0.12) or
            (rw > 4.0 and ar > 0.04) or
            (y + h) > (h_c * 0.88) or
            (ar < 0.07 and rh < 1.6 and rw < 1.6)):
            cv2.drawContours(mask_rm, [cnt], -1, 255, -1)
    mask_strict = cv2.bitwise_and(mask_strict, cv2.bitwise_not(mask_rm))
    mask_strict = cv2.morphologyEx(mask_strict, cv2.MORPH_CLOSE, k, iterations=1)
    return mask_strict


# =============================================
# LUMINOSITÉ MÉDIANE D'UN POLYGONE
# On utilise uniquement V (luminosité) du HSV
# en excluant les 15% extrêmes (reflets, ombres)
# =============================================
def get_poly_luminosity(hsv_img, body_mask, polygon):
    """
    Retourne la luminosité médiane V et le nombre de pixels.
    C'est LA valeur utilisée pour toutes les comparaisons.
    """
    h, w      = hsv_img.shape[:2]
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [np.array(polygon, dtype=np.int32)], 255)
    combined  = cv2.bitwise_and(poly_mask, body_mask)
    valid     = hsv_img[combined > 0]

    if len(valid) < 80:
        return None, 0

    V     = valid[:, 2].astype(np.float32)
    p15   = np.percentile(V, 15)
    p85   = np.percentile(V, 85)
    keep  = (V >= p15) & (V <= p85)
    V_filtered = V[keep]

    if len(V_filtered) < 50:
        return None, 0

    return float(np.median(V_filtered)), len(V_filtered)


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
    col_sum = np.convolve(mask_valid.sum(axis=0).astype(float), np.ones(15)/15, mode='same')
    row_sum = np.convolve(mask_valid.sum(axis=1).astype(float), np.ones(15)/15, mode='same')
    vc = np.where(col_sum > col_sum.max()*0.08)[0]
    vr = np.where(row_sum > row_sum.max()*0.08)[0]
    if len(vc) < 20 or len(vr) < 20:
        return x1, y1, x2, y2
    PAD    = 8
    new_x1 = max(0,            x1 + int(vc[0])  - PAD)
    new_x2 = min(img.shape[1], x1 + int(vc[-1]) + PAD)
    new_y1 = max(0,            y1 + int(vr[0])  - PAD)
    new_y2 = min(img.shape[0], y1 + int(vr[-1]) + PAD)
    if (new_x2-new_x1) < 100 or (new_y2-new_y1) < 80:
        return x1, y1, x2, y2
    return new_x1, new_y1, new_x2, new_y2


def detect_lights(car_crop):
    h, w    = car_crop.shape[:2]
    band_w  = int(w * 0.28)
    fy1, fy2 = int(h*0.40), int(h*0.78)
    lf = cv2.cvtColor(car_crop[fy1:fy2, 0:band_w],   cv2.COLOR_BGR2HSV)
    rf = cv2.cvtColor(car_crop[fy1:fy2, w-band_w:w], cv2.COLOR_BGR2HSV)
    def cr(hsv):
        m1 = cv2.inRange(hsv,(0,  90,70),(12, 255,255))
        m2 = cv2.inRange(hsv,(165,90,70),(180,255,255))
        return int(cv2.countNonZero(cv2.bitwise_or(m1,m2)))
    def cw(hsv):
        w  = cv2.inRange(hsv,(0,0,210),   (180,60,255))
        y  = cv2.inRange(hsv,(18,130,190),(35,255,255))
        return int(cv2.countNonZero(cv2.bitwise_or(w,y)))
    rl=cr(lf); rr=cr(rf); wl=cw(lf); wr=cw(rf)
    ba = band_w*(fy2-fy1)
    return {"red_left":rl,"red_right":rr,"white_left":wl,"white_right":wr,
            "red_tot":rl+rr,"white_tot":wl+wr,"band_area":ba,
            "red_left_ratio":rl/max(ba,1),"red_right_ratio":rr/max(ba,1),
            "whi_left_ratio":wl/max(ba,1),"whi_right_ratio":wr/max(ba,1)}


def detect_front_rear(lights):
    log=[]
    rl,rr=lights["red_left"],lights["red_right"]
    wl,wr=lights["white_left"],lights["white_right"]
    rt=rl+rr; wt=wl+wr; ba=lights["band_area"]
    rt_thr=max(120,int(ba*0.003)); wt_thr=max(400,int(ba*0.020))
    if rt>=rt_thr:       facing="rear";  log.append(f"REAR rouge={rt}")
    elif wt>=wt_thr and wt>rt*3: facing="front"; log.append(f"FRONT blanc={wt}")
    else:                facing="side";  log.append(f"SIDE r={rt} w={wt}")
    rs = "left" if rl>rr*1.25 else ("right" if rr>rl*1.25 else None)
    fs = "left" if wl>wr*1.25 else ("right" if wr>wl*1.25 else None)
    if rs and not fs: fs = "right" if rs=="left" else "left"
    if fs and not rs: rs = "right" if fs=="left" else "left"
    if not rs and not fs: rs,fs="right","left"; log.append("Fallback")
    log.append(f"rear={rs} front={fs}")
    return rs, fs, facing, log


def estimate_angle(lights, crop_w, crop_h, facing):
    rl,rr=lights["red_left"],lights["red_right"]
    wl,wr=lights["white_left"],lights["white_right"]
    if   facing=="rear":  big=max(rl,rr); sml=min(rl,rr)
    elif facing=="front": big=max(wl,wr); sml=min(wl,wr)
    else:                 big=max(rl+wl,rr+wr); sml=min(rl+wl,rr+wr)
    if big==0: return 45.0
    sym=big/max(sml,1)
    if   sym>=8.0: a=10.0
    elif sym>=4.0: a=10.0+(8.0-sym)/4.0*30.0
    elif sym>=2.0: a=40.0+(4.0-sym)/2.0*25.0
    elif sym>=1.3: a=65.0+(2.0-sym)/0.7*20.0
    else:          a=87.0
    rw=crop_w/max(crop_h,1)
    if rw>1.6: a=min(a,35.0)
    elif rw<0.9: a=max(a,60.0)
    return round(a,1)


def make_poly(crop_w, crop_h, xA, xB, top_base, bot_base, persp, tilt_dir):
    drop=0.08
    def ty(x):
        f=x/max(1,crop_w)
        return top_base+(persp*drop*f if tilt_dir>0 else persp*drop*(1-f))
    def by(x):
        f=x/max(1,crop_w)
        return bot_base-(persp*drop*f if tilt_dir>0 else persp*drop*(1-f))
    return [(xA,int(ty(xA)*crop_h)),(xB,int(ty(xB)*crop_h)),
            (xB,int(by(xB)*crop_h)),(xA,int(by(xA)*crop_h))]


# =============================================
# ZONES avec noms normalisés pour la comparaison
# Les noms DOIVENT être dans cette liste exacte :
#   "Aile AV", "Porte AV", "Porte AR", "Aile AR"
#   "Capot", "Coffre", "Pare-ch.AV", "Pare-ch.AR"
# =============================================
def build_zones(crop_w, crop_h, angle, rear_side, front_side, facing, lights):
    rl,rr=lights["red_left"],lights["red_right"]
    wl,wr=lights["white_left"],lights["white_right"]
    if   facing=="rear":  ns="left" if rl>rr else "right"
    elif facing=="front": ns="left" if wl>wr else "right"
    else:                 ns="left" if (rl+wl)>(rr+wr) else "right"
    fs = "right" if ns=="left" else "left"
    if   angle<=10: persp=0.0
    elif angle<=55: persp=0.6*min(1.0,(angle-10)/45.0)
    elif angle<=80: persp=0.6*max(0.0,1.0-(angle-55)/25.0)
    else:           persp=0.0
    td = +1 if fs=="right" else -1
    tb, bb = 0.20, 0.88
    is_rear = (facing=="rear") or (facing=="side" and (rl+rr)>=(wl+wr))
    def z(name,a,b):
        return {"name":name,"poly":make_poly(crop_w,crop_h,
                int(a*crop_w),int(b*crop_w),tb,bb,persp,td)}
    if angle<=25:
        log=f"PROFIL({angle}°) near={ns}"
        if ns=="right":
            return ([z("Aile AV",0,0.20),z("Porte AV",0.20,0.48),
                     z("Porte AR",0.48,0.78),z("Aile AR",0.78,1)]
                    if is_rear else
                    [z("Aile AR",0,0.20),z("Porte AR",0.20,0.48),
                     z("Porte AV",0.48,0.78),z("Aile AV",0.78,1)]),log
        else:
            return ([z("Aile AR",0,0.22),z("Porte AR",0.22,0.52),
                     z("Porte AV",0.52,0.80),z("Aile AV",0.80,1)]
                    if is_rear else
                    [z("Aile AV",0,0.22),z("Porte AV",0.22,0.52),
                     z("Porte AR",0.52,0.80),z("Aile AR",0.80,1)]),log
    elif angle<=55:
        log=f"3/4 LEGER({angle}°) near={ns}"
        if ns=="right":
            return ([z("Porte AV",0,0.22),z("Porte AR",0.22,0.50),
                     z("Aile AR",0.50,0.72),z("Pare-ch.AR",0.72,1)]
                    if is_rear else
                    [z("Porte AR",0,0.22),z("Porte AV",0.22,0.50),
                     z("Aile AV",0.50,0.72),z("Pare-ch.AV",0.72,1)]),log
        else:
            return ([z("Pare-ch.AR",0,0.28),z("Aile AR",0.28,0.50),
                     z("Porte AR",0.50,0.78),z("Porte AV",0.78,1)]
                    if is_rear else
                    [z("Pare-ch.AV",0,0.28),z("Aile AV",0.28,0.50),
                     z("Porte AV",0.50,0.78),z("Porte AR",0.78,1)]),log
    elif angle<=80:
        log=f"3/4 MARQUE({angle}°) near={ns}"
        if ns=="right":
            return [z("Aile AR" if is_rear else "Aile AV",0.42,0.68),
                    z("Coffre" if is_rear else "Capot",0.68,0.85),
                    z("Pare-ch.AR" if is_rear else "Pare-ch.AV",0.85,1)],log
        else:
            return [z("Pare-ch.AR" if is_rear else "Pare-ch.AV",0,0.15),
                    z("Coffre" if is_rear else "Capot",0.15,0.32),
                    z("Aile AR" if is_rear else "Aile AV",0.32,0.58)],log
    else:
        log=f"FACE/DOS({angle}°)"
        if is_rear:
            return [z("Aile AR",0,0.20),z("Pare-ch.AR",0.20,0.55),
                    z("Coffre",0.45,0.80),z("Aile AR",0.80,1)],log
        else:
            return [z("Aile AV",0,0.20),z("Pare-ch.AV",0.20,0.55),
                    z("Capot",0.45,0.80),z("Aile AV",0.80,1)],log


# =============================================
# COMPARAISON PAR VOISINAGE
#
# PRINCIPE :
# Chaque pièce est comparée à ses voisines directes.
# Si une pièce est anormalement plus sombre OU plus
# claire que SES DEUX VOISINS → repeinture confirmée.
# Si elle l'est par rapport à UN seul voisin → suspect.
#
# Relations de voisinage (par nom) :
#   Aile AV  ↔  Porte AV
#   Porte AV ↔  Porte AR  (et Aile AV)
#   Porte AR ↔  Aile AR   (et Porte AV)
#   Aile AR  ↔  Porte AR
#   Capot    ↔  Aile AV   (si visible)
#   Coffre   ↔  Aile AR   (si visible)
#
# SEUILS :
#   diff_V > SEUIL_ROUGE  avec LES DEUX voisins → rouge
#   diff_V > SEUIL_ROUGE  avec UN seul voisin   → orange
#   diff_V > SEUIL_ORANGE avec UN seul voisin   → orange
#
# SEUIL calibré :
#   La luminosité V varie naturellement entre zones
#   à cause de la courbure et de l'éclairage.
#   On fixe SEUIL_ROUGE = 12 (variation naturelle ≈ 5-8)
#   et SEUIL_ORANGE = 7
#
# Adaptation selon voiture :
#   Voiture très sombre (V_med < 80) → seuils ×1.5
#   car les V sont naturellement proches de 0
#   et les différences relatives sont plus grandes.
# =============================================

# Voisins directs par nom de pièce
NEIGHBORS = {
    "Aile AV":    ["Porte AV", "Capot"],
    "Porte AV":   ["Aile AV",  "Porte AR"],
    "Porte AR":   ["Porte AV", "Aile AR"],
    "Aile AR":    ["Porte AR", "Coffre"],
    "Capot":      ["Aile AV",  "Pare-ch.AV"],
    "Coffre":     ["Aile AR",  "Pare-ch.AR"],
    "Pare-ch.AV": ["Capot",    "Aile AV"],
    "Pare-ch.AR": ["Coffre",   "Aile AR"],
}


def compare_with_neighbors(piece_name, piece_V, pieces_V_dict,
                            seuil_rouge, seuil_orange):
    """
    piece_name  : nom de la pièce à juger
    piece_V     : luminosité V médiane de cette pièce
    pieces_V_dict : {nom_piece: V_median} pour toutes les zones mesurées
    seuil_rouge, seuil_orange : seuils de détection

    Retourne : (verdict_state, detail_dict)
    verdict_state : "refaite" | "suspecte" | "ok"
    """
    neighbors = NEIGHBORS.get(piece_name, [])

    # Collecter les voisins disponibles dans ce scan
    voisins_disponibles = []
    for vname in neighbors:
        if vname in pieces_V_dict and pieces_V_dict[vname] is not None:
            voisins_disponibles.append((vname, pieces_V_dict[vname]))

    if not voisins_disponibles:
        # Pas de voisin disponible → on ne peut pas comparer
        return "ok", {"raison": "pas_de_voisin", "voisins": []}

    details = []
    deviations = []  # (voisin_name, diff_V, direction)

    for vname, vV in voisins_disponibles:
        diff = piece_V - vV   # positif = pièce plus claire que voisin
        # On regarde la direction : + = plus clair, - = plus sombre
        details.append({
            "voisin":    vname,
            "V_piece":   round(piece_V, 1),
            "V_voisin":  round(vV, 1),
            "diff":      round(diff, 1),
            "direction": "plus_clair" if diff > 0 else "plus_sombre"
        })
        deviations.append(diff)

    # Vérifier la cohérence directionnelle
    # Si tous les écarts sont dans la même direction ET dépassent le seuil
    all_positive = all(d > 0 for d in deviations)   # pièce plus claire que tous
    all_negative = all(d < 0 for d in deviations)   # pièce plus sombre que tous
    same_dir     = all_positive or all_negative

    abs_devs   = [abs(d) for d in deviations]
    max_dev    = max(abs_devs) if abs_devs else 0
    min_dev    = min(abs_devs) if abs_devs else 0
    count_over_rouge  = sum(1 for d in abs_devs if d >= seuil_rouge)
    count_over_orange = sum(1 for d in abs_devs if d >= seuil_orange)
    n_voisins  = len(voisins_disponibles)

    # ---- RÈGLE DE DÉCISION ----
    #
    # ROUGE si :
    #   - même direction ET tous les voisins dépassent seuil_rouge
    #   - même direction ET au moins 1 voisin dépasse seuil_rouge ET max > seuil_rouge*1.4
    #
    # ORANGE si :
    #   - même direction ET au moins 1 voisin dépasse seuil_rouge
    #   - même direction ET tous les voisins dépassent seuil_orange
    #   - 1 voisin dépasse seuil_rouge (même si direction mixte)
    #
    # VERT sinon

    verdict = "ok"

    if same_dir:
        if count_over_rouge == n_voisins:
            # Tous les voisins dépassent seuil_rouge → repeinture confirmée
            verdict = "refaite"
        elif count_over_rouge >= 1 and max_dev >= seuil_rouge * 1.3:
            # Au moins 1 dépasse clairement → rouge
            verdict = "refaite"
        elif count_over_orange >= 1 and max_dev >= seuil_orange:
            # Au moins 1 dépasse seuil orange → suspect
            verdict = "suspecte"
    else:
        # Directions mixtes = variation naturelle due à l'éclairage
        # Sauf si l'écart est très grand avec au moins un voisin
        if max_dev >= seuil_rouge * 1.5:
            verdict = "suspecte"

    return verdict, {
        "voisins":          details,
        "same_direction":   same_dir,
        "max_deviation":    round(max_dev, 1),
        "direction":        ("plus_clair" if all_positive else
                             "plus_sombre" if all_negative else "mixte")
    }


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
        detections  = yolo_result.get("detections", [])
        cars = [d for d in detections if d.get("class") == 2]
        if not cars:
            return jsonify({"error": "Car not detected"}), 400

        car_info = call_car_make_model(resized_path)
        if yolo_result.get("make"):
            car_info["make"]  = yolo_result.get("make",  car_info["make"])
        if yolo_result.get("model"):
            car_info["model"] = yolo_result.get("model", car_info["model"])

        scale_x = orig_w / YOLO_W
        scale_y = orig_h / YOLO_H

        raw_x1 = int(min(d["box"][0] for d in cars) * scale_x)
        raw_y1 = int(min(d["box"][1] for d in cars) * scale_y)
        raw_x2 = int(max(d["box"][2] for d in cars) * scale_x)
        raw_y2 = int(max(d["box"][3] for d in cars) * scale_y)

        pad_x=int(15*scale_x); pad_y=int(10*scale_y)
        thr_x=int(150*scale_x); thr_y=int(80*scale_y)
        x1=0      if raw_x1<thr_x           else max(0,     raw_x1-pad_x)
        x2=orig_w if (orig_w-raw_x2)<thr_x  else min(orig_w,raw_x2+pad_x)
        y1=0      if raw_y1<thr_y           else max(0,     raw_y1-pad_y)
        y2=orig_h if (orig_h-raw_y2)<thr_y  else min(orig_h,raw_y2+pad_y)

        x1,y1,x2,y2 = refine_car_bbox(img_orig,x1,y1,x2,y2)
        car_crop = img_orig[y1:y2, x1:x2]
        if car_crop.size == 0:
            return jsonify({"error": "invalid crop"}), 400
        crop_h, crop_w = car_crop.shape[:2]

        lights                              = detect_lights(car_crop)
        rear_side,front_side,facing,fr_log  = detect_front_rear(lights)
        angle                               = estimate_angle(lights,crop_w,crop_h,facing)
        fr_log.append(f"Angle={angle}°")

        zones, zone_dec = build_zones(crop_w,crop_h,angle,rear_side,front_side,facing,lights)
        fr_log.append(zone_dec)

        hsv_full  = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
        mask_body = build_body_mask(car_crop, hsv_full)

        # Vérifier le masque
        if cv2.countNonZero(mask_body) < 100:
            md = cv2.inRange(hsv_full,(0,0,0),(180,255,45))
            ms = cv2.inRange(hsv_full,(0,0,210),(180,18,255))
            mask_body = cv2.bitwise_not(cv2.bitwise_or(md,ms))
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            mask_body = cv2.morphologyEx(mask_body,cv2.MORPH_CLOSE,k)

        # Luminosité de référence globale pour les seuils
        all_valid = hsv_full[mask_body > 0]
        if len(all_valid) < 100:
            return jsonify({"error": "No body pixels found"}), 400
        global_V = float(np.median(all_valid[:, 2]))

        # =============================================
        # SEUILS ADAPTATIFS selon la luminosité
        # Voiture sombre → variations V plus comprimées
        # → seuils relatifs plus bas pour rester sensible
        # Voiture claire → variations V plus larges naturellement
        # → seuils un peu plus hauts
        # =============================================
        if global_V < 70:
            # Noir / bleu très foncé
            SEUIL_ROUGE  = 8.0
            SEUIL_ORANGE = 4.0
        elif global_V < 110:
            # Gris foncé / bordeaux
            SEUIL_ROUGE  = 10.0
            SEUIL_ORANGE = 5.5
        elif global_V < 160:
            # Gris / rouge / vert médium
            # Golf grise : Porte AR doit être ROUGE vs voisins
            SEUIL_ROUGE  = 12.0
            SEUIL_ORANGE = 6.0
        else:
            # Blanc / beige / argent / jaune clair
            SEUIL_ROUGE  = 15.0
            SEUIL_ORANGE = 8.0

        # =============================================
        # ÉTAPE 1 : Mesurer la luminosité de chaque zone
        # =============================================
        pieces_V = {}  # {nom_piece: V_median}
        zone_raw = {}  # {nom_piece: (V, n_pixels, poly_local)}

        for zone in zones:
            name       = zone["name"]
            poly_local = zone["poly"]
            V_med, n   = get_poly_luminosity(hsv_full, mask_body, poly_local)
            # Gérer les doublons de noms (ex: 2x "Aile AR" vue dos)
            if name in pieces_V:
                name = name + "_2"
            pieces_V[name]  = V_med
            zone_raw[name]  = (V_med, n, poly_local)

        # =============================================
        # ÉTAPE 2 : Comparer chaque pièce avec ses voisins
        # =============================================
        zone_verdicts = {}  # {nom: (verdict_state, score, detail)}

        for name, (V_med, n, _) in zone_raw.items():
            base_name = name.replace("_2", "")
            if V_med is None or n < 80:
                zone_verdicts[name] = ("ok", 0.0, {"raison": "insuf_pixels"})
                continue
            verdict_state, detail = compare_with_neighbors(
                base_name, V_med, pieces_V,
                SEUIL_ROUGE, SEUIL_ORANGE
            )
            max_dev = detail.get("max_deviation", 0.0)
            zone_verdicts[name] = (verdict_state, max_dev, detail)

        # =============================================
        # ÉTAPE 3 : Dessin
        # =============================================
        final_img  = img_orig.copy()
        thick_box  = max(3, int(4*min(scale_x,scale_y)))
        thick_line = max(1, int(1*min(scale_x,scale_y)))
        font_big   = max(0.55, 0.58*min(scale_x,scale_y))
        font_med   = max(0.42, 0.44*min(scale_x,scale_y))
        font_thick = max(2, int(2*min(scale_x,scale_y)))

        cv2.rectangle(final_img,(x1,y1),(x2,y2),(220,220,220),thick_line)

        header = (f"{car_info['make']} {car_info['model']} | "
                  f"{'AR' if facing=='rear' else ('AV' if facing=='front' else 'COTE')} | "
                  f"{angle}° | V={global_V:.0f}")
        (hw,hh),_ = cv2.getTextSize(header,cv2.FONT_HERSHEY_SIMPLEX,font_med*1.1,font_thick)
        cv2.rectangle(final_img,(5,5),(15+hw,20+hh),(0,0,0),-1)
        cv2.putText(final_img,header,(10,15+hh),cv2.FONT_HERSHEY_SIMPLEX,
                    font_med*1.1,(255,255,255),font_thick)

        results_zones = []
        detected      = 0
        zone_names_ordered = list(zone_raw.keys())

        for idx, name in enumerate(zone_names_ordered, start=1):
            V_med, n, poly_local = zone_raw[name]
            poly_global = np.array(
                [[x1+p[0], y1+p[1]] for p in poly_local], dtype=np.int32
            )
            verdict_state, max_dev, detail = zone_verdicts[name]

            if V_med is None:
                color_rect = (150,150,150)
                verdict    = "Non analysable"
                label      = "N/A"
            else:
                if   verdict_state == "refaite":
                    color_rect = (0,0,255)
                    verdict    = "Peinture refaite!"
                    detected  += 1
                elif verdict_state == "suspecte":
                    color_rect = (0,165,255)
                    verdict    = "Variation suspecte"
                    detected  += 1
                else:
                    color_rect = (0,210,0)
                    verdict    = "OK"

                direction = detail.get("direction","?")
                dir_sym   = "↑" if direction=="plus_clair" else ("↓" if direction=="plus_sombre" else "~")
                label     = f"V:{V_med:.0f} {dir_sym}dV:{max_dev:.0f}"

            # Remplissage + contour
            overlay = final_img.copy()
            cv2.fillPoly(overlay,[poly_global],color_rect)
            cv2.addWeighted(overlay,0.22,final_img,0.78,0,final_img)
            cv2.polylines(final_img,[poly_global],True,color_rect,thick_box)

            # Cercle numéroté
            cx = int(np.mean(poly_global[:,0]))
            cy = int(np.mean(poly_global[:,1]))
            r  = max(18,int(20*min(scale_x,scale_y)))
            cv2.circle(final_img,(cx+2,cy+2),r,(0,0,0),-1)
            cv2.circle(final_img,(cx,cy),r,color_rect,-1)
            cv2.circle(final_img,(cx,cy),r,(255,255,255),2)
            ntxt=str(idx)
            (tw,th),_=cv2.getTextSize(ntxt,cv2.FONT_HERSHEY_SIMPLEX,font_big*1.3,font_thick+1)
            cv2.putText(final_img,ntxt,(cx-tw//2,cy+th//2),
                        cv2.FONT_HERSHEY_SIMPLEX,font_big*1.3,(255,255,255),font_thick+1)

            # Étiquette
            top_pt = poly_global[poly_global[:,1].argmin()]
            lx     = max(5,int(top_pt[0]))
            ly     = max(20,int(top_pt[1])-10)
            lbl    = f"{idx}. {name.replace('_2','')}  {label}"
            (lw,lh),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,font_med,font_thick)
            lx = min(lx, orig_w-lw-10)
            cv2.rectangle(final_img,(lx-4,ly-lh-6),(lx+lw+6,ly+4),(0,0,0),-1)
            cv2.putText(final_img,lbl,(lx,ly),
                        cv2.FONT_HERSHEY_SIMPLEX,font_med,(255,255,255),font_thick)

            results_zones.append({
                "idx":         idx,
                "zone":        name.replace("_2",""),
                "V_median":    round(V_med,1) if V_med else None,
                "max_deviation": round(max_dev,1),
                "direction":   detail.get("direction","?"),
                "voisins":     detail.get("voisins",[]),
                "pixels":      n,
                "verdict":     verdict
            })

        # Score global
        devs = [z["max_deviation"] for z in results_zones if z["max_deviation"] > 0]
        final_score = min(int(np.mean(devs)*3) if devs else 0, 100)
        if   detected >= 2: result = "Difference importante - repeinture probable"
        elif detected == 1: result = "Variation detectee - verification recommandee"
        else:               result = "Peinture homogene (OK)"

        analysed_name = "analysed_" + filename
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, analysed_name), final_img)
        if os.path.exists(resized_path):
            os.remove(resized_path)

        return jsonify({
            "yolo":            yolo_result,
            "car":             car_info,
            "angle_estime":    angle,
            "score":           final_score,
            "result":          result,
            "zones":           results_zones,
            "zones_detected":  detected,
            "facing":          facing,
            "rear_side":       rear_side,
            "front_side":      front_side,
            "seuils":          {"rouge": SEUIL_ROUGE, "orange": SEUIL_ORANGE, "V_global": round(global_V,1)},
            "orientation_log": fr_log,
            "lights":          lights,
            "image_size":      {"width": orig_w, "height": orig_h},
            "image_result":    analysed_name,
            "image_url":       request.host_url + "uploads/" + analysed_name
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
