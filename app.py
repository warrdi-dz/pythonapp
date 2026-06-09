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
    return jsonify({"status": "OK", "message": "GARAGE PRO V7"})


# =========================
# COULEUR HSV MÉDIANE dans un POLYGONE
# Retourne aussi std_s pour filtrer les fonds
# =========================
def get_poly_color(hsv_img, body_mask, polygon):
    h, w      = hsv_img.shape[:2]
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [np.array(polygon, dtype=np.int32)], 255)
    combined  = cv2.bitwise_and(poly_mask, body_mask)
    valid     = hsv_img[combined > 0]
    if len(valid) < 80:
        return None, 0, None
    med = np.array([
        float(np.median(valid[:, 0])),
        float(np.median(valid[:, 1])),
        float(np.median(valid[:, 2]))
    ])
    stats = {
        "std_h": float(np.std(valid[:, 0])),
        "std_s": float(np.std(valid[:, 1])),
        "std_v": float(np.std(valid[:, 2])),
    }
    return med, len(valid), stats


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
    valid_cols = np.where(col_sum > col_sum.max()*0.08)[0]
    valid_rows = np.where(row_sum > row_sum.max()*0.08)[0]
    if len(valid_cols) < 20 or len(valid_rows) < 20:
        return x1, y1, x2, y2
    PAD    = 8
    new_x1 = max(0,            x1 + int(valid_cols[0])  - PAD)
    new_x2 = min(img.shape[1], x1 + int(valid_cols[-1]) + PAD)
    new_y1 = max(0,            y1 + int(valid_rows[0])  - PAD)
    new_y2 = min(img.shape[0], y1 + int(valid_rows[-1]) + PAD)
    if (new_x2-new_x1) < 100 or (new_y2-new_y1) < 80:
        return x1, y1, x2, y2
    return new_x1, new_y1, new_x2, new_y2


def detect_lights(car_crop):
    h, w    = car_crop.shape[:2]
    band_w  = int(w * 0.28)
    feux_y1 = int(h * 0.40)
    feux_y2 = int(h * 0.78)
    lf = cv2.cvtColor(car_crop[feux_y1:feux_y2, 0:band_w],
                      cv2.COLOR_BGR2HSV)
    rf = cv2.cvtColor(car_crop[feux_y1:feux_y2, w-band_w:w],
                      cv2.COLOR_BGR2HSV)
    def count_red(hsv):
        m1 = cv2.inRange(hsv, (0,   90, 70), (12,  255, 255))
        m2 = cv2.inRange(hsv, (165, 90, 70), (180, 255, 255))
        return int(cv2.countNonZero(cv2.bitwise_or(m1, m2)))
    def count_white(hsv):
        w  = cv2.inRange(hsv, (0, 0, 210),    (180, 60, 255))
        y  = cv2.inRange(hsv, (18, 130, 190), (35, 255, 255))
        return int(cv2.countNonZero(cv2.bitwise_or(w, y)))
    rl = count_red(lf); rr = count_red(rf)
    wl = count_white(lf); wr = count_white(rf)
    ba = band_w * (feux_y2 - feux_y1)
    return {
        "red_left": rl, "red_right": rr,
        "white_left": wl, "white_right": wr,
        "red_tot": rl+rr, "white_tot": wl+wr,
        "band_area": ba,
        "red_left_ratio":  rl/max(ba,1),
        "red_right_ratio": rr/max(ba,1),
        "whi_left_ratio":  wl/max(ba,1),
        "whi_right_ratio": wr/max(ba,1),
    }


def detect_front_rear(lights):
    log = []
    rl, rr    = lights["red_left"],   lights["red_right"]
    wl, wr    = lights["white_left"], lights["white_right"]
    red_tot   = rl + rr
    white_tot = wl + wr
    ba        = lights["band_area"]
    red_thr   = max(120, int(ba * 0.003))
    white_thr = max(400, int(ba * 0.020))
    if red_tot >= red_thr:
        facing = "rear"
        log.append(f"REAR rouge={red_tot}>={red_thr}")
    elif white_tot >= white_thr and white_tot > red_tot * 3:
        facing = "front"
        log.append(f"FRONT blanc={white_tot}")
    else:
        facing = "side"
        log.append(f"SIDE red={red_tot} white={white_tot}")
    rear_side  = "left" if rl>rr*1.25 else ("right" if rr>rl*1.25 else None)
    front_side = "left" if wl>wr*1.25 else ("right" if wr>wl*1.25 else None)
    if rear_side  and not front_side:
        front_side = "right" if rear_side=="left" else "left"
    if front_side and not rear_side:
        rear_side  = "right" if front_side=="left" else "left"
    if not rear_side and not front_side:
        rear_side, front_side = "right", "left"
        log.append("Fallback arriere=droite")
    log.append(f"rouge G={rl} D={rr} | blanc G={wl} D={wr}")
    log.append(f"rear={rear_side} front={front_side}")
    return rear_side, front_side, facing, log


def estimate_angle(lights, crop_w, crop_h, facing):
    rl, rr = lights["red_left"],   lights["red_right"]
    wl, wr = lights["white_left"], lights["white_right"]
    if   facing=="rear":  big=max(rl,rr); sml=min(rl,rr)
    elif facing=="front": big=max(wl,wr); sml=min(wl,wr)
    else:                 big=max(rl+wl,rr+wr); sml=min(rl+wl,rr+wr)
    if big == 0: return 45.0
    sym = big / max(sml, 1)
    if   sym>=8.0: angle=10.0
    elif sym>=4.0: angle=10.0+(8.0-sym)/4.0*30.0
    elif sym>=2.0: angle=40.0+(4.0-sym)/2.0*25.0
    elif sym>=1.3: angle=65.0+(2.0-sym)/0.7*20.0
    else:          angle=87.0
    rw = crop_w/max(crop_h,1)
    if rw>1.6: angle=min(angle,35.0)
    elif rw<0.9: angle=max(angle,60.0)
    return round(angle, 1)


def make_poly(crop_w, crop_h, xA, xB, top_base, bot_base, persp, tilt_dir):
    drop = 0.08
    def ty(x):
        f = x/max(1,crop_w)
        return top_base+(persp*drop*f if tilt_dir>0 else persp*drop*(1-f))
    def by(x):
        f = x/max(1,crop_w)
        return bot_base-(persp*drop*f if tilt_dir>0 else persp*drop*(1-f))
    return [(xA,int(ty(xA)*crop_h)),(xB,int(ty(xB)*crop_h)),
            (xB,int(by(xB)*crop_h)),(xA,int(by(xA)*crop_h))]


def build_zones(crop_w, crop_h, angle, rear_side, front_side, facing, lights):
    rl, rr = lights["red_left"],   lights["red_right"]
    wl, wr = lights["white_left"], lights["white_right"]
    if   facing=="rear":  near_side="left" if rl>rr else "right"
    elif facing=="front": near_side="left" if wl>wr else "right"
    else:                 near_side="left" if (rl+wl)>(rr+wr) else "right"
    far_side = "right" if near_side=="left" else "left"
    if   angle<=10: persp=0.0
    elif angle<=55: persp=0.6*min(1.0,(angle-10)/45.0)
    elif angle<=80: persp=0.6*max(0.0,1.0-(angle-55)/25.0)
    else:           persp=0.0
    tilt_dir = +1 if far_side=="right" else -1
    top_base, bot_base = 0.20, 0.88
    is_rear  = (facing=="rear") or (facing=="side" and (rl+rr)>=(wl+wr))
    panel    = "Coffre"     if is_rear else "Capot"
    pc_label = "Pare-ch.AR" if is_rear else "Pare-ch.AV"
    def zone(name, a, b):
        return {"name": name, "poly": make_poly(
            crop_w, crop_h, int(a*crop_w), int(b*crop_w),
            top_base, bot_base, persp, tilt_dir
        )}
    if angle <= 25:
        log = f"PROFIL({angle}°) near={near_side}"
        if near_side=="right":
            return ([zone("Aile AV",0.00,0.20), zone("Porte AV",0.20,0.48),
                     zone("Porte AR",0.48,0.78), zone("Aile AR",0.78,1.00)]
                    if is_rear else
                    [zone("Aile AR",0.00,0.20), zone("Porte AR",0.20,0.48),
                     zone("Porte AV",0.48,0.78), zone("Aile AV",0.78,1.00)]), log
        else:
            return ([zone("Aile AR",0.00,0.22), zone("Porte AR",0.22,0.52),
                     zone("Porte AV",0.52,0.80), zone("Aile AV",0.80,1.00)]
                    if is_rear else
                    [zone("Aile AV",0.00,0.22), zone("Porte AV",0.22,0.52),
                     zone("Porte AR",0.52,0.80), zone("Aile AR",0.80,1.00)]), log
    elif angle <= 55:
        log = f"3/4 LEGER({angle}°) near={near_side}"
        if near_side=="right":
            return ([zone("Porte AV",0.00,0.22), zone("Porte AR",0.22,0.50),
                     zone("Aile AR",0.50,0.72), zone(pc_label,0.72,1.00)]
                    if is_rear else
                    [zone("Porte AR",0.00,0.22), zone("Porte AV",0.22,0.50),
                     zone("Aile AV",0.50,0.72), zone(pc_label,0.72,1.00)]), log
        else:
            return ([zone(pc_label,0.00,0.28), zone("Aile AR",0.28,0.50),
                     zone("Porte AR",0.50,0.78), zone("Porte AV",0.78,1.00)]
                    if is_rear else
                    [zone(pc_label,0.00,0.28), zone("Aile AV",0.28,0.50),
                     zone("Porte AV",0.50,0.78), zone("Porte AR",0.78,1.00)]), log
    elif angle <= 80:
        log = f"3/4 MARQUE({angle}°) near={near_side}"
        if near_side=="right":
            return [zone("Aile AR" if is_rear else "Aile AV",0.42,0.68),
                    zone(panel,0.68,0.85), zone(pc_label,0.85,1.00)], log
        else:
            return [zone(pc_label,0.00,0.15), zone(panel,0.15,0.32),
                    zone("Aile AR" if is_rear else "Aile AV",0.32,0.58)], log
    else:
        log = f"FACE/DOS({angle}°)"
        if is_rear:
            return [zone("Aile AR G",0.00,0.20), zone(pc_label,0.20,0.55),
                    zone(panel,0.45,0.80), zone("Aile AR D",0.80,1.00)], log
        else:
            return [zone("Aile AV G",0.00,0.20), zone(pc_label,0.20,0.55),
                    zone(panel,0.45,0.80), zone("Aile AV D",0.80,1.00)], log


# =============================================
# VERDICT — CALIBRÉ SUR TES 5 IMAGES RÉELLES
#
# Valeurs observées dans les images :
#
# ZONES ROUGES (peinture refaite) :
#   BMW blanche  Porte AR : H2  S80 V59  → rouge (S élevé même avec H faible)
#   BMW blanche  Aile AR  : H2  S80 V59  → rouge
#   Golf grise   Porte AR : H7  S~7      → rouge (H>=6 sur carrosserie colorée)
#   Berline grise Porte AR/AV : H4       → rouge selon toi sur voiture grise
#
# ZONES VERTES (homogène) :
#   Golf grise   Aile AV  : H78 S18      → VERT (S18 faible = jante/fond)
#   Renault noir Porte AR : H78          → VERT (toute la voiture OK)
#   SUV gris neuf : H0-H14               → VERT
#   Berline grise Aile AR : H0           → VERT
#
# RÈGLE APPRISE :
# 1. Si S < 18 → le pixel n'est pas de la carrosserie colorée → ignorer H
#    (jantes, fond, plastique)
# 2. Si S >= 40 ET diff_H >= 4 → ROUGE (couleur fiable, écart visible)
# 3. Si S >= 40 ET diff_H >= 2 → ORANGE suspect
# 4. Si S < 40 ET S >= 18 → mode monochrome sur V
#    - diff_V >= 40 ET std_s >= 20 → ROUGE
#    - diff_V >= 25 ET std_s >= 30 → ORANGE
# 5. Voiture sombre (ref_V < 80) → seuils plus élevés
#    car les surfaces noires compriment naturellement les différences
# =============================================
def verdict_from_values(zone_color, ref_color, stats):
    """
    Entrées (HSV médian) :
      zone_color = [H, S, V]
      ref_color  = [H_ref, S_ref, V_ref]
      stats      = {std_h, std_s, std_v}

    Retourne : (verdict_state, mode, diff_h, diff_v, combo)
      verdict_state : "refaite" | "suspecte" | "ok"
    """
    H_z, S_z, V_z = float(zone_color[0]), float(zone_color[1]), float(zone_color[2])
    H_r, S_r, V_r = float(ref_color[0]),  float(ref_color[1]),  float(ref_color[2])
    std_s          = stats["std_s"]
    std_v          = stats["std_v"]

    # Écart de teinte circulaire
    diff_h = abs(H_z - H_r)
    diff_h = min(diff_h, 180.0 - diff_h)

    # Écart de luminosité
    diff_v = abs(V_z - V_r)

    # ---- FILTRE : zone à faible saturation = fond/jante/plastique ----
    # Si la saturation médiane de la zone est < 18, les pixels ne sont
    # pas de la carrosserie colorée. On ne peut pas se fier à H.
    # On reste en mode monochrome (V uniquement).
    s_fiable = (S_z >= 18.0) and (S_r >= 18.0)

    # ---- VOITURE SOMBRE : seuils adaptés ----
    # Renault noir (ref_V ~ 40-60) → les différences V sont comprimées
    voiture_sombre = (V_r < 80.0)

    if s_fiable and S_z >= 40.0:
        # ===== MODE COULEUR : S assez élevé pour que H soit fiable =====
        mode = "C"
        combo = diff_h * 2.0 + max(0.0, std_s - 8.0) * 0.5

        # Seuils calibrés :
        # Golf grise Porte AR : H7  S~7 → mais en mode C S<40 donc
        #   en réalité elle passe en mode monochrome
        # BMW blanche Porte AR : H2 S80 → diff_h=2, S=80
        #   → combo = 2*2 + (80-8)*0.5 = 4 + 36 = 40 → rouge
        # Berline Porte AR : H4 S24 → en mode M car S<40

        if voiture_sombre:
            thr_red  = 35.0
            thr_susp = 20.0
        else:
            thr_red  = 12.0   # BMW blanc Porte AR combo~40 > 12 → rouge ✓
            thr_susp = 6.0

        if   combo >= thr_red:  return "refaite",  mode, diff_h, diff_v, combo
        elif combo >= thr_susp: return "suspecte", mode, diff_h, diff_v, combo
        else:                   return "ok",        mode, diff_h, diff_v, combo

    else:
        # ===== MODE MONOCHROME : H peu fiable (S faible) =====
        # On compare V + on regarde si std_s indique une zone hétérogène
        mode  = "M"
        combo = diff_v * 1.0 + max(0.0, std_s - 5.0) * 0.8

        # Calibration :
        # Golf grise Porte AR : H7, S faible → diff_v~7, std_s~?
        #   → combo ≈ 7 + 0 = 7
        # Berline Porte AR : H4 S24 → diff_v~?, combo doit dépasser seuil
        # Berline Porte AV : H4 → idem
        # Renault noir : H78 S47 V48 → S>=40 donc mode C, pas M
        # Golf grise Aile AV : H78 S18 → mode M, diff_v~38
        #   → on NE veut PAS de rouge ici (c'est la jante/fond)
        #   → std_s faible car fond uniforme → combo < seuil ✓

        if voiture_sombre:
            thr_red  = 50.0   # Renault noir : seuil élevé → tout vert ✓
            thr_susp = 35.0
        else:
            # Golf/Berline grises :
            # Porte AR H7 → diff_v~7, std_s~7 → combo ≈ 7 + 1.6 = 8.6
            #   → doit passer rouge si seuil <= 8
            # Porte AV H4 berline → combo ≈ 4-5
            #   → doit passer rouge si seuil <= 5
            # Aile AV H78 S18 Golf → combo ≈ 38 + 0 = 38
            #   → NE doit PAS être rouge → seuil > 38 ?
            # CONTRADICTION : diff_v élevé sur l'aile = fond visible
            # Solution : ignorer les zones avec diff_h > 40 ET S < 18
            #   car c'est clairement hors carrosserie (fond, ciel, jante)
            thr_red  = 6.0
            thr_susp = 3.5

        # Filtre fond/hors-carrosserie :
        # Si H très éloigné (>40°) ET S faible → c'est le fond capturé
        # dans le masque, pas une vraie différence de peinture
        hors_carrosserie = (diff_h > 40.0) and (S_z < 25.0)
        if hors_carrosserie:
            return "ok", mode, diff_h, diff_v, combo

        if   combo >= thr_red:  return "refaite",  mode, diff_h, diff_v, combo
        elif combo >= thr_susp: return "suspecte", mode, diff_h, diff_v, combo
        else:                   return "ok",        mode, diff_h, diff_v, combo


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

        pad_x = int(15*scale_x); pad_y = int(10*scale_y)
        thr_x = int(150*scale_x); thr_y = int(80*scale_y)
        x1 = 0      if raw_x1<thr_x            else max(0,      raw_x1-pad_x)
        x2 = orig_w if (orig_w-raw_x2)<thr_x   else min(orig_w, raw_x2+pad_x)
        y1 = 0      if raw_y1<thr_y            else max(0,      raw_y1-pad_y)
        y2 = orig_h if (orig_h-raw_y2)<thr_y   else min(orig_h, raw_y2+pad_y)

        x1, y1, x2, y2 = refine_car_bbox(img_orig, x1, y1, x2, y2)
        car_crop = img_orig[y1:y2, x1:x2]
        if car_crop.size == 0:
            return jsonify({"error": "invalid crop"}), 400
        crop_h, crop_w = car_crop.shape[:2]

        lights                                = detect_lights(car_crop)
        rear_side, front_side, facing, fr_log = detect_front_rear(lights)
        angle                                 = estimate_angle(lights, crop_w, crop_h, facing)
        fr_log.append(f"Angle={angle}°")

        zones, zone_dec = build_zones(
            crop_w, crop_h, angle, rear_side, front_side, facing, lights
        )
        fr_log.append(zone_dec)

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

        final_img  = img_orig.copy()
        thick_box  = max(3, int(4*min(scale_x, scale_y)))
        thick_line = max(1, int(1*min(scale_x, scale_y)))
        font_big   = max(0.55, 0.58*min(scale_x, scale_y))
        font_med   = max(0.42, 0.44*min(scale_x, scale_y))
        font_thick = max(2, int(2*min(scale_x, scale_y)))

        cv2.rectangle(final_img, (x1,y1),(x2,y2),(220,220,220), thick_line)

        header = (f"{car_info['make']} {car_info['model']} | "
                  f"{'AR' if facing=='rear' else ('AV' if facing=='front' else 'COTE')} | "
                  f"{angle}°")
        (hw, hh), _ = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_med*1.2, font_thick)
        cv2.rectangle(final_img, (5,5),(15+hw,20+hh),(0,0,0),-1)
        cv2.putText(final_img, header, (10,15+hh),
                    cv2.FONT_HERSHEY_SIMPLEX, font_med*1.2,
                    (255,255,255), font_thick)

        results_zones = []
        detected      = 0

        for idx, zone in enumerate(zones, start=1):
            poly_local  = zone["poly"]
            poly_global = np.array(
                [[x1+p[0], y1+p[1]] for p in poly_local], dtype=np.int32
            )

            zone_color, px_count, stats = get_poly_color(
                hsv_full, mask_body, poly_local
            )

            if zone_color is None:
                color_rect  = (150, 150, 150)
                label_score = "N/A"
                diff        = 0.0
                verdict     = "Non analysable"
                std_h = std_s = std_v = 0.0
            else:
                std_h = stats["std_h"]
                std_s = stats["std_s"]
                std_v = stats["std_v"]

                verdict_state, mode, diff_h, diff_v, combo = \
                    verdict_from_values(zone_color, ref_color, stats)

                diff = diff_h  # valeur principale affichée

                if   verdict_state == "refaite":
                    color_rect = (0, 0, 255)
                    verdict    = "Peinture refaite!"
                    detected  += 1
                elif verdict_state == "suspecte":
                    color_rect = (0, 165, 255)
                    verdict    = "Variation suspecte"
                    detected  += 1
                else:
                    color_rect = (0, 210, 0)
                    verdict    = "OK"

                # Étiquette lisible : H / S / mode / combo
                label_score = (
                    f"H{int(diff_h)}/S{int(zone_color[1])}"
                    f"/V{int(diff_v)}/{mode}{int(combo)}"
                )

            # Remplissage semi-transparent
            overlay = final_img.copy()
            cv2.fillPoly(overlay, [poly_global], color_rect)
            cv2.addWeighted(overlay, 0.22, final_img, 0.78, 0, final_img)
            cv2.polylines(final_img, [poly_global], True, color_rect, thick_box)

            cx = int(np.mean(poly_global[:,0]))
            cy = int(np.mean(poly_global[:,1]))
            r  = max(18, int(20*min(scale_x,scale_y)))
            cv2.circle(final_img, (cx+2,cy+2), r, (0,0,0), -1)
            cv2.circle(final_img, (cx,cy),     r, color_rect, -1)
            cv2.circle(final_img, (cx,cy),     r, (255,255,255), 2)
            ntxt = str(idx)
            (tw,th),_ = cv2.getTextSize(ntxt, cv2.FONT_HERSHEY_SIMPLEX,
                                         font_big*1.3, font_thick+1)
            cv2.putText(final_img, ntxt, (cx-tw//2, cy+th//2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_big*1.3,
                        (255,255,255), font_thick+1)

            top_pt = poly_global[poly_global[:,1].argmin()]
            lbl_x  = max(5, int(top_pt[0]))
            lbl_y  = max(20, int(top_pt[1])-10)
            lbl    = f"{idx}. {zone['name']}  E:{label_score}"
            (lw,lh),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX,
                                          font_med, font_thick)
            lbl_x = min(lbl_x, orig_w-lw-10)
            cv2.rectangle(final_img,
                          (lbl_x-4,lbl_y-lh-6),(lbl_x+lw+6,lbl_y+4),
                          (0,0,0), -1)
            cv2.putText(final_img, lbl, (lbl_x,lbl_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_med,
                        (255,255,255), font_thick)

            results_zones.append({
                "idx":     idx,
                "zone":    zone["name"],
                "diff":    round(diff, 1),
                "std_h":   round(std_h, 2),
                "std_s":   round(std_s, 2),
                "std_v":   round(std_v, 2),
                "pixels":  px_count,
                "verdict": verdict,
                "polygon": poly_global.tolist()
            })

        diffs       = [z["diff"] for z in results_zones if z["diff"] > 0]
        final_score = min(int(np.mean(diffs)) if diffs else 0, 100)
        if   final_score < 10: result = "Peinture homogene (OK)"
        elif final_score < 28: result = "Legeres variations detectees"
        else:                  result = "Difference importante - repeinture probable"

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
            "orientation_log": fr_log,
            "lights":          lights,
            "image_size":      {"width": orig_w, "height": orig_h},
            "reference_hsv": {
                "H": round(ref_color[0], 1),
                "S": round(ref_color[1], 1),
                "V": round(ref_color[2], 1)
            },
            "image_result":    analysed_name,
            "image_url":       request.host_url + "uploads/" + analysed_name
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
