import os
import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import logging
import uuid
from datetime import datetime

# Configuration
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# FONCTIONS DE TRAITEMENT D'IMAGE
# ============================================================

def get_car_mask_excluding_white_ground(hsv_img):
    """
    Crée un masque qui isole la carrosserie en ignorant :
    - Le ciel (bleu clair)
    - Le sol blanc/clair
    - Les ombres très sombres
    """
    # Plage HSV pour la carrosserie (couleurs de voiture courantes)
    # On exclut le blanc pur (sol) et le bleu clair (ciel)
    
    lower_car = np.array([0, 30, 30])      # Teinte min, Saturation min, Value min
    upper_car = np.array([180, 255, 230])   # Teinte max, Saturation max, Value max (exclut blanc)
    
    # Masque principal : tout ce qui ressemble à une carrosserie
    car_mask = cv2.inRange(hsv_img, lower_car, upper_car)
    
    # Exclure les zones trop claires (sol blanc)
    white_threshold = 240
    not_white = cv2.inRange(hsv_img, (0, 0, 0), (180, 60, white_threshold))
    
    # Exclure les zones trop sombres (ombres)
    dark_threshold = 40
    not_dark = cv2.inRange(hsv_img, (0, 0, dark_threshold), (180, 255, 255))
    
    # Combiner les masques
    final_mask = cv2.bitwise_and(car_mask, not_white)
    final_mask = cv2.bitwise_and(final_mask, not_dark)
    
    # Nettoyage morphologique
    kernel = np.ones((5,5), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    
    return final_mask


def get_dominant_color(hsv_img, mask):
    """
    Trouve la couleur la plus fréquente (mode) dans les pixels valides
    au lieu de la moyenne qui est faussée par les valeurs aberrantes
    """
    # Extraire les pixels valides
    valid_pixels = hsv_img[mask > 0]
    
    if len(valid_pixels) == 0:
        return np.array([0, 0, 0])
    
    # Trouver la teinte dominante (H)
    h_values = valid_pixels[:, 0]
    h_hist = np.bincount(h_values.astype(int), minlength=180)
    dominant_h = np.argmax(h_hist)
    
    # Trouver la saturation dominante (S) pour cette teinte
    s_values = valid_pixels[(valid_pixels[:, 0] >= dominant_h - 5) & 
                            (valid_pixels[:, 0] <= dominant_h + 5)][:, 1]
    if len(s_values) > 0:
        s_hist = np.bincount(s_values.astype(int), minlength=256)
        dominant_s = np.argmax(s_hist)
    else:
        dominant_s = 0
    
    # Trouver la valeur dominante (V) pour cette teinte ET cette saturation
    v_values = valid_pixels[(valid_pixels[:, 0] >= dominant_h - 5) & 
                            (valid_pixels[:, 0] <= dominant_h + 5) &
                            (valid_p

                            (valid_pixels[:, 1] >= dominant_s - 8) & 
                            (valid_pixels[:, 1] <= dominant_s + 8)
                           ][:, 2]
    
    if len(v_values) > 0:
        v_hist = np.bincount(v_values.astype(int), minlength=256)
        dominant_v = np.argmax(v_hist)
    else:
        dominant_v = 0
    
    return np.array([dominant_h, dominant_s, dominant_v])


def analyze_texture(gray_img, mask):
    """
    Analyse la texture de la carrosserie en utilisant :
    - Variance locale (Laplacian)
    - Gradient moyen
    - Homogénéité
    """
    # Appliquer le masque
    masked_gray = cv2.bitwise_and(gray_img, gray_img, mask=mask)
    
    # 1. Variance locale (Laplacian) - détecte les détails fins
    laplacian = cv2.Laplacian(masked_gray, cv2.CV_64F)
    laplacian_variance = np.var(laplacian[laplacian != 0]) if np.sum(laplacian != 0) > 0 else 0
    
    # 2. Gradient moyen (Sobel) - détecte les transitions
    sobel_x = cv2.Sobel(masked_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(masked_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    mean_gradient = np.mean(gradient_magnitude[gradient_magnitude != 0]) if np.sum(gradient_magnitude != 0) > 0 else 0
    
    # 3. Homogénéité (GLCM simplifié)
    # On divise l'image en blocs et on calcule la variance intra-bloc
    h, w = gray_img.shape
    block_size = 20
    block_variances = []
    
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block_mask = mask[y:y+block_size, x:x+block_size]
            if np.sum(block_mask) > (block_size * block_size * 0.3):  # Au moins 30% du bloc est valide
                block = gray_img[y:y+block_size, x:x+block_size]
                block_var = np.var(block[block_mask > 0]) if np.sum(block_mask > 0) > 0 else 0
                block_variances.append(block_var)
    
    homogeneity = np.mean(block_variances) if block_variances else 0
    
    # Score de texture normalisé (0 = très lisse, 1 = très texturé)
    texture_score = min(1.0, (laplacian_variance / 1000 + mean_gradient / 50 + homogeneity / 500) / 3)
    
    return {
        "laplacian_variance": float(laplacian_variance),
        "mean_gradient": float(mean_gradient),
        "homogeneity": float(homogeneity),
        "texture_score": float(texture_score)
    }


def analyze_paint_quality(hsv_img, gray_img, mask):
    """
    Analyse complète de la qualité de la peinture
    """
    # 1. Couleur dominante
    dominant_color = get_dominant_color(hsv_img, mask)
    
    # 2. Analyse de texture
    texture_data = analyze_texture(gray_img, mask)
    
    # 3. Évaluation de la qualité basée sur la texture
    # Une bonne peinture = texture lisse et homogène
    if texture_data["texture_score"] < 0.3:
        quality = "excellente"
        quality_score = 5
    elif texture_data["texture_score"] < 0.5:
        quality = "bonne"
        quality_score = 4
    elif texture_data["texture_score"] < 0.7:
        quality = "moyenne"
        quality_score = 3
    elif texture_data["texture_score"] < 0.85:
        quality = "passable"
        quality_score = 2
    else:
        quality = "mauvaise

        quality = "mauvaise"
        quality_score = 1

    # 4. Détection de défauts (rayures, bosses)
    # Utilisation du gradient pour trouver des anomalies
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Seuil adaptatif pour les défauts
    defect_threshold = np.percentile(gradient_magnitude[mask > 0], 95) if np.sum(mask > 0) > 0 else 50
    defects = (gradient_magnitude > defect_threshold) & (mask > 0)
    defect_percentage = float(np.sum(defects) / np.sum(mask > 0) * 100) if np.sum(mask > 0) > 0 else 0

    return {
        "dominant_color_hsv": {
            "hue": int(dominant_color[0]),
            "saturation": int(dominant_color[1]),
            "value": int(dominant_color[2])
        },
        "dominant_color_rgb": hsv_to_rgb(dominant_color),
        "texture": texture_data,
        "quality": quality,
        "quality_score": quality_score,
        "defect_percentage": defect_percentage
    }


def hsv_to_rgb(hsv_color):
    """Convertit une couleur HSV en RGB"""
    h, s, v = hsv_color
    # OpenCV HSV : H=0-180, S=0-255, V=0-255
    # Conversion vers HSV standard : H*2, S/255, V/255
    h_std = h * 2
    s_std = s / 255.0
    v_std = v / 255.0
    
    # Conversion HSV standard vers RGB
    c = v_std * s_std
    x = c * (1 - abs((h_std / 60) % 2 - 1))
    m = v_std - c
    
    if h_std < 60:
        r, g, b = c, x, 0
    elif h_std < 120:
        r, g, b = x, c, 0
    elif h_std < 180:
        r, g, b = 0, c, x
    elif h_std < 240:
        r, g, b = 0, x, c
    elif h_std < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    
    return {
        "red": int((r + m) * 255),
        "green": int((g + m) * 255),
        "blue": int((b + m) * 255)
    }


def process_image(image_path):
    """
    Traite une image : charge, analyse peinture, retourne résultats
    """
    # Charger l'image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossible de charger l'image : {image_path}")
    
    # Redimensionner si trop grande
    h, w = img.shape[:2]
    if max(h, w) > 1200:
        scale = 1200 / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
    
    # Convertir en HSV et gris
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Obtenir le masque de la carrosserie (sans le sol blanc)
    mask = get_car_mask_excluding_white_ground(hsv_img)
    
    # Vérifier qu'on a assez de pixels valides
    if np.sum(mask > 0) < 1000:
        return {
            "error": "Pas assez de carrosserie détectée. Vérifiez que la voiture est bien visible.",
            "mask_coverage": float(np.sum(mask > 0) / (

    if np.sum(mask > 0) < 500:  # Moins de 500 pixels valides
        logger.warning(f"Pas assez de pixels de carrosserie détectés dans l'image")
        return None, "Impossible de détecter la carrosserie. L'image doit contenir une voiture visible."
    
    # Analyser la peinture
    result = analyze_paint_quality(hsv_img, gray_img, mask)
    
    # Sauvegarder l'image avec le masque (pour déboguer)
    debug_img = img.copy()
    debug_img[mask > 0] = [0, 255, 0]  # Marquer la zone analysée en vert
    debug_path = os.path.join(app.config["UPLOAD_FOLDER"], f"debug_{filename}")
    cv2.imwrite(debug_path, debug_img)
    result["debug_image"] = debug_path
    
    return result, None


# ============================================================
# API ENDPOINTS
# ============================================================

@app.route("/analyze", methods=["POST"])
def analyze_image():
    """
    Endpoint pour analyser une image de voiture
    Accepte : fichier image ou URL
    """
    # Vérifier si fichier ou URL
    if "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Aucun fichier sélectionné"}), 400
        
        # Sauvegarder le fichier
        filename = secure_filename(file.filename)
        # Ajouter un UUID pour éviter les collisions
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
        file.save(filepath)
        
    elif "url" in request.json:
        url = request.json["url"]
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return jsonify({"error": f"Impossible de télécharger l'image (HTTP {response.status_code})"}), 400
            
            # Sauvegarder l'image téléchargée
            filename = f"url_{uuid.uuid4().hex}.jpg"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            with open(filepath, "wb") as f:
                f.write(response.content)
        except Exception as e:
            return jsonify({"error": f"Erreur lors du téléchargement : {str(e)}"}), 400
    else:
        return jsonify({"error": "Veuillez fournir un fichier (file) ou une URL (url)"}), 400
    
    # Analyser l'image
    result, error = process_image(filepath, filename)
    
    if error:
        return jsonify({"error": error}), 400
    
    # Ajouter des métadonnées
    result["filename"] = filename
    result["analyzed_at"] = datetime.now().isoformat()
    
    return jsonify(result), 200


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """Servir les fichiers uploadés (pour le débogage)"""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/health", methods=["GET"])
def health_check():
    """Endpoint de vérification de santé"""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }), 200


# ============================================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================================

if __name__ == "__main__":
    logger.info("🚗 Démarrage du service d'analyse de peinture automobile...")
    logger.info(f"📁 Dossier d'upload : {UPLOAD_FOLDER}")
    logger.info("🌐 Serveur démarré sur http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
Fichier requirements.txt
Flask==3.0.0
opencv-python==4.9.0.80
numpy==1.26.2
requests==2.31.0
Werkzeug==3.0.1
gunicorn==21.2.0
Instructions d’installation
# 1. Créer un environnement virt
suite ne t aret pas

# 1. Créer un environnement virtuel
python -m venv venv

# 2. Activer l'environnement virtuel
# Sur Windows :
venv\Scripts\activate
# Sur MacOS/Linux :
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
python app.py

# 5. Tester avec curl (exemple)
curl -X POST -F "file=@test_car.jpg" http://localhost:5000/analyze

# 6. Ou avec une URL
curl -X POST -H "Content-Type: application/json" \
     -d '{"url": "https://example.com/car.jpg"}' \
     http://localhost:5000/analyze
