@app.route("/analyse", methods=["POST"])
def analyse():

    try:

        if 'image' not in request.files:
            return jsonify({"error": "no image"}), 400

        file = request.files['image']

        filename = str(int(time.time())) + "_" + secure_filename(file.filename)

        path = os.path.join(UPLOAD_FOLDER, filename)

        file.save(path)

        img = cv2.imread(path)

        if img is None:
            return jsonify({"error": "image not readable"}), 400

        # =========================
        # RESIZE SAFE
        # =========================

        img = cv2.resize(img, (900, 500))
        original = img.copy()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # =========================
        # DETECTION CAR (CONTOUR MAIN)
        # =========================

        edges = cv2.Canny(blur, 60, 120)

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return jsonify({"error": "no object detected"}), 400

        car_contour = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(car_contour)

        car = gray[y:y+h, x:x+w]

        if car.size == 0:
            return jsonify({"error": "empty crop"}), 400

        car = cv2.resize(car, (600, 300))

        # =========================
        # PATCH ANALYSIS (IA LÉGÈRE)
        # =========================

        h_patches = 6
        w_patches = 10

        ph = car.shape[0] // h_patches
        pw = car.shape[1] // w_patches

        heatmap = np.zeros_like(car)

        total_score = 0
        zones = 0

        for i in range(h_patches):
            for j in range(w_patches):

                patch = car[i*ph:(i+1)*ph, j*pw:(j+1)*pw]

                if patch.size == 0:
                    continue

                brightness = np.mean(patch)
                texture = cv2.Laplacian(patch, cv2.CV_64F).var()

                score_local = 0

                # =========================
                # LOGIQUE IA SIMPLE MAIS EFFICACE
                # =========================

                if texture < 60:
                    score_local += 40

                if brightness > 170 or brightness < 60:
                    score_local += 30

                if 80 < brightness < 120 and texture < 100:
                    score_local += 30

                if score_local > 50:

                    zones += 1
                    total_score += score_local

                    cv2.rectangle(
                        original,
                        (j*pw, i*ph),
                        ((j+1)*pw, (i+1)*ph),
                        (0, 0, 255),
                        2
                    )

                    cv2.rectangle(
                        heatmap,
                        (j*pw, i*ph),
                        ((j+1)*pw, (i+1)*ph),
                        255,
                        -1
                    )

        # =========================
        # HEATMAP
        # =========================

        heat_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        final = cv2.addWeighted(original, 0.85, heat_color, 0.35, 0)

        # =========================
        # SCORE FINAL
        # =========================

        score = int(min(total_score / 2, 100))

        if score < 20:
            result = "Peinture normale"
        elif score < 50:
            result = "Doute léger sur peinture"
        elif score < 75:
            result = "Peinture probablement refaite"
        else:
            result = "Forte suspicion de retouche peinture"

        # =========================
        # SAVE IMAGE RESULT
        # =========================

        analysed_name = "analysed_" + filename
        analysed_path = os.path.join(UPLOAD_FOLDER, analysed_name)

        cv2.imwrite(analysed_path, final)

        return jsonify({
            "score": score,
            "result": result,
            "zones_detected": zones,
            "image_result": analysed_name
        })

    except Exception as e:

        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500
