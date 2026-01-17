import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import os

# =========================
# MediaPipe + Landmarks
# =========================
mp_face_mesh = mp.solutions.face_mesh

LMS = {
    "nose": 1,
    "chin": 152,
    "left_eye": 33,
    "right_eye": 263,

    "forehead": 10,
    "upper_lip": 13,
    "lower_lip": 14,
    "mouth_left": 61,
    "mouth_right": 291,

    "face_left": 234,
    "face_right": 454,

    # iris (cuando refine_landmarks=True)
    "l_iris_1": 468,
    "l_iris_2": 469,
    "l_iris_3": 470,
    "l_iris_4": 471,
    "l_iris_5": 472,
    "r_iris_1": 473,
    "r_iris_2": 474,
    "r_iris_3": 475,
    "r_iris_4": 476,
    "r_iris_5": 477,
}

# =========================
# Filtros
# =========================
FILTROS = [
    "A: Sombrero (PNG)",
    "B: Gafas (PNG)",
    "C: Bigote (PNG)",
    "D: Orejas (PNG)",
    "E: Escena (Color)",
    "F: Ojos Glow (Simple)",

    # 5 NUEVOS (deformaciones)
    "G: Ojos Grandes (Deform)",
    "H: Boca Grande (Deform)",
    "I: Nariz Pequeña (Deform)",
    "J: Cara Ancha (Deform)",
    "K: Cara Delgada (Deform)",
]
active_filter_index = 0

# =========================
# Control por gesto (tu lógica)
# =========================
baseline_ratio = None
window = deque(maxlen=7)
THRESHOLD = 0.10
STATE = "NEUTRO"
DEBOUNCE_DELAY_MS = 1000
last_change_time = 0

frame_count = 0
start_time = time.time()

# =========================
# Assets (PNGs)
# =========================
ASSET_DIR = "assets"
ASSET_PATHS = {
    "hat": os.path.join(ASSET_DIR, "sombrero.png"),
    "glasses": os.path.join(ASSET_DIR, "gafas.png"),
    "mustache": os.path.join(ASSET_DIR, "bigote.png"),
    "ears": os.path.join(ASSET_DIR, "orejas.png"),
}

def lm_xy(face, idx, w, h):
    lm = face.landmark[idx]
    return int(lm.x * w), int(lm.y * h)

def dist(p1, p2):
    return float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))

def angle_deg(p1, p2):
    return float(np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])))

def clamp(v, a, b):
    return max(a, min(b, v))

# --------- FIX PNG: alpha real o “key” si trae tablero ----------
def ensure_alpha_clean(bgra):
    """
    Si el PNG no trae alpha útil o trae checkerboard/blanco pegado,
    fabricamos un alpha por "distancia a colores de tablero" (dos grises).
    """
    if bgra is None:
        return None

    if bgra.ndim == 2:
        bgra = cv2.cvtColor(bgra, cv2.COLOR_GRAY2BGRA)
    if bgra.shape[2] == 3:
        bgra = cv2.cvtColor(bgra, cv2.COLOR_BGR2BGRA)

    b, g, r, a = cv2.split(bgra)

    # Si alpha ya sirve, nos quedamos con ese (pero lo suavizamos un pelo)
    if np.mean(a) < 250:
        a2 = cv2.GaussianBlur(a, (3, 3), 0)
        return cv2.merge([b, g, r, a2])

    bgr = cv2.merge([b, g, r]).astype(np.float32)

    # Dos grises típicos del checkerboard (ajustables)
    c1 = np.array([235, 235, 235], np.float32)  # muy claro
    c2 = np.array([200, 200, 200], np.float32)  # gris medio claro

    d1 = np.linalg.norm(bgr - c1, axis=2)
    d2 = np.linalg.norm(bgr - c2, axis=2)
    d = np.minimum(d1, d2)

    # Umbral: mientras más pequeño, más agresivo borrando el fondo
    thr = 40.0
    alpha = np.clip((d - 5.0) / thr, 0.0, 1.0)  # 0 fondo, 1 objeto

    # Refuerza objeto (exagera un poco el recorte)
    alpha = alpha ** 1.6

    a_new = (alpha * 255).astype(np.uint8)

    # Limpieza
    a_new = cv2.GaussianBlur(a_new, (5, 5), 0)
    a_new = cv2.threshold(a_new, 18, 255, cv2.THRESH_TOZERO)[1]

    return cv2.merge([b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8), a_new])

def load_rgba(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img = ensure_alpha_clean(img)
    return img

ASSETS = {k: load_rgba(v) for k, v in ASSET_PATHS.items()}

def rotate_rgba(img, angle):
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(
        img, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_TRANSPARENT
    )

def overlay_rgba(frame_bgr, overlay_bgra, x, y):
    if overlay_bgra is None:
        return frame_bgr

    h, w = overlay_bgra.shape[:2]
    H, W = frame_bgr.shape[:2]

    if x >= W or y >= H or x + w <= 0 or y + h <= 0:
        return frame_bgr

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)

    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    roi = frame_bgr[y1:y2, x1:x2]
    overlay_crop = overlay_bgra[oy1:oy2, ox1:ox2]

    alpha = overlay_crop[:, :, 3].astype(np.float32) / 255.0
    alpha = alpha[..., None]

    blended = (1.0 - alpha) * roi.astype(np.float32) + alpha * overlay_crop[:, :, :3].astype(np.float32)
    roi[:] = blended.astype(np.uint8)
    return frame_bgr

# =========================
# Overlays (Sombrero/Gafas/Bigote/Orejas)
# =========================
def aplicar_sombrero(frame, face, w, h):
    hat = ASSETS["hat"]
    if hat is None:
        return frame

    le = lm_xy(face, LMS["left_eye"], w, h)
    re = lm_xy(face, LMS["right_eye"], w, h)
    fh = lm_xy(face, LMS["forehead"], w, h)

    eye_dist = dist(le, re)
    if eye_dist < 2:
        return frame

    ang = angle_deg(le, re)

    target_w = int(eye_dist * 2.5)
    scale = target_w / hat.shape[1]
    target_h = max(1, int(hat.shape[0] * scale))

    hat_rs = cv2.resize(hat, (max(1, target_w), target_h), interpolation=cv2.INTER_AREA)
    hat_rot = rotate_rgba(hat_rs, ang)

    cx = int((le[0] + re[0]) / 2)
    cy = int(fh[1] - eye_dist * 0.60)

    x = cx - hat_rot.shape[1] // 2
    y = cy - hat_rot.shape[0] // 2
    return overlay_rgba(frame, hat_rot, x, y)

def aplicar_gafas(frame, face, w, h):
    glasses = ASSETS["glasses"]
    if glasses is None:
        return frame

    le = lm_xy(face, LMS["left_eye"], w, h)
    re = lm_xy(face, LMS["right_eye"], w, h)

    eye_dist = dist(le, re)
    if eye_dist < 2:
        return frame

    ang = angle_deg(le, re)

    target_w = int(eye_dist * 2.5)
    scale = target_w / glasses.shape[1]
    target_h = max(1, int(glasses.shape[0] * scale))

    g_rs = cv2.resize(glasses, (max(1, target_w), target_h), interpolation=cv2.INTER_AREA)
    g_rot = rotate_rgba(g_rs, ang)

    cx = int((le[0] + re[0]) / 2)
    cy = int((le[1] + re[1]) / 2) + int(eye_dist * 0.05)

    x = cx - g_rot.shape[1] // 2
    y = cy - g_rot.shape[0] // 2
    return overlay_rgba(frame, g_rot, x, y)

def aplicar_bigote(frame, face, w, h):
    must = ASSETS["mustache"]
    if must is None:
        return frame

    le = lm_xy(face, LMS["left_eye"], w, h)
    re = lm_xy(face, LMS["right_eye"], w, h)
    nose = lm_xy(face, LMS["nose"], w, h)
    lip = lm_xy(face, LMS["upper_lip"], w, h)

    eye_dist = dist(le, re)
    if eye_dist < 2:
        return frame

    ang = angle_deg(le, re)

    target_w = int(eye_dist * 1.8)
    scale = target_w / must.shape[1]
    target_h = max(1, int(must.shape[0] * scale))

    m_rs = cv2.resize(must, (max(1, target_w), target_h), interpolation=cv2.INTER_AREA)
    m_rot = rotate_rgba(m_rs, ang)

    cx = nose[0]
    cy = int((nose[1] + lip[1]) / 2)

    x = cx - m_rot.shape[1] // 2
    y = cy - m_rot.shape[0] // 2
    return overlay_rgba(frame, m_rot, x, y)

def aplicar_orejas(frame, face, w, h):
    ears = ASSETS["ears"]
    if ears is None:
        return frame

    le = lm_xy(face, LMS["left_eye"], w, h)
    re = lm_xy(face, LMS["right_eye"], w, h)
    fl = lm_xy(face, LMS["face_left"], w, h)
    fr = lm_xy(face, LMS["face_right"], w, h)

    face_w = dist(fl, fr)
    if face_w < 2:
        return frame

    ang = angle_deg(le, re)

    target_w = int(face_w * 1.55)
    scale = target_w / ears.shape[1]
    target_h = max(1, int(ears.shape[0] * scale))

    e_rs = cv2.resize(ears, (max(1, target_w), target_h), interpolation=cv2.INTER_AREA)
    e_rot = rotate_rgba(e_rs, ang)

    cx = int((fl[0] + fr[0]) / 2)
    cy = int(min(le[1], re[1]) - face_w * 0.55)

    x = cx - e_rot.shape[1] // 2
    y = cy - e_rot.shape[0] // 2
    return overlay_rgba(frame, e_rot, x, y)

# =========================
# Filtros “social” (color / glow)
# =========================
def filtro_escena(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= 1.15
    hsv[..., 2] *= 1.08
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    out = cv2.convertScaleAbs(out, alpha=1.10, beta=5)
    return out

def iris_center(face, side, w, h):
    if side == "L":
        ids = [LMS["l_iris_1"], LMS["l_iris_2"], LMS["l_iris_3"], LMS["l_iris_4"], LMS["l_iris_5"]]
    else:
        ids = [LMS["r_iris_1"], LMS["r_iris_2"], LMS["r_iris_3"], LMS["r_iris_4"], LMS["r_iris_5"]]

    pts = [lm_xy(face, i, w, h) for i in ids]
    cx = int(sum(p[0] for p in pts) / len(pts))
    cy = int(sum(p[1] for p in pts) / len(pts))
    return (cx, cy)

def filtro_ojos_glow(frame, face, w, h):
    le = lm_xy(face, LMS["left_eye"], w, h)
    re = lm_xy(face, LMS["right_eye"], w, h)
    eye_dist = dist(le, re)
    r = max(6, int(eye_dist * 0.08))

    overlay = frame.copy()
    cv2.circle(overlay, le, r, (255, 255, 255), -1)
    cv2.circle(overlay, re, r, (255, 255, 255), -1)
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
    return frame

# =========================
# Deformaciones (bulge/pinch) con remap (rápido y “TikTok-like”)
# =========================
def bulge_pinch_roi(img, center, radius, strength):
    """
    strength > 0  => bulge (agrandar)
    strength < 0  => pinch (encoger)
    """
    H, W = img.shape[:2]
    cx, cy = center
    radius = int(radius)
    if radius < 5:
        return img

    x1 = clamp(cx - radius, 0, W - 1)
    x2 = clamp(cx + radius, 0, W - 1)
    y1 = clamp(cy - radius, 0, H - 1)
    y2 = clamp(cy + radius, 0, H - 1)

    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return img

    roi = img[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]

    # grilla
    ys, xs = np.mgrid[0:rh, 0:rw].astype(np.float32)
    dx = xs - (cx - x1)
    dy = ys - (cy - y1)
    r = np.sqrt(dx * dx + dy * dy)
    rn = r / float(radius)

    # factor radial suave (solo dentro del radio)
    mask = rn < 1.0
    eps = 1e-6

    # Para bulge: r_src < r cerca del centro => estira (agranda)
    # Para pinch: r_src > r cerca del centro => comprime (encoge)
    # curva suave: (1 - rn^2)
    k = (1.0 - rn * rn)
    s = float(clamp(strength, -0.90, 0.90))
    r_src = r * (1.0 - s * k)

    scale = r_src / (r + eps)
    map_x = (cx - x1) + dx * scale
    map_y = (cy - y1) + dy * scale

    # fuera del radio: identidad
    map_x[~mask] = xs[~mask]
    map_y[~mask] = ys[~mask]

    warped = cv2.remap(roi, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    img[y1:y2, x1:x2] = warped
    return img

def filtro_ojos_grandes(frame, face, w, h):
    le = iris_center(face, "L", w, h)
    re = iris_center(face, "R", w, h)
    # radio basado en distancia entre ojos
    eye_span = dist(le, re)
    rad = int(eye_span * 0.18)
    out = frame.copy()
    out = bulge_pinch_roi(out, le, rad, strength=0.55)
    out = bulge_pinch_roi(out, re, rad, strength=0.55)
    return out

def filtro_boca_grande(frame, face, w, h):
    ml = lm_xy(face, LMS["mouth_left"], w, h)
    mr = lm_xy(face, LMS["mouth_right"], w, h)
    up = lm_xy(face, LMS["upper_lip"], w, h)
    lo = lm_xy(face, LMS["lower_lip"], w, h)

    mouth_w = dist(ml, mr)
    mouth_h = dist(up, lo)
    cx = int((ml[0] + mr[0]) / 2)
    cy = int((up[1] + lo[1]) / 2)

    rad = int(max(mouth_w, mouth_h) * 0.65)
    out = frame.copy()
    out = bulge_pinch_roi(out, (cx, cy), rad, strength=0.55)
    return out

def filtro_nariz_pequena(frame, face, w, h):
    nose = lm_xy(face, LMS["nose"], w, h)
    le = lm_xy(face, LMS["left_eye"], w, h)
    re = lm_xy(face, LMS["right_eye"], w, h)
    eye_dist = dist(le, re)
    rad = int(eye_dist * 0.20)
    out = frame.copy()
    # pinch (strength negativo) para “reducir”
    out = bulge_pinch_roi(out, nose, rad, strength=-0.45)
    return out

def filtro_cara_ancha(frame, face, w, h):
    fl = lm_xy(face, LMS["face_left"], w, h)
    fr = lm_xy(face, LMS["face_right"], w, h)
    face_w = dist(fl, fr)
    rad = int(face_w * 0.22)
    out = frame.copy()
    # bulge en mejillas
    out = bulge_pinch_roi(out, fl, rad, strength=0.50)
    out = bulge_pinch_roi(out, fr, rad, strength=0.50)
    return out

def filtro_cara_delgada(frame, face, w, h):
    fl = lm_xy(face, LMS["face_left"], w, h)
    fr = lm_xy(face, LMS["face_right"], w, h)
    face_w = dist(fl, fr)
    rad = int(face_w * 0.22)
    out = frame.copy()
    # pinch en mejillas
    out = bulge_pinch_roi(out, fl, rad, strength=-0.50)
    out = bulge_pinch_roi(out, fr, rad, strength=-0.50)
    return out

# =========================
# Selector de filtro activo
# =========================
def aplicar_filtro_activo(frame, face, w, h, idx):
    if idx == 0:
        return aplicar_sombrero(frame, face, w, h)
    elif idx == 1:
        return aplicar_gafas(frame, face, w, h)
    elif idx == 2:
        return aplicar_bigote(frame, face, w, h)
    elif idx == 3:
        return aplicar_orejas(frame, face, w, h)
    elif idx == 4:
        return filtro_escena(frame)
    elif idx == 5:
        return filtro_ojos_glow(frame, face, w, h)

    # deformaciones (5 nuevos)
    elif idx == 6:
        return filtro_ojos_grandes(frame, face, w, h)
    elif idx == 7:
        return filtro_boca_grande(frame, face, w, h)
    elif idx == 8:
        return filtro_nariz_pequena(frame, face, w, h)
    elif idx == 9:
        return filtro_cara_ancha(frame, face, w, h)
    elif idx == 10:
        return filtro_cara_delgada(frame, face, w, h)

    return frame

# =========================
# Tu detección de gesto vertical
# =========================
def change_filter(direction, current_index, max_index):
    if direction == "ARRIBA":
        return (current_index - 1 + max_index) % max_index
    elif direction == "ABAJO":
        return (current_index + 1) % max_index
    return current_index

def calcular_ratio_vertical(face, w, h):
    nose = face.landmark[LMS["nose"]]
    chin = face.landmark[LMS["chin"]]
    le = face.landmark[LMS["left_eye"]]
    re = face.landmark[LMS["right_eye"]]

    nose_y = nose.y * h
    chin_y = chin.y * h
    eyes_y = (le.y + re.y) / 2 * h

    face_altura = chin_y - eyes_y
    if face_altura <= 0:
        return None

    ratio = (nose_y - eyes_y) / face_altura
    return ratio

# =========================
# Main
# =========================
def main():
    global baseline_ratio, STATE, active_filter_index, last_change_time, frame_count, start_time

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("No se pudo abrir la cámara.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as face_mesh:

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            current_time = int(time.time() * 1000)
            h, w, _ = frame.shape

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            mensaje = "Pulsa 'c' mirando al frente para calibrar"
            detected_command = None

            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]

                ratio = calcular_ratio_vertical(face, w, h)
                if ratio is not None:
                    window.append(ratio)
                    ratio_suav = float(np.mean(window))

                    if baseline_ratio is not None:
                        delta = ratio_suav - baseline_ratio

                        if abs(delta) < THRESHOLD:
                            nuevo_estado = "NEUTRO"
                        elif delta < -THRESHOLD:
                            nuevo_estado = "ARRIBA"
                        else:
                            nuevo_estado = "ABAJO"

                        mensaje = f"Δ: {delta:.3f} (TH={THRESHOLD}, STATE: {nuevo_estado})"

                        if (
                            nuevo_estado != STATE
                            and nuevo_estado != "NEUTRO"
                            and (current_time - last_change_time) > DEBOUNCE_DELAY_MS
                        ):
                            detected_command = nuevo_estado
                            last_change_time = current_time

                        STATE = nuevo_estado
                    else:
                        mensaje = f"ratio sin calibrar: {ratio_suav:.3f}"

                if detected_command:
                    active_filter_index = change_filter(
                        detected_command, active_filter_index, len(FILTROS)
                    )

                # Aplica filtro actual
                frame = aplicar_filtro_activo(frame, face, w, h, active_filter_index)

                # UI
                cv2.putText(
                    frame,
                    f"FILTRO ACTIVO: {FILTROS[active_filter_index]}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            # FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0.0
            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (w - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            # Mensaje
            cv2.putText(
                frame,
                mensaje,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            # Aviso si faltan assets
            faltan = [k for k, v in ASSETS.items() if v is None]
            if faltan:
                cv2.putText(
                    frame,
                    f"Faltan PNGs en /assets: {', '.join(faltan)}",
                    (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Filtros tipo redes sociales (FaceMesh + OpenCV)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord("c") and results.multi_face_landmarks:
                ratio = calcular_ratio_vertical(face, w, h)
                if ratio is not None:
                    baseline_ratio = float(ratio)
                    window.clear()
                    window.append(baseline_ratio)
                    print(f"Calibrado. baseline_ratio = {baseline_ratio:.3f}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
