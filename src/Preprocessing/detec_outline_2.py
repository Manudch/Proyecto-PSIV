import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
from Geometry_correction import aplicar_warp

ruta_src = os.path.join(os.getcwd(), 'src')
sys.path.append(ruta_src)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
image_path = BASE_DIR / "data" / "ticket13.jpeg"

img = cv2.imread(str(image_path))
assert img is not None, "Error llegint imatge"

IMG_H, IMG_W = img.shape[:2]
IMG_AREA = IMG_H * IMG_W

# ── Utilidades ────────────────────────────────────────────────────

def ordenar_puntos(pts):
    pts = np.asarray(pts).reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    x_sorted = pts[np.argsort(pts[:, 0])]
    left, right = x_sorted[:2], x_sorted[2:]
    rect[0] = left[np.argmin(left[:, 1])]    # top-left
    rect[3] = left[np.argmax(left[:, 1])]    # bottom-left
    rect[1] = right[np.argmin(right[:, 1])]  # top-right
    rect[2] = right[np.argmax(right[:, 1])]  # bottom-right
    return rect

def es_contorno_valido(approx, img_area, min_ratio=0.05, max_ratio=0.98):
    if len(approx) != 4:
        return False
    area = cv2.contourArea(approx)
    return min_ratio < (area / img_area) < max_ratio

def es_cuadrilatero_razonable(pts):
    pts = pts.reshape(4, 2)
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        p3 = pts[(i + 2) % 4]
        v1 = p1 - p2
        v2 = p3 - p2
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
        if angle < 40 or angle > 140:
            return False
    return True

# ── NUEVO: Segmentación del ticket por color/luminancia ──────────
# La mesa de madera es marrón oscuro; el ticket es papel blanco/gris claro.
# Trabajamos en LAB para separar luminancia de color.

def segmentar_ticket(img):
    """
    Devuelve una máscara binaria donde el ticket (papel claro)
    está en blanco y el fondo (mesa) en negro.
    Robusto ante sombras porque normaliza la luminancia localmente.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # CLAHE sobre canal L para igualar iluminación con sombras
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    L_eq  = clahe.apply(L)

    # Otsu sobre L ecualizado — separa papel claro de mesa oscura
    _, mask = cv2.threshold(L_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morfología para rellenar huecos del texto y sombras internas
    k_fill = max(15, int(IMG_W * 0.03))
    kernel  = np.ones((k_fill, k_fill), np.uint8)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Eliminar ruido pequeño (reflejos, esquinas de mesa)
    k_open = max(7, int(IMG_W * 0.01))
    kernel2 = np.ones((k_open, k_open), np.uint8)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2)

    return mask

def contorno_desde_mascara(mask):
    """Extrae el contorno exterior más grande de la máscara."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)

def aproximar_a_4_lados(contour):
    """Intenta reducir un contorno a 4 esquinas con varias tolerancias."""
    hull = cv2.convexHull(contour)
    for eps in [0.02, 0.03, 0.04, 0.05, 0.07, 0.10]:
        peri   = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if (len(approx) == 4 and
                es_contorno_valido(approx, IMG_AREA) and
                es_cuadrilatero_razonable(approx)):
            return approx
    return None

# ── Pipeline de bordes (fallback si la máscara falla) ─────────────

def build_edge_map(gray, strategy):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if strategy == 0:
        blurred = cv2.bilateralFilter(gray, 3, 75, 75)
        edged   = cv2.Canny(blurred, 50, 150)
    elif strategy == 1:
        eq      = clahe.apply(gray)
        blurred = cv2.bilateralFilter(eq, 7, 75, 75)
        edged   = cv2.Canny(blurred, 30, 120)
    elif strategy == 2:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        kernel  = np.ones((3, 3), np.uint8)
        grad    = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel)
        edged   = cv2.Canny(grad, 40, 130)
    elif strategy == 3:
        eq      = clahe.apply(gray)
        blurred = cv2.GaussianBlur(eq, (7, 7), 0)
        thresh  = cv2.adaptiveThreshold(blurred, 255,
                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                      cv2.THRESH_BINARY_INV, 11, 2)
        edged   = cv2.Canny(thresh, 30, 100)
    else:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edged   = cv2.Canny(otsu, 50, 150)

    k = max(3, int(gray.shape[1] * 0.01))
    if k % 2 == 0: k += 1
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    edged = cv2.dilate(edged, np.ones((5, 5), np.uint8), iterations=1)
    return edged

def trobar_esquines_bordes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for strategy in range(5):
        edged = build_edge_map(gray, strategy)
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
        for c in cnts:
            for eps in [0.02, 0.03, 0.05]:
                peri   = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, eps * peri, True)
                if es_contorno_valido(approx, IMG_AREA) and es_cuadrilatero_razonable(approx):
                    return approx, edged
        if cnts:
            hull   = cv2.convexHull(cnts[0])
            peri   = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.04 * peri, True)
            if es_contorno_valido(approx, IMG_AREA) and es_cuadrilatero_razonable(approx):
                return approx, edged
    h, w = img.shape[:2]
    return np.array([[[0,0]],[[w-1,0]],[[w-1,h-1]],[[0,h-1]]]), np.zeros((h,w), np.uint8)

# ── Main: primero máscara, luego fallback por bordes ──────────────

approx = None
debug_mask = None

# INTENTO 1: segmentación por color/luminancia (robusto ante sombras)
mask = segmentar_ticket(img)
debug_mask = mask
contorno = contorno_desde_mascara(mask)

if contorno is not None and cv2.contourArea(contorno) > IMG_AREA * 0.05:
    approx = aproximar_a_4_lados(contorno)
    edged   = mask  # para visualización
    if approx is not None:
        print("[OK] Esquinas encontradas via segmentación LAB")

# INTENTO 2: fallback por detección de bordes clásica
if approx is None:
    print("[INFO] Fallback: detección por bordes")
    approx, edged = trobar_esquines_bordes(img)
else:
    edged = mask

# ── Resultado ─────────────────────────────────────────────────────

pts           = np.asarray(approx).reshape(4, 2)
pts_ordenados = ordenar_puntos(pts.astype("float32"))

img_puntos = img.copy()
for p in approx:
    x, y = p[0]
    cv2.circle(img_puntos, (x, y), 20, (0, 255, 0), -1)

ticket_corregido = aplicar_warp(img, pts_ordenados)

# ── Visualización ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(15, 5))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB));              axes[0].set_title("Original")
axes[1].imshow(edged, cmap="gray");                                axes[1].set_title("Segmentación / Bordes")
axes[2].imshow(cv2.cvtColor(img_puntos, cv2.COLOR_BGR2RGB));       axes[2].set_title("Esquinas")
axes[3].imshow(cv2.cvtColor(ticket_corregido, cv2.COLOR_BGR2RGB)); axes[3].set_title("Warp Final")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()