import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

ruta_src = os.path.join(os.getcwd(), 'src') 
sys.path.append(ruta_src)

from Preprocessing.Geometry_correction import aplicar_warp

BASE_DIR = Path(__file__).resolve().parent.parent.parent
image_path = BASE_DIR / "data" / "ticket10.jpeg"

img = cv2.imread(str(image_path))
assert img is not None, "Error llegint imatge"

def trobar_esquines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    configs = [(3, False), (7, True)]
    
    for diam, usar_clahe in configs:
        procesada = gray
        if usar_clahe:
            procesada = clahe.apply(procesada)
    
        blurred = cv2.bilateralFilter(procesada, diam, 75, 75)
        kernel_pre = np.ones((3,3), np.uint8)
        blurred = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel_pre)
        edged = cv2.Canny(blurred, 50, 150)

        k_size = int(img.shape[1] * 0.01) 
        if k_size % 2 == 0: k_size += 1
        kernel_close = np.ones((k_size, k_size), np.uint8)
        edged_closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel_close)

        kernel_open = np.ones((7,7), np.uint8)
        edged_final = cv2.dilate(edged_closed, kernel_open, iterations=1)

        cnts, _ = cv2.findContours(edged_final.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        for c in cnts:
            peri = cv2.arcLength(c, True)
            this_approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(this_approx) == 4:
                area = cv2.contourArea(c)
                if area > (img.shape[0] * img.shape[1] * 0.05):
                    return this_approx, edged
    
    return [], np.zeros_like(gray)

def ordenar_puntos(pts):
    # npts es un array de (4, 2)
    # 1. Encontrar el centro de los puntos
    center = np.mean(pts, axis=0)
    
    # 2. Calcular el ángulo de cada punto respecto al centro
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    
    # 3. Ordenar los puntos según el ángulo
    sorted_indices = np.argsort(angles)
    pts = pts[sorted_indices]
    
    rect = np.zeros((4, 2), dtype="float32")
    
    # Ordenar por X para separar izquierda de derecha
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_side = x_sorted[:2, :]
    right_side = x_sorted[2:, :]
    
    # En la izquierda, el que tenga Y menor es top-left, el otro bottom-left
    rect[0] = left_side[np.argmin(left_side[:, 1]), :]
    rect[3] = left_side[np.argmax(left_side[:, 1]), :]
    
    # En la derecha, el que tenga Y menor es top-right, el otro bottom-right
    rect[1] = right_side[np.argmin(right_side[:, 1]), :]
    rect[2] = right_side[np.argmax(right_side[:, 1]), :]
    
    return rect

approx, edged = trobar_esquines(img)

if len(approx) == 4:
    # 2. Quitar dimensiones extra: de (4, 1, 2) a (4, 2)
    pts = np.asarray(approx).reshape(4, 2)
    
    # 3. Ordenar: fundamental para que el warp no salga "retorcido"
    pts_ordenados = ordenar_puntos(pts.astype("float32"))

    img_puntos = img.copy()
    for p in approx:
        x, y = p[0]
        cv2.circle(img_puntos, (x, y), 20, (0, 255, 0), -1)

    ticket_corregido = aplicar_warp(img, pts_ordenados)
else:
    print("No se encontraron las 4 esquinas en ninguno de los intentos.")


fig, axes = plt.subplots(1, 4, figsize=(15, 5))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original")
axes[1].imshow(edged, cmap="gray")
axes[1].set_title("Bordes Detectados")
axes[2].imshow(cv2.cvtColor(img_puntos, cv2.COLOR_BGR2RGB))
axes[2].set_title("Resultado")
axes[3].imshow(cv2.cvtColor(ticket_corregido, cv2.COLOR_BGR2RGB))
axes[3].set_title("Warp Correcto")
for ax in axes: ax.axis("off")
plt.tight_layout()
plt.show()