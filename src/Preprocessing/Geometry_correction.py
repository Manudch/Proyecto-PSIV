import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def aplicar_warp(img, pts):
    # pts ya viene de 'ordenar_puntos', así que sabemos que:
    # pts[0] = Top-Left, pts[1] = Top-Right, pts[2] = Bottom-Right, pts[3] = Bottom-Left
    (tl, tr, br, bl) = pts

    # --- CALCULO DEL ANCHO ---
    # Medimos la distancia entre esquinas superiores y entre inferiores
    ancho_sup = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    ancho_inf = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    # Nos quedamos con el ancho más grande para no perder info
    max_ancho = max(int(ancho_sup), int(ancho_inf))

    # --- CALCULO DEL ALTO ---
    # Medimos la distancia entre esquinas izquierdas y entre derechas
    alto_izq = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    alto_der = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    max_alto = max(int(alto_izq), int(alto_der))

    # En lugar de 300x300, usamos las medidas reales que acabamos de calcular
    pts_destino = np.array([
        [0, 0],
        [max_ancho - 1, 0],
        [max_ancho - 1, max_alto - 1],
        [0, max_alto - 1]
        ], dtype=np.float32)

    # Matriz de transformación
    M = cv2.getPerspectiveTransform(pts, pts_destino)
    
    # Warp final
    return cv2.warpPerspective(img, M, (max_ancho, max_alto))


