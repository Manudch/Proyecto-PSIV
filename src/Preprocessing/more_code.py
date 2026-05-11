import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
image_path = BASE_DIR / "data" / "ticket1.jpg"

image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

image_canvas = image.copy()

# ─────────────────────────────────────────────
# PASO 1: Gris + Bilateral 
# ─────────────────────────────────────────────
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.bilateralFilter(image_gray, 9, 75, 75)

# ─────────────────────────────────────────────
# PASO 2: Umbral Adaptativo (como tenías tú)
# pero invertido: el ticket blanco → negro para findContours
# ─────────────────────────────────────────────
thresh = cv2.adaptiveThreshold(
    blurred, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11, 2
)

# Invertir: queremos el ticket como región BLANCA (foreground)
thresh_inv = cv2.bitwise_not(thresh)

# Morfología agresiva para cerrar todo el texto interno del ticket
# y convertirlo en un blob sólido
kernel_close = np.ones((25, 25), np.uint8)
thresh_solid = cv2.morphologyEx(thresh_inv, cv2.MORPH_CLOSE, kernel_close)

# Dilatar un poco para unir bordes rotos
kernel_dilate = np.ones((10, 10), np.uint8)
thresh_solid = cv2.dilate(thresh_solid, kernel_dilate, iterations=2)

# ─────────────────────────────────────────────
# PASO 3: Canny sobre el blob sólido
# ─────────────────────────────────────────────
edges = cv2.Canny(thresh_solid, 75, 200)
kernel_edge = np.ones((5, 5), np.uint8)
edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_edge)

# ─────────────────────────────────────────────
# PASO 4: Contornos — buscar el ticket
# ─────────────────────────────────────────────
contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

ticket_contour = None
for c in contours:
    peri = cv2.arcLength(c, True)
    # Tolerancia más alta que tu 0.02 original
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    if len(approx) == 4:
        ticket_contour = approx
        break

# Fallback: bounding rect del contorno más grande
if ticket_contour is None and contours:
    print("[WARN] Fallback: usando bounding rect")
    x, y, w, h = cv2.boundingRect(contours[0])
    ticket_contour = np.array([
        [[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]
    ])

# ─────────────────────────────────────────────
# PASO 5: Perspective Transform (recorte limpio)
# ─────────────────────────────────────────────
def order_points(pts):
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]    # top-left
    rect[2] = pts[np.argmax(s)]    # bottom-right
    rect[1] = pts[np.argmin(diff)] # top-right
    rect[3] = pts[np.argmax(diff)] # bottom-left
    return rect

def four_point_transform(img, pts):
    rect = order_points(pts)
    tl, tr, br, bl = rect
    max_w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    max_h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.array([[0, 0], [max_w-1, 0], [max_w-1, max_h-1], [0, max_h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (max_w, max_h))

# Dibujar contorno y esquinas
if ticket_contour is not None:
    cv2.drawContours(image_canvas, [ticket_contour], -1, (0, 255, 0), 10)
    for pt in ticket_contour:
        x, y = pt[0]
        cv2.circle(image_canvas, (x, y), 20, (255, 0, 0), -1)

warped = None
if ticket_contour is not None:
    warped = four_point_transform(image, ticket_contour)

# ─────────────────────────────────────────────
# Visualización (tu formato original: 1x4)
# ─────────────────────────────────────────────
n = 5 if warped is not None else 4
plt.figure(figsize=(20, 6))

plt.subplot(1, n, 1); plt.title("Gris + Filtro")
plt.imshow(image_gray, cmap='gray'); plt.axis('off')

plt.subplot(1, n, 2); plt.title("Umbral Adaptativo")
plt.imshow(thresh, cmap='gray'); plt.axis('off')

plt.subplot(1, n, 3); plt.title("Bordes (Canny)")
plt.imshow(edges_closed, cmap='gray'); plt.axis('off')

plt.subplot(1, n, 4); plt.title("Ticket Detectado")
plt.imshow(cv2.cvtColor(image_canvas, cv2.COLOR_BGR2RGB)); plt.axis('off')

if warped is not None:
    plt.subplot(1, n, 5); plt.title("Ticket Recortado")
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)); plt.axis('off')

plt.tight_layout()
plt.show()

if warped is not None:
    out = BASE_DIR / "data" / "ticket-2-recortado.jpeg"
    cv2.imwrite(str(out), warped)
    print(f"Guardado: {out}")