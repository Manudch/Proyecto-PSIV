# Segmentation.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
image_path = BASE_DIR / "data" / "ticket-2-recortado.jpeg"  # ticket ya corregido

img = cv2.imread(str(image_path))
assert img is not None, "Error llegint imatge"

# ── PASO 1: Preprocesar para binarizar limpiamente ────────────────
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray_eq = clahe.apply(gray)

blurred = cv2.GaussianBlur(gray_eq, (5, 5), 0)
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# THRESH_BINARY_INV → texto en BLANCO, fondo en NEGRO (necesario para proyección)

# ── PASO 2: Proyección horizontal (suma de píxeles por fila) ──────
# Cada fila con texto tendrá suma alta; filas vacías tendrán suma ~0
h_projection = np.sum(binary, axis=1)  # shape: (altura,)

# Suavizar la proyección para evitar cortes en medio de letras
kernel_smooth = np.ones(5) / 5
h_smooth = np.convolve(h_projection, kernel_smooth, mode='same')

# ── PASO 3: Detectar bloques (zonas con texto) ────────────────────
umbral = h_smooth.max() * 0.05  # filas con al menos 5% del máximo
en_bloque = h_smooth > umbral

bloques = []
inicio = None

for i, activo in enumerate(en_bloque):
    if activo and inicio is None:
        inicio = i
    elif not activo and inicio is not None:
        # Filtrar bloques demasiado pequeños (ruido)
        if (i - inicio) > 5:
            bloques.append((inicio, i))
        inicio = None

if inicio is not None:
    bloques.append((inicio, len(en_bloque)))

print(f"[INFO] Bloques detectados: {len(bloques)}")

# ── PASO 4: Clasificar bloques por posición relativa ──────────────
# Heurística simple basada en posición vertical
total_h = img.shape[0]

def clasificar_bloque(y_start, y_end, total_h, idx, n_total):
    centro = (y_start + y_end) / 2 / total_h  # posición relativa 0-1
    alto   = y_end - y_start

    if idx == 0:
        return "CABECERA"
    elif idx == n_total - 1:
        return "PIE"
    elif alto > total_h * 0.02 and centro > 0.3 and centro < 0.85:
        return "PRODUCTO"
    else:
        return "SEPARADOR / TOTAL"

etiquetas = [
    clasificar_bloque(y0, y1, total_h, i, len(bloques))
    for i, (y0, y1) in enumerate(bloques)
]

# ── PASO 5: Visualización ─────────────────────────────────────────
COLORES = {
    "CABECERA":          (255, 100, 100),
    "PRODUCTO":          (100, 200, 100),
    "SEPARADOR / TOTAL": (100, 100, 255),
    "PIE":               (200, 200, 100),
}

img_bloques = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()

for (y0, y1), etiqueta in zip(bloques, etiquetas):
    color = COLORES.get(etiqueta, (200, 200, 200))
    # Rectángulo semitransparente
    overlay = img_bloques.copy()
    cv2.rectangle(overlay, (0, y0), (img.shape[1], y1), color, -1)
    img_bloques = cv2.addWeighted(overlay, 0.25, img_bloques, 0.75, 0)
    # Borde y etiqueta
    cv2.rectangle(img_bloques, (0, y0), (img.shape[1], y1), color, 2)
    cv2.putText(img_bloques, etiqueta, (10, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

fig, axes = plt.subplots(1, 3, figsize=(16, 8))

axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("Ticket corregido")
axes[0].axis("off")

# Proyección horizontal
axes[1].plot(h_smooth, np.arange(len(h_smooth)), color='steelblue')
axes[1].invert_yaxis()
axes[1].set_title("Proyección horizontal")
axes[1].set_xlabel("Suma píxeles por fila")
axes[1].axvline(x=umbral, color='red', linestyle='--', label='umbral')
for y0, y1 in bloques:
    axes[1].axhspan(y0, y1, alpha=0.2, color='green')
axes[1].legend()

axes[2].imshow(img_bloques)
axes[2].set_title(f"Bloques detectados ({len(bloques)})")
axes[2].axis("off")

plt.tight_layout()
plt.show()

# ── Exportar bloques recortados (para pasarlos a OCR) ─────────────
output_dir = BASE_DIR / "data" / "bloques"
output_dir.mkdir(exist_ok=True)

for i, ((y0, y1), etiqueta) in enumerate(zip(bloques, etiquetas)):
    recorte = img[y0:y1, :]
    nombre  = f"bloque_{i:02d}_{etiqueta.replace(' / ', '_')}.jpeg"
    cv2.imwrite(str(output_dir / nombre), recorte)

print(f"[OK] Bloques guardados en: {output_dir}")