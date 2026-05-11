# Segmentation.py - DETECTAR SOLO LÍNEAS DE PRODUCTOS
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
image_path = BASE_DIR / "data" / "ticket20.jpg"

img = cv2.imread(str(image_path))
assert img is not None, "Error llegint imatge"

total_h = img.shape[0]

# ── PASO 1: Preprocesar ──────────────────────────────────────────
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray_eq = clahe.apply(gray)
blurred = cv2.GaussianBlur(gray_eq, (5, 5), 0)
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# ── PASO 2: Encontrar ZONA DE PRODUCTOS por densidad de texto ────
# La zona de productos es donde hay más líneas de texto seguidas
h_projection_full = np.sum(binary, axis=1)
kernel_smooth_full = np.ones(50) / 50  # suavizado fuerte para ver zonas grandes
h_smooth_full = np.convolve(h_projection_full, kernel_smooth_full, mode='same')

# Encontrar la zona con mayor densidad de texto (productos)
# Buscar en el rango 20%-70% de la altura (donde suelen estar los productos)
zona_inicio = int(total_h * 0.20)
zona_fin = int(total_h * 0.70)

# Dentro de esa zona, encontrar dónde hay texto denso
max_densidad = 0
mejor_inicio = zona_inicio
mejor_fin = zona_fin

# Buscar ventana de 200px con máxima densidad
ventana = 200
for y in range(zona_inicio, zona_fin - ventana, 20):
    densidad = np.sum(h_smooth_full[y:y+ventana])
    if densidad > max_densidad:
        max_densidad = densidad
        mejor_inicio = y
        mejor_fin = y + ventana

# Expandir un poco la zona encontrada
y_productos_inicio = max(0, mejor_inicio - 30)
y_productos_fin = min(total_h, mejor_fin + 50)

print(f"[INFO] Zona de productos: Y={y_productos_inicio} a Y={y_productos_fin}")

# ── PASO 3: Proyección horizontal SOLO en zona de productos ─────
binary_productos = binary[y_productos_inicio:y_productos_fin, :]
h_projection = np.sum(binary_productos, axis=1)

# Suavizado SUAVE para no unir líneas diferentes
kernel_smooth = np.ones(3) / 3  # muy pequeño para mantener líneas separadas
h_smooth = np.convolve(h_projection, kernel_smooth, mode='same')

# ── PASO 4: Detectar cada línea de producto individual ──────────
umbral = h_smooth.max() * 0.15  # más alto = más estricto, separa mejor líneas
en_bloque = h_smooth > umbral

bloques = []
inicio = None

for i, activo in enumerate(en_bloque):
    if activo and inicio is None:
        inicio = i
    elif not activo and inicio is not None:
        altura = i - inicio
        # Productos suelen tener entre 15-50 píxeles de alto
        if 15 < altura < 60:
            # Ajustar coordenadas a imagen original
            bloques.append((y_productos_inicio + inicio, y_productos_inicio + i))
        inicio = None

if inicio is not None:
    altura = len(en_bloque) - inicio
    if 15 < altura < 60:
        bloques.append((y_productos_inicio + inicio, y_productos_inicio + len(en_bloque)))

print(f"[INFO] Líneas de productos detectadas: {len(bloques)}")

# ── PASO 5: Visualización ────────────────────────────────────────
img_viz = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()

# Dibujar zona de productos (rectángulo amarillo punteado)
cv2.rectangle(img_viz, (0, y_productos_inicio), (img.shape[1], y_productos_fin), (255, 255, 0), 3)

# Dibujar cada línea de producto
for idx, (y0, y1) in enumerate(bloques):
    overlay = img_viz.copy()
    cv2.rectangle(overlay, (0, y0), (img.shape[1], y1), (0, 255, 0), -1)
    img_viz = cv2.addWeighted(overlay, 0.3, img_viz, 0.7, 0)
    cv2.rectangle(img_viz, (0, y0), (img.shape[1], y1), (0, 200, 0), 2)
    cv2.putText(img_viz, f"PROD {idx+1}", (10, y0 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 2)

fig, axes = plt.subplots(1, 3, figsize=(18, 10))

axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("Ticket corregido")
axes[0].axis("off")

axes[1].plot(h_smooth, np.arange(len(h_smooth)), color='steelblue')
axes[1].invert_yaxis()
axes[1].set_title("Proyección (zona productos)")
axes[1].set_xlabel("Suma píxeles por fila")
axes[1].axvline(x=umbral, color='red', linestyle='--', label='umbral')
for y0, y1 in [(b[0]-y_productos_inicio, b[1]-y_productos_inicio) for b in bloques]:
    axes[1].axhspan(y0, y1, alpha=0.3, color='green')
axes[1].legend()

axes[2].imshow(img_viz)
axes[2].set_title(f"Productos: {len(bloques)} líneas")
axes[2].axis("off")

plt.tight_layout()
plt.show()

# ── Exportar SOLO productos ───────────────────────────────────────
output_dir = BASE_DIR / "data" / "productos"
output_dir.mkdir(exist_ok=True)

for i, (y0, y1) in enumerate(bloques):
    recorte = img[y0:y1, :]
    nombre = f"producto_{i:02d}.jpeg"
    cv2.imwrite(str(output_dir / nombre), recorte)

print(f"[OK] {len(bloques)} productos guardados en: {output_dir}")