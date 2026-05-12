# Segmentation.py
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def segmentar(image_input):
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise FileNotFoundError(f"No se encontró imagen: {image_input}")
    else:
        img = image_input

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h_projection = np.sum(binary, axis=1)
    kernel_smooth = np.ones(5) / 5
    h_smooth = np.convolve(h_projection, kernel_smooth, mode='same')
    umbral = h_smooth.max() * 0.05
    en_bloque = h_smooth > umbral

    bloques = []
    inicio = None
    for i, activo in enumerate(en_bloque):
        if activo and inicio is None:
            inicio = i
        elif not activo and inicio is not None:
            if (i - inicio) > 5:
                bloques.append((inicio, i))
            inicio = None
    if inicio is not None:
        bloques.append((inicio, len(en_bloque)))

    def clasificar_bloque(y_start, y_end, total_h, idx, n_total):
        centro = (y_start + y_end) / 2 / total_h
        if idx == 0:
            return "CABECERA"
        elif idx == n_total - 1:
            return "PIE"
        elif y_end - y_start > total_h * 0.02 and centro > 0.3 and centro < 0.85:
            return "PRODUCTO"
        else:
            return "SEPARADOR"

    etiquetas = [clasificar_bloque(y0, y1, img.shape[0], i, len(bloques)) for i, (y0, y1) in enumerate(bloques)]

    return img, bloques, etiquetas


def guardar_bloques(img, bloques, etiquetas, output_dir=None):
    if output_dir is None:
        output_dir = BASE_DIR / "data" / "bloques"
    output_dir.mkdir(exist_ok=True)
    for i, ((y0, y1), etiqueta) in enumerate(zip(bloques, etiquetas)):
        recorte = img[y0:y1, :]
        nombre = f"bloque_{i:02d}_{etiqueta.replace(' / ', '_')}.jpeg"
        cv2.imwrite(str(output_dir / nombre), recorte)
    return output_dir


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    image_path = BASE_DIR / "data" / "ticket-2-recortado.jpeg"
    img, bloques, etiquetas = segmentar(image_path)

    print(f"[INFO] Bloques detectados: {len(bloques)}")

    COLORES = {"CABECERA": (255, 100, 100), "PRODUCTO": (100, 200, 100),
               "SEPARADOR": (100, 100, 255), "PIE": (200, 200, 100)}

    img_bloques = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    for (y0, y1), etiqueta in zip(bloques, etiquetas):
        color = COLORES.get(etiqueta, (200, 200, 200))
        overlay = img_bloques.copy()
        cv2.rectangle(overlay, (0, y0), (img.shape[1], y1), color, -1)
        img_bloques = cv2.addWeighted(overlay, 0.25, img_bloques, 0.75, 0)
        cv2.rectangle(img_bloques, (0, y0), (img.shape[1], y1), color, 2)
        cv2.putText(img_bloques, etiqueta, (10, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h_projection = np.sum(binary, axis=1)
    kernel_smooth = np.ones(5) / 5
    h_smooth = np.convolve(h_projection, kernel_smooth, mode='same')
    umbral = h_smooth.max() * 0.05

    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Ticket corregido")
    axes[0].axis("off")
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

    guardar_bloques(img, bloques, etiquetas)
    print(f"[OK] Bloques guardados en: {output_dir}")