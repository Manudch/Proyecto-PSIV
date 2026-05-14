# text_extractionOCR.py
import cv2
import numpy as np
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys

sys.path.append(str(Path(__file__).resolve().parent))

try:
    from Segmentation import segmentar
    HAS_SEGMENTATION = True
except Exception:
    HAS_SEGMENTATION = False

BASE_DIR = Path(__file__).resolve().parent.parent

OCR_ENGINE = None
USE_EASYOCR = False

try:
    import easyocr
    OCR_ENGINE = "easyocr"
    print("[INFO] Usando EasyOCR (español)")
    reader = easyocr.Reader(['es', 'en'], gpu=False)
except Exception as e:
    print(f"[WARN] EasyOCR no disponible: {e}")
    try:
        import pytesseract
        from pytesseract import Output

        TESSDATA_PATH = r"C:\Program Files\Tesseract-OCR\tessdata"
        for path in [r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                     r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]:
            if Path(path).exists():
                pytesseract.pytesseract.tesseract_cmd = path
                break

        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        import os
        os.environ['TESSDATA_PREFIX'] = TESSDATA_PATH

        pytesseract.get_tesseract_version()
        OCR_ENGINE = "tesseract"
        print("[INFO] Usando Tesseract OCR")
    except Exception as e2:
        raise RuntimeError("No se encontró OCR. Instala tesseract o easyocr.")


def preprocesar_ocr(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    h, w = gray.shape
    if h > 3000:
        scale = 1500 / h
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
    except Exception:
        gray_eq = gray
    
    blurred = cv2.GaussianBlur(gray_eq, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def ocr_bloque_tesseract(img_bloque):
    binary = preprocesar_ocr(img_bloque)
    config = "--psm 6"
    lang = 'spa' if Path(TESSDATA_PATH).joinpath('spa.traineddata').exists() else 'eng'
    datos = pytesseract.image_to_data(binary, lang=lang, config=config, output_type=Output.DICT)
    lineas_dict = {}
    for i in range(len(datos['text'])):
        texto = datos['text'][i].strip()
        conf = int(datos['conf'][i])
        if not texto or conf <= 30:
            continue
        y = datos['top'][i]
        x = datos['left'][i]
        y_clave = round(y / 10) * 10
        if y_clave not in lineas_dict:
            lineas_dict[y_clave] = []
        lineas_dict[y_clave].append((texto, x))
    lineas = []
    for y_clave in sorted(lineas_dict.keys()):
        palabras = sorted(lineas_dict[y_clave], key=lambda t: t[1])
        lineas.append(' '.join([p[0] for p in palabras]))
    return lineas


def ocr_bloque_easyocr(img_bloque):
    _, binary = cv2.threshold(cv2.cvtColor(img_bloque, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    result = reader.readtext(binary, detail=0)
    return list(result)


def ocr_bloque(img_bloque):
    if OCR_ENGINE == "tesseract":
        return ocr_bloque_tesseract(img_bloque)
    return ocr_bloque_easyocr(img_bloque)


def extraer_precio(texto):
    texto_limpio = texto.replace(' ', '').replace(',', '.')
    patrones = [
        r'(\d+\.\d{2})',
        r'(\d+,\d{2})',
        r'(\d+\.\d{3})',
    ]
    for patron in patrones:
        match = re.search(patron, texto_limpio)
        if match:
            precio = match.group(1).replace(',', '.')
            if '.' in precio:
                parte_entera, parte_decimal = precio.rsplit('.', 1)
                if len(parte_decimal) == 2:
                    return precio
    return None


def extraer_fecha(texto):
    patrones = [r'(\d{2}[/-]\d{2}[/-]\d{4})', r'(\d{2}[/-]\d{2}[/-]\d{2})']
    for patron in patrones:
        match = re.search(patron, texto)
        if match:
            return match.group(1)
    return None


def extraer_total(texto):
    texto_lower = texto.lower()
    if any(k in texto_lower for k in ['total', 'suma', 'importe', 'a pagar', 'importe']):
        precio = extraer_precio(texto)
        if precio:
            return precio
    return None


def procesar_linea_producto(linea):
    resultado = {'texto_raw': linea, 'nombre': None, 'precio': None}
    
    linea_limpia = re.sub(r'[^\w\s€.,]', '', linea)
    linea_limpia = re.sub(r'\s+', ' ', linea_limpia).strip()
    
    precio = extraer_precio(linea_limpia)
    
    if precio:
        resultado['precio'] = float(precio)
        
        precio_formats = [precio, precio.replace('.', ','), precio.replace('.', ' ')]
        idx = -1
        for pf in precio_formats:
            temp_idx = linea_limpia.rfind(pf)
            if temp_idx > idx:
                idx = temp_idx
        
        if idx > 0:
            nombre_raw = linea_limpia[:idx].strip()
            nombre_limpio = re.sub(r'[*_|‾\-–—]+$', '', nombre_raw)
            nombre_limpio = re.sub(r'^\d+\s+', '', nombre_limpio)
            if nombre_limpio and len(nombre_limpio) > 1:
                resultado['nombre'] = nombre_limpio
    
    return resultado


def procesar_ticket(image_path, guardar_ocr=True, debug=True):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"No existe: {image_path}")

    print(f"\n{'='*60}")
    print(f"PROCESANDO: {image_path.name}")
    print(f"{'='*60}\n")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"No se pudo cargar: {image_path}")

    try:
        if HAS_SEGMENTATION:
            img_seg, bloques, etiquetas = segmentar(str(image_path))
        else:
            raise Exception("Sin módulo de segmentación")
    except Exception as e:
        h, w = img.shape[:2]
        bloques = [(0, h // 3), (h // 3, 2 * h // 3), (2 * h // 3, h)]
        etiquetas = ['CABECERA', 'PRODUCTO', 'PIE']

    resultados = []
    datos_ticket = {'tienda': None, 'fecha': None, 'total': None, 'productos': []}
    todas_lineas = []

    for i, ((y0, y1), etiqueta) in enumerate(zip(bloques, etiquetas)):
        recorte = img[y0:y1, :]
        lineas = ocr_bloque(recorte)
        
        todas_lineas.extend([(l, etiqueta) for l in lineas])

        if guardar_ocr:
            output_dir = BASE_DIR / "data" / "bloques_ocr"
            output_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(output_dir / f"bloque_{i:02d}_{etiqueta}.jpeg"), recorte)

        bloque_info = {'bloque_id': i, 'tipo': etiqueta, 'y0': y0, 'y1': y1, 'lineas': lineas}
        resultados.append(bloque_info)

        if etiqueta == 'CABECERA' and lineas:
            for l in lineas:
                if l.strip() and not l.strip().isdigit():
                    datos_ticket['tienda'] = l.strip()
                    break
        elif etiqueta in ['PRODUCTO', 'SEPARADOR']:
            for linea in lineas:
                if not datos_ticket['fecha']:
                    datos_ticket['fecha'] = extraer_fecha(linea)
                if not datos_ticket['total']:
                    datos_ticket['total'] = extraer_total(linea)
                prod = procesar_linea_producto(linea)
                if prod['precio'] and prod['precio'] > 0 and prod['precio'] < 1000:
                    if prod not in datos_ticket['productos']:
                        datos_ticket['productos'].append(prod)
        elif etiqueta == 'PIE':
            for linea in lineas:
                if not datos_ticket['fecha']:
                    datos_ticket['fecha'] = extraer_fecha(linea)
                if not datos_ticket['total']:
                    datos_ticket['total'] = extraer_total(linea)

    productos_encontrados = {}
    for linea, _ in todas_lineas:
        precio = extraer_precio(linea)
        if precio:
            try:
                precio_float = float(precio)
                if 0.01 < precio_float < 500:
                    nombre_candidato = re.sub(r'[^\w\s]', '', linea.replace(precio, '').replace(',', '.')).strip()
                    nombre_candidato = re.sub(r'\s+', ' ', nombre_candidato)
                    nombre_candidato = re.sub(r'^(IVA|Total|Importe|TOTAL|Subtotal|Visa|CajaBank|Tarjeta)\s*', '', nombre_candidato, flags=re.IGNORECASE)
                    
                    if nombre_candidato and len(nombre_candidato) > 1:
                        if precio_float not in productos_encontrados or len(nombre_candidato) > len(productos_encontrados.get(precio_float, '')):
                            productos_encontrados[precio_float] = nombre_candidato
            except (ValueError, AttributeError):
                pass
    
    for precio, nombre in productos_encontrados.items():
        datos_ticket['productos'].append({
            'texto_raw': nombre,
            'nombre': nombre,
            'precio': precio
        })
    
    datos_ticket['productos'].sort(key=lambda x: x['precio'] if x['precio'] else 0)
    
    seen = set()
    productos_unicos = []
    for p in datos_ticket['productos']:
        key = (p.get('nombre'), p.get('precio'))
        if key not in seen:
            seen.add(key)
            productos_unicos.append(p)
    datos_ticket['productos'] = productos_unicos

    if debug:
        print(f"{'='*60}")
        print("RESULTADOS OCR")
        print(f"{'='*60}")

        if datos_ticket.get('tienda'):
            print(f"\nTIENDA: {datos_ticket['tienda']}")
        if datos_ticket.get('fecha'):
            print(f"FECHA: {datos_ticket['fecha']}")
        if datos_ticket.get('total'):
            print(f"TOTAL: {datos_ticket['total']} EUR")

        if datos_ticket.get('productos'):
            print(f"\n{'-'*60}")
            print(f"PRODUCTOS ({len(datos_ticket['productos'])})")
            print(f"{'-'*60}")
            for j, p in enumerate(datos_ticket['productos'], 1):
                nombre = p['nombre'] or '(sin nombre)'
                precio = f"{p['precio']:.2f} EUR" if p['precio'] else '?'
                print(f"  {j:2d}. {nombre[:40]:<40s} {precio:>10s}")

        print(f"\n{'-'*60}")
        print("BLOQUES")
        print(f"{'-'*60}")
        for r in resultados:
            print(f"\n[{r['tipo']}] y:{r['y0']}-{r['y1']}")
            for linea in r['lineas']:
                print(f"   {linea}")

    return resultados, datos_ticket


def exportar_json(datos_ticket, output_path=None):
    if output_path is None:
        output_path = BASE_DIR / "data" / "resultado.json"
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(datos_ticket, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Guardado en: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('imagen', nargs='?', default=str(BASE_DIR / "data" / "ticket-2-recortado.jpeg"))
    parser.add_argument('--export', '-e', action='store_true')
    args = parser.parse_args()

    try:
        resultados, datos = procesar_ticket(args.imagen)
        if args.export:
            exportar_json(datos)
        print("\n[OK] Completado")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)