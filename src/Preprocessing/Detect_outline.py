import cv2
import numpy as np
import matplotlib.pyplot as plt

# carga de datos
image_path = "../data/ticket-2.jpeg"
image = cv2.imread(image_path)

image_canvas = image.copy()

# gris
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# blur
blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)

# edges (Canny)
edges = cv2.Canny(blurred, 50, 150)

kernel = np.ones((5, 5), np.uint8)

edges_closed = cv2.morphologyEx(
    edges,
    cv2.MORPH_CLOSE,
    kernel
)

kernel = np.ones((3, 3), np.uint8)

edges_open = cv2.morphologyEx(
    edges_closed,
    cv2.MORPH_OPEN,
    kernel
)

image_outline = cv2.subtract(edges_closed, edges_open)
# Hough Lines (MEJOR sobre edges_closed)
lines = cv2.HoughLinesP(
    image_outline,
    1,
    np.pi / 180,
    threshold=50,
    minLineLength=100,
    maxLineGap=500)




# dibujar líneas
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_canvas, (x1, y1), (x2, y2), (0, 255, 0), 3)

# mostrar resultados
plt.figure(figsize=(15, 6))

plt.subplot(1, 4, 1)
plt.title("Original")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 4, 2)
plt.title("Canny")
plt.imshow(edges, cmap='gray')

plt.subplot(1, 4, 3)
plt.title("Closing (morfología)")
plt.imshow(image_outline, cmap='gray')

plt.subplot(1, 4, 4)
plt.title("Hough Lines")
plt.imshow(cv2.cvtColor(image_canvas, cv2.COLOR_BGR2RGB))

plt.show()