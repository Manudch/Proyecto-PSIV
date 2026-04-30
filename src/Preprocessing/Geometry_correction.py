import cv2
import numpy as np
import matplotlib.pyplot as plt


image_path = "data/ticket-2.jpeg"
image = cv2.imread(image_path)

image_canvas = image.copy()  
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, 
                        minLineLength=100, maxLineGap=10)


if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_canvas, (x1, y1), (x2, y2), (0, 255, 0), 3)


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 2)
plt.title("Canny Edges")
plt.imshow(edges, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Hough Lines")
plt.imshow(cv2.cvtColor(image_canvas, cv2.COLOR_BGR2RGB))

plt.show()
