import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent.parent 
image_path = BASE_DIR / "data" / "ticket-2.jpeg"

img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
rows, cols = img.shape

pts1 = np.array([[50,100],[200,50],[50,200]], dtype=np.float32) 
pts2 = np.array([[10,100],[200,50],[100,250]], dtype=np.float32) 

M = cv2.getAffineTransform(pts1, pts2)

dst = cv2.warpAffine(img, M,(cols,rows))

plt.subplot(121),plt.imshow(img),plt.title('Input')

plt.subplot(122),plt.imshow(dst),plt.title('Output')

plt.show()
