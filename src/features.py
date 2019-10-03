import cv2
import numpy as np
from skimage import feature


def get_otsu_canny(gray_image, sigma=0.33):
    v = np.median(gray_image)

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(gray_image, lower, upper)

    return edged


def preprocess_image(image):
    image = cv2.resize(image, (32, 32))
    if image.shape[-1] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.shape[-1] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = image.copy()

    return get_otsu_canny(gray)
