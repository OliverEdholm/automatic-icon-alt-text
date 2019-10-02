import cv2
import numpy as np
from skimage import feature


def _preprocess_image(image):
    image = cv2.resize(image, (64, 64))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray


def get_hog_feature(image):
    image = image.copy()

    preprocessed = _preprocess_image(image)

    hog_feature = feature.hog(
        preprocessed, 
        orientations=6,
        pixels_per_cell=(4,4),
        cells_per_block=(4,4),
        transform_sqrt=True
    )

    return hog_feature
