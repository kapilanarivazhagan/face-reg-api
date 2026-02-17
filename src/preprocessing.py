import numpy as np
import cv2

def gamma_correct(img, gamma):
    table = np.array(
        [(i / 255.0) ** (1 / gamma) * 255 for i in range(256)]
    ).astype("uint8")
    return cv2.LUT(img, table)

def generate_variants(img):
    return {
        "original": img,
        "brightened": gamma_correct(img, 0.7),
        "darkened": gamma_correct(img, 1.3)
    }
