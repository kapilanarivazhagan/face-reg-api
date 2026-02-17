import cv2
import numpy as np
from src.config import (
    BLUR_THRESHOLD,
    LIGHT_STD_THRESHOLD,
    EMBEDDING_SPREAD_THRESHOLD
)

def blur_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def lighting_stats(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return np.mean(gray), np.std(gray)

def evaluate_liveness(img, embedding_spread):
    flags = []
    score = 1.0

    blur = blur_score(img)
    _, std_light = lighting_stats(img)

    if blur < BLUR_THRESHOLD:
        flags.append("LOW_SHARPNESS")
        score *= 0.7

    if std_light < LIGHT_STD_THRESHOLD:
        flags.append("FLAT_LIGHTING")
        score *= 0.75

    if embedding_spread > EMBEDDING_SPREAD_THRESHOLD:
        flags.append("EMBEDDING_INCONSISTENT")
        score *= 0.8

    return round(score, 2), flags
