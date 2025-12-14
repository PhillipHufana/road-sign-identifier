import cv2
import numpy as np

def enhance_clahe(frame_bgr):
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    out = cv2.merge([l2, a, b])
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

def restore_unsharp(frame_bgr, sigma=1.2, amount=1.6):
    blur = cv2.GaussianBlur(frame_bgr, (0, 0), sigma)
    return cv2.addWeighted(frame_bgr, amount, blur, -(amount - 1.0), 0)

def restore_denoise(frame_bgr):
    return cv2.fastNlMeansDenoisingColored(frame_bgr, None, 5, 5, 7, 21)

def apply_gamma(frame_bgr, gamma, lut_cache):
    g = float(gamma)
    if g <= 0.01:
        g = 0.01
    if lut_cache.get("gamma") != g:
        inv = 1.0 / g
        lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
        lut_cache["gamma"] = g
        lut_cache["lut"] = lut
    return cv2.LUT(frame_bgr, lut_cache["lut"])
