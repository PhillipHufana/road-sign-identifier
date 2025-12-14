# restoration.py
import cv2

def restore_denoise(img_bgr):
    return cv2.medianBlur(img_bgr, 5)

def restore_sharpen(img_bgr):
    blur = cv2.GaussianBlur(img_bgr, (0, 0), 1.2)
    return cv2.addWeighted(img_bgr, 1.7, blur, -0.7, 0)

def restore_contrast(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge([l2, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def restore_pipeline(img_bgr):
    x = restore_denoise(img_bgr)
    x = restore_sharpen(x)
    x = restore_contrast(x)
    return x
