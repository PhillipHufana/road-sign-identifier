import cv2
import numpy as np
from anonymize import face_polygon_from_landmarks

def crop_roi(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return img[0:0, 0:0]
    return img[y1:y2, x1:x2]

def skin_fraction_bgr(roi_bgr):
    if roi_bgr.size == 0:
        return 0.0
    b = roi_bgr[:, :, 0].astype(np.int32)
    g = roi_bgr[:, :, 1].astype(np.int32)
    r = roi_bgr[:, :, 2].astype(np.int32)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    cond = (
        (r > 95) & (g > 40) & (b > 20) &
        ((mx - mn) > 15) &
        (np.abs(r - g) > 15) &
        (r > g) & (r > b)
    )
    return float(cond.mean())

def lap_var(gray_roi):
    if gray_roi.size == 0:
        return 0.0
    if gray_roi.ndim == 3:
        gray_roi = cv2.cvtColor(gray_roi, cv2.COLOR_BGR2GRAY)
    if gray_roi.shape[0] < 5 or gray_roi.shape[1] < 5:
        return 0.0
    return float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())

def estimate_night_vision_mode(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    s_mean = float(hsv[:, :, 1].mean())
    v_mean = float(hsv[:, :, 2].mean())
    return (s_mean < 25.0) or (v_mean < 55.0)

def classify_mask_color(frame_bgr_original, pts68, thr_mouth=0.15, thr_nose=0.15):
    mouth = pts68[48:68]
    x1, y1 = mouth[:, 0].min() - 6, mouth[:, 1].min() - 6
    x2, y2 = mouth[:, 0].max() + 6, mouth[:, 1].max() + 6
    mouth_roi = crop_roi(frame_bgr_original, x1, y1, x2, y2)

    nose = pts68[31:36]
    nx1, ny1 = nose[:, 0].min() - 6, nose[:, 1].min() - 6
    nx2, ny2 = nose[:, 0].max() + 6, nose[:, 1].max() + 6
    nose_roi = crop_roi(frame_bgr_original, nx1, ny1, nx2, ny2)

    mouth_skin = skin_fraction_bgr(mouth_roi)
    nose_skin = skin_fraction_bgr(nose_roi)

    if mouth_skin < thr_mouth and nose_skin < thr_nose:
        return "MASK"
    if mouth_skin < thr_mouth and nose_skin >= thr_nose:
        return "INCORRECT"
    return "NO_MASK"

def classify_mask_nightvision(frame_bgr_original, pts68, thr_mouth_ratio=0.60, thr_nose_ratio=0.60):
    gray = cv2.cvtColor(frame_bgr_original, cv2.COLOR_BGR2GRAY)

    poly = face_polygon_from_landmarks(pts68)
    fx1, fy1 = int(poly[:, 0].min()), int(poly[:, 1].min())
    fx2, fy2 = int(poly[:, 0].max()), int(poly[:, 1].max())
    face_roi = crop_roi(gray, fx1, fy1, fx2, fy2)

    mouth = pts68[48:68]
    mx1, my1 = int(mouth[:, 0].min() - 6), int(mouth[:, 1].min() - 6)
    mx2, my2 = int(mouth[:, 0].max() + 6), int(mouth[:, 1].max() + 6)
    mouth_roi = crop_roi(gray, mx1, my1, mx2, my2)

    nose = pts68[31:36]
    nx1, ny1 = int(nose[:, 0].min() - 6), int(nose[:, 1].min() - 6)
    nx2, ny2 = int(nose[:, 0].max() + 6), int(nose[:, 1].max() + 6)
    nose_roi = crop_roi(gray, nx1, ny1, nx2, ny2)

    v_face = lap_var(face_roi)
    v_mouth = lap_var(mouth_roi)
    v_nose = lap_var(nose_roi)

    eps = 1e-6
    mouth_ratio = v_mouth / (v_face + eps)
    nose_ratio = v_nose / (v_face + eps)

    if mouth_ratio < thr_mouth_ratio and nose_ratio < thr_nose_ratio:
        return "MASK"
    if mouth_ratio < thr_mouth_ratio and nose_ratio >= thr_nose_ratio:
        return "INCORRECT"
    return "NO_MASK"
