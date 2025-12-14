import cv2
import numpy as np

def face_polygon_from_landmarks(pts68):
    jaw = pts68[0:17]
    brow = pts68[17:27][::-1]
    return np.vstack([jaw, brow])

def polygon_mask(shape_hw, poly_pts):
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_pts.astype(np.int32)], 255)
    return mask

def feather_mask(mask, feather_k):
    k = int(feather_k)
    if k <= 1:
        return mask
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(mask, (k, k), 0)

def blur_with_mask(frame_bgr, mask_u8, blur_k):
    k = int(blur_k)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(frame_bgr, (k, k), 0)
    out = frame_bgr.copy()
    out[mask_u8 == 255] = blurred[mask_u8 == 255]
    return out
