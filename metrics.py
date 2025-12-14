# metrics.py
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

def psnr(a_bgr, b_bgr) -> float:
    mse = np.mean((a_bgr.astype(np.float32) - b_bgr.astype(np.float32)) ** 2)
    if mse <= 1e-10:
        return 99.0
    return float(20 * np.log10(255.0 / np.sqrt(mse)))

def ssim_bgr(a_bgr, b_bgr) -> float:
    a = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2GRAY)
    return float(ssim(a, b, data_range=255))

def compute_score(original_bgr, restored_bgr, time_bonus: int = 0):
    s = ssim_bgr(original_bgr, restored_bgr)
    p = psnr(original_bgr, restored_bgr)
    p_cap = min(p, 40.0)
    score = 700 * s + (p_cap * 7.5) + time_bonus
    return int(round(score)), s, p
