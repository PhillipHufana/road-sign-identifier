#quality.py
import cv2
import numpy as np

def variance_of_laplacian(gray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def estimate_noise_simple(bgr) -> float:
    """
    Simple noise proxy:
    noise ≈ mean absolute difference between image and a fast denoised version.
    """
    den = cv2.fastNlMeansDenoisingColored(bgr, None, 7, 7, 7, 21)
    diff = cv2.absdiff(bgr, den)
    return float(np.mean(diff))

def basic_quality_diagnostics(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur_score = variance_of_laplacian(gray)     # lower = blurrier
    brightness = float(np.mean(gray))            # 0..255
    contrast = float(np.std(gray))               # higher = more contrast
    noise = estimate_noise_simple(bgr)           # higher = noisier
    return {
        "blur": blur_score,
        "brightness": brightness,
        "contrast": contrast,
        "noise": noise,
    }

def retake_guidance(diag):
    """
    Returns 1–2 short actionable tips.
    Thresholds are pragmatic defaults; tune later per dataset.
    """
    tips = []

    # Blur
    if diag["blur"] < 60:
        tips.append("Too blurry → hold steady / move closer / avoid zoom.")
    # Dark / Bright
    if diag["brightness"] < 70:
        tips.append("Too dark → add light / use flash / avoid backlight.")
    elif diag["brightness"] > 190:
        tips.append("Too bright → avoid direct glare / lower exposure.")

    # Noise
    if diag["noise"] > 10:
        tips.append("Too noisy → add light / lower ISO / denoise first.")

    # Low contrast
    if diag["contrast"] < 35:
        tips.append("Low contrast → change angle / avoid haze / boost contrast.")

    if not tips:
        tips.append("Looks usable → proceed to restore and export report.")

    # Keep it simple: max 2 tips
    return tips[:2]

def recommended_order(diag):
    """
    Decide a recommended restoration order (simple rule-based).
    """
    order = []
    if diag["noise"] > 10:
        order.append("Denoise")
    if diag["blur"] < 60:
        order.append("Sharpen")
    if diag["contrast"] < 35 or diag["brightness"] < 70:
        order.append("Contrast")
    if not order:
        order = ["Light Denoise", "Light Contrast"]
    return order
