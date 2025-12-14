#degradations.py
import numpy as np
import cv2

def blend_strength(base_bgr, effected_bgr, alpha: float):
    return cv2.addWeighted(base_bgr, 1.0 - alpha, effected_bgr, alpha, 0)

# Noise
def noise_gaussian(img_bgr):
    x = img_bgr.astype(np.float32)
    noise = np.random.normal(0, 22, x.shape).astype(np.float32)
    return np.clip(x + noise, 0, 255).astype(np.uint8)

def noise_saltpepper(img_bgr):
    out = img_bgr.copy()
    prob = 0.03
    rnd = np.random.rand(*img_bgr.shape[:2])
    out[rnd < prob] = 0
    out[rnd > 1 - prob] = 255
    return out

def noise_speckle(img_bgr):
    x = img_bgr.astype(np.float32)
    noise = np.random.randn(*x.shape).astype(np.float32)
    return np.clip(x + x * noise * 0.18, 0, 255).astype(np.uint8)

NOISE_VARIANTS = {"gaussian": noise_gaussian, "saltpepper": noise_saltpepper, "speckle": noise_speckle}

# Blur
def blur_gaussian(img_bgr):
    return cv2.GaussianBlur(img_bgr, (13, 13), 0)

def blur_motion(img_bgr):
    k = 17
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    kernel /= kernel.sum()
    return cv2.filter2D(img_bgr, -1, kernel)

def blur_defocus(img_bgr):
    return cv2.blur(img_bgr, (15, 15))

BLUR_VARIANTS = {"gaussian": blur_gaussian, "motion": blur_motion, "defocus": blur_defocus}

# Visibility
def vis_low_contrast(img_bgr):
    return cv2.convertScaleAbs(img_bgr, alpha=0.50, beta=40)

def vis_haze(img_bgr):
    fog = np.full(img_bgr.shape, 215, dtype=np.uint8)
    return cv2.addWeighted(img_bgr, 0.68, fog, 0.32, 0)

def vis_underexpose(img_bgr):
    return cv2.convertScaleAbs(img_bgr, alpha=0.70, beta=-35)

VIS_VARIANTS = {"low_contrast": vis_low_contrast, "haze": vis_haze, "underexpose": vis_underexpose}

def apply_all_degradations(original_bgr, alpha: float, truth_noise: str, truth_blur: str, truth_vis: str):
    x = original_bgr.copy()
    x1 = NOISE_VARIANTS[truth_noise](x)
    x = blend_strength(x, x1, alpha)
    x2 = BLUR_VARIANTS[truth_blur](x)
    x = blend_strength(x, x2, alpha)
    x3 = VIS_VARIANTS[truth_vis](x)
    x = blend_strength(x, x3, alpha)
    return x
