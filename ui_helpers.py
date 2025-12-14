# ui_helpers.py
import tkinter as tk
import cv2
from PIL import Image, ImageTk

def draw_gradient(canvas: tk.Canvas, w: int, h: int, c1: str, c2: str):
    canvas.delete("all")

    def hex_to_rgb(hx):
        hx = hx.lstrip("#")
        return tuple(int(hx[i:i + 2], 16) for i in (0, 2, 4))

    def rgb_to_hex(rgb):
        return "#%02x%02x%02x" % rgb

    r1, g1, b1 = hex_to_rgb(c1)
    r2, g2, b2 = hex_to_rgb(c2)

    steps = max(1, h)
    for y in range(h):
        t = y / (steps - 1) if steps > 1 else 1
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        canvas.create_rectangle(0, y, w, y + 1, outline="", fill=rgb_to_hex((r, g, b)))

def set_group_state(canvas: tk.Canvas, state: str):
    if state == "correct":
        draw_gradient(canvas, 380, 110, "#1b5e20", "#66bb6a")
    elif state == "partial":
        draw_gradient(canvas, 380, 110, "#f9a825", "#fff59d")
    elif state == "wrong":
        draw_gradient(canvas, 380, 110, "#b71c1c", "#ef5350")
    else:
        draw_gradient(canvas, 380, 110, "#2b2b2b", "#1f1f1f")

def to_tk(img_bgr, max_w=620, max_h=260):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    nw, nh = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    return ImageTk.PhotoImage(Image.fromarray(img_resized))

def nice_label(key: str) -> str:
    mapping = {
        "gaussian": "Gaussian",
        "saltpepper": "Salt & Pepper",
        "speckle": "Speckle",
        "motion": "Motion",
        "defocus": "Defocus",
        "low_contrast": "Low Contrast",
        "haze": "Haze/Fog",
        "underexpose": "Underexposed",
    }
    return mapping.get(key, key)
