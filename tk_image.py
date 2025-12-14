import cv2
import numpy as np
from PIL import Image, ImageTk
from config import DISPLAY_MAX_W

def bgr_to_tk_photo(frame_bgr, max_w=DISPLAY_MAX_W):
    h, w = frame_bgr.shape[:2]
    if w > max_w:
        scale = max_w / float(w)
        nw, nh = int(w * scale), int(h * scale)
        frame_bgr = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    return ImageTk.PhotoImage(im)

def make_placeholder(msg="Starting..."):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, msg, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return img
