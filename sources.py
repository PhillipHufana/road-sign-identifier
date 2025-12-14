import os
import cv2
from config import IMG_EXTS, VID_EXTS

def open_camera():
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for idx in range(0, 4):
        for api in backends:
            try:
                cap = cv2.VideoCapture(idx, api)
                if cap is not None and cap.isOpened():
                    return cap, f"Webcam index={idx}"
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            try:
                cap = cv2.VideoCapture(idx)
                if cap is not None and cap.isOpened():
                    return cap, f"Webcam index={idx}"
                if cap is not None:
                    cap.release()
            except Exception:
                pass
    return None, None

def open_media(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in IMG_EXTS:
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return {"mode": "image", "path": path, "image": img, "cap": None, "fps": 0.0, "label": f"Image: {os.path.basename(path)}"}

    if ext in VID_EXTS:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1e-3:
            fps = 20.0
        return {"mode": "video", "path": path, "image": None, "cap": cap, "fps": float(fps), "label": f"Video: {os.path.basename(path)}"}

    raise RuntimeError(f"Unsupported file type: {ext}")
