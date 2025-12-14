# mask_anon_dlib_tk_mainwindow_fixed.py
# Tkinter is the MAIN window (video drawn into a Tk Label)
# Default: live webcam feed (if available)
# Upload photo/video switches source and MUST show in the same preview.
# No pretrained models (dlib face detector + 68-landmark predictor only)
# Mask classification ALWAYS computed from the current ORIGINAL frame (not blurred output, not previous frames)

import os
os.environ.setdefault("QT_LOGGING_RULES", "qt.core.qmimedatabase=false;qt.qpa.*=false")

import time
import base64
import cv2
import dlib
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk


# -----------------------------
# CONFIG
# -----------------------------
PREDICTOR_PATH = os.environ.get(
    "DLIB_PREDICTOR",
    r"C:\Users\Phillip\Downloads\shape_predictor_68_face_landmarks.dat"
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v"}
DISPLAY_MAX_W = 960  # display-only scaling (processing stays at original resolution)

# -----------------------------
# dlib init
# -----------------------------
detector = dlib.get_frontal_face_detector()
if not os.path.exists(PREDICTOR_PATH):
    raise FileNotFoundError(
        f"shape_predictor_68_face_landmarks.dat not found at:\n{PREDICTOR_PATH}\n"
        "Fix PREDICTOR_PATH or set env var DLIB_PREDICTOR."
    )
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# -----------------------------
# Enhancement / restoration
# -----------------------------
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

# -----------------------------
# Segmentation + anonymization
# -----------------------------
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

# -----------------------------
# Classification (ALWAYS from ORIGINAL frame of this tick)
# -----------------------------
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

# -----------------------------
# Motion compensation (tracking)
# -----------------------------
class Track:
    def __init__(self, tid, tracker):
        self.tid = tid
        self.tracker = tracker

def rect_from_pos(pos, w, h):
    x1 = int(max(0, pos.left()))
    y1 = int(max(0, pos.top()))
    x2 = int(min(w - 1, pos.right()))
    y2 = int(min(h - 1, pos.bottom()))
    return x1, y1, x2, y2

# -----------------------------
# Source handling
# -----------------------------
def open_camera():
    # robust webcam open (some builds crash with (idx, api))
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
        return {
            "mode": "image",
            "path": path,
            "image": img,
            "cap": None,
            "fps": 0.0,
            "label": f"Image: {os.path.basename(path)}",
        }
    if ext in VID_EXTS:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1e-3:
            fps = 20.0
        return {
            "mode": "video",
            "path": path,
            "image": None,
            "cap": cap,
            "fps": float(fps),
            "label": f"Video: {os.path.basename(path)}",
        }
    raise RuntimeError(f"Unsupported file type: {ext}")

# -----------------------------
# Tk image conversion (NO Pillow)
# base64-encoded PPM (P6) for Tk PhotoImage
# -----------------------------
def bgr_to_tk_photo(frame_bgr, max_w=DISPLAY_MAX_W):
    # Resize for display only
    h, w = frame_bgr.shape[:2]
    if w > max_w:
        scale = max_w / float(w)
        nw, nh = int(w * scale), int(h * scale)
        frame_bgr = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    # Convert to RGB for Tk
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Pillow -> Tk PhotoImage
    im = Image.fromarray(rgb)
    return ImageTk.PhotoImage(im)


def make_placeholder(msg="Starting..."):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, msg, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return img

# -----------------------------
# Tk App
# -----------------------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Mask + Anonymization (dlib)")

        self.source = {"mode": "none", "cap": None, "image": None, "fps": 20.0, "label": "No source"}
        self.writer = None
        self.recording = False
        self.out_video_path = None

        self.paused = False
        self.frame_i = 0
        self.tracks = []
        self.next_id = 1

        self.lut_cache = {"gamma": None, "lut": None}
        self.nv_mode = "auto"  # "auto" | "nv" | "color"

        self.last_frame_out = make_placeholder("Opening webcam...")

        # ----- Layout -----
        self.container = tk.Frame(root, padx=10, pady=10)
        self.container.grid(row=0, column=0, sticky="nsew")
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)

        # Left: preview
        self.video_box = tk.LabelFrame(self.container, text="Preview", padx=8, pady=8)
        self.video_box.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.video_label = tk.Label(self.video_box)
        self.video_label.grid(row=0, column=0, sticky="nsew")
        self.video_box.grid_rowconfigure(0, weight=1)
        self.video_box.grid_columnconfigure(0, weight=1)

        # Right: controls
        self.ctrl_col = tk.Frame(self.container)
        self.ctrl_col.grid(row=0, column=1, sticky="ns")

        box_source = tk.LabelFrame(self.ctrl_col, text="Source", padx=8, pady=8)
        box_source.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        tk.Button(box_source, text="Use Webcam (Default)", command=self.on_webcam).grid(row=0, column=0, sticky="ew", pady=3)
        tk.Button(box_source, text="Upload Photo/Video", command=self.on_upload).grid(row=1, column=0, sticky="ew", pady=3)
        tk.Button(box_source, text="Pause/Resume", command=self.on_pause).grid(row=2, column=0, sticky="ew", pady=3)

        box_nv = tk.LabelFrame(self.ctrl_col, text="Night Vision", padx=8, pady=8)
        box_nv.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.nv_label_var = tk.StringVar(value="Mode: AUTO")
        tk.Label(box_nv, textvariable=self.nv_label_var).grid(row=0, column=0, sticky="w")
        tk.Button(box_nv, text="Toggle NV Mode", command=self.on_toggle_nv).grid(row=1, column=0, sticky="ew", pady=3)

        box_priv = tk.LabelFrame(self.ctrl_col, text="Privacy / Output", padx=8, pady=8)
        box_priv.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        tk.Button(box_priv, text="Snapshot (JPG)", command=self.on_snapshot).grid(row=0, column=0, sticky="ew", pady=3)
        self.rec_btn = tk.Button(box_priv, text="Start Recording", command=self.on_toggle_record)
        self.rec_btn.grid(row=1, column=0, sticky="ew", pady=3)

        box_params = tk.LabelFrame(self.ctrl_col, text="Parameters", padx=8, pady=8)
        box_params.grid(row=3, column=0, sticky="ew", pady=(0, 8))

        self.var_enh = tk.IntVar(value=1)
        self.var_den = tk.IntVar(value=0)
        self.var_ush = tk.IntVar(value=0)
        tk.Checkbutton(box_params, text="CLAHE", variable=self.var_enh).grid(row=0, column=0, sticky="w")
        tk.Checkbutton(box_params, text="Denoise", variable=self.var_den).grid(row=0, column=1, sticky="w")
        tk.Checkbutton(box_params, text="Unsharp", variable=self.var_ush).grid(row=0, column=2, sticky="w")

        self.gamma = tk.DoubleVar(value=1.00)
        tk.Label(box_params, text="Gamma").grid(row=1, column=0, sticky="w")
        tk.Scale(box_params, from_=0.4, to=2.5, resolution=0.05, orient="horizontal",
                 variable=self.gamma, length=220).grid(row=1, column=1, columnspan=2, sticky="ew")

        self.detect_every = tk.IntVar(value=10)
        tk.Label(box_params, text="DetectEveryN").grid(row=2, column=0, sticky="w")
        tk.Scale(box_params, from_=1, to=30, orient="horizontal",
                 variable=self.detect_every, length=220).grid(row=2, column=1, columnspan=2, sticky="ew")

        self.blur_mode = tk.IntVar(value=1)  # 0 off,1 all,2 only nomask/incorrect
        tk.Label(box_params, text="BlurMode").grid(row=3, column=0, sticky="w")
        tk.Radiobutton(box_params, text="Off", variable=self.blur_mode, value=0).grid(row=3, column=1, sticky="w")
        tk.Radiobutton(box_params, text="All", variable=self.blur_mode, value=1).grid(row=3, column=2, sticky="w")
        tk.Radiobutton(box_params, text="NoMask/Bad", variable=self.blur_mode, value=2).grid(row=4, column=1, columnspan=2, sticky="w")

        self.blur_k = tk.IntVar(value=31)
        tk.Label(box_params, text="BlurK").grid(row=5, column=0, sticky="w")
        tk.Scale(box_params, from_=3, to=101, resolution=2, orient="horizontal",
                 variable=self.blur_k, length=220).grid(row=5, column=1, columnspan=2, sticky="ew")

        self.feather_k = tk.IntVar(value=11)
        tk.Label(box_params, text="FeatherK").grid(row=6, column=0, sticky="w")
        tk.Scale(box_params, from_=1, to=51, resolution=2, orient="horizontal",
                 variable=self.feather_k, length=220).grid(row=6, column=1, columnspan=2, sticky="ew")

        self.thr_mouth_color = tk.DoubleVar(value=0.15)
        self.thr_nose_color = tk.DoubleVar(value=0.15)
        tk.Label(box_params, text="ThrMouthColor").grid(row=7, column=0, sticky="w")
        tk.Scale(box_params, from_=0.05, to=0.60, resolution=0.01, orient="horizontal",
                 variable=self.thr_mouth_color, length=220).grid(row=7, column=1, columnspan=2, sticky="ew")
        tk.Label(box_params, text="ThrNoseColor").grid(row=8, column=0, sticky="w")
        tk.Scale(box_params, from_=0.05, to=0.60, resolution=0.01, orient="horizontal",
                 variable=self.thr_nose_color, length=220).grid(row=8, column=1, columnspan=2, sticky="ew")

        self.thr_mouth_nv = tk.DoubleVar(value=0.60)
        self.thr_nose_nv = tk.DoubleVar(value=0.60)
        tk.Label(box_params, text="ThrMouthNV").grid(row=9, column=0, sticky="w")
        tk.Scale(box_params, from_=0.20, to=1.00, resolution=0.02, orient="horizontal",
                 variable=self.thr_mouth_nv, length=220).grid(row=9, column=1, columnspan=2, sticky="ew")
        tk.Label(box_params, text="ThrNoseNV").grid(row=10, column=0, sticky="w")
        tk.Scale(box_params, from_=0.20, to=1.00, resolution=0.02, orient="horizontal",
                 variable=self.thr_nose_nv, length=220).grid(row=10, column=1, columnspan=2, sticky="ew")

        self.show_landmarks = tk.IntVar(value=0)
        tk.Checkbutton(box_params, text="Show Landmarks", variable=self.show_landmarks).grid(row=11, column=0, columnspan=3, sticky="w")

        self.jpeg_q = tk.IntVar(value=75)
        tk.Label(box_params, text="JPEG_Q").grid(row=12, column=0, sticky="w")
        tk.Scale(box_params, from_=10, to=95, orient="horizontal",
                 variable=self.jpeg_q, length=220).grid(row=12, column=1, columnspan=2, sticky="ew")

        self.status_var = tk.StringVar(value="Starting...")
        self.status_bar = tk.Label(self.container, textvariable=self.status_var, anchor="w")
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

        # DEFAULT: start live webcam feed
        self.on_webcam(silent=True)

        # Start loop
        self.root.after(15, self.tick)

    # ----- UI actions -----
    def set_status(self, s):
        self.status_var.set(s)

    def on_toggle_nv(self):
        if self.nv_mode == "auto":
            self.nv_mode = "nv"
            self.nv_label_var.set("Mode: FORCED NV")
        elif self.nv_mode == "nv":
            self.nv_mode = "color"
            self.nv_label_var.set("Mode: FORCED COLOR")
        else:
            self.nv_mode = "auto"
            self.nv_label_var.set("Mode: AUTO")

    def on_pause(self):
        self.paused = not self.paused
        self.set_status(f"{'Paused' if self.paused else 'Running'} | {self.source.get('label','')}")

    def on_upload(self):
        path = filedialog.askopenfilename(
            title="Select an image or video",
            filetypes=[
                ("Media files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp *.mp4 *.avi *.mov *.mkv *.wmv *.webm *.m4v"),
                ("All files", "*.*")
            ],
        )
        if not path:
            return
        try:
            self.paused = False
            self._stop_recording()
            self._release_source()
            self.source = open_media(path)
            self._reset_tracking()

            # Force an immediate preview update (so you see it right away)
            if self.source["mode"] == "image":
                self.last_frame_out = self._process_frame(self.source["image"].copy())
            else:
                f = self._read_frame()
                if f is not None:
                    self.last_frame_out = self._process_frame(f)

            self.set_status(f"Loaded: {self.source['label']}")
        except Exception as e:
            messagebox.showerror("Open failed", str(e))

    def on_webcam(self, silent=False):
        self.paused = False
        self._stop_recording()
        self._release_source()
        cap, label = open_camera()
        if cap is None:
            self.source = {"mode": "none", "cap": None, "image": None, "fps": 20.0, "label": "No webcam"}
            self.last_frame_out = make_placeholder("Webcam unavailable. Upload media.")
            if not silent:
                messagebox.showwarning("Webcam", "Could not open webcam. Use Upload Photo/Video.")
            self.set_status("Webcam unavailable. Use Upload.")
            return

        self.source = {"mode": "webcam", "cap": cap, "image": None, "fps": 20.0, "label": label}
        self._reset_tracking()

        # Prime preview with first frame
        f = self._read_frame()
        if f is not None:
            self.last_frame_out = self._process_frame(f)
        else:
            self.last_frame_out = make_placeholder("Webcam opened, no frame yet...")

        self.set_status(f"Using: {label}")

    def on_snapshot(self):
        if self.last_frame_out is None:
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = f"snapshot_{ts}.jpg"
        cv2.imwrite(path, self.last_frame_out, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_q.get())])
        self.set_status(f"Saved snapshot: {path}")

    def on_toggle_record(self):
        if self.recording:
            self._stop_recording()
            return
        if self.last_frame_out is None:
            return

        ts = time.strftime("%Y%m%d_%H%M%S")
        self.out_video_path = f"record_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = float(self.source.get("fps", 20.0) or 20.0)
        h, w = self.last_frame_out.shape[:2]
        self.writer = cv2.VideoWriter(self.out_video_path, fourcc, fps, (w, h))
        if not self.writer.isOpened():
            self.writer = None
            self.out_video_path = None
            self.set_status("Recording failed to start.")
            return
        self.recording = True
        self.rec_btn.config(text="Stop Recording")
        self.set_status(f"Recording: {self.out_video_path}")

    def on_exit(self):
        self._stop_recording()
        self._release_source()
        self.root.destroy()

    # ----- internals -----
    def _reset_tracking(self):
        self.tracks = []
        self.next_id = 1
        self.frame_i = 0

    def _release_source(self):
        cap = self.source.get("cap")
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        self.source["cap"] = None

    def _stop_recording(self):
        if self.writer is not None:
            try:
                self.writer.release()
            except Exception:
                pass
        self.writer = None
        if self.recording:
            self.set_status(f"Saved recording: {self.out_video_path}" if self.out_video_path else "Recording stopped.")
        self.recording = False
        self.out_video_path = None
        self.rec_btn.config(text="Start Recording")

    # ----- main loop -----
    def tick(self):
        try:
            if not self.paused:
                frame = self._read_frame()
                if frame is not None:
                    out = self._process_frame(frame)
                    self.last_frame_out = out

                    if self.recording and self.writer is not None:
                        self.writer.write(out)
                else:
                    # keep last preview; just update status
                    if self.source.get("mode") in ("webcam", "video"):
                        self.set_status(f"No frame read ({self.source.get('label','')}).")

            # Always refresh preview from latest output
            if self.last_frame_out is not None:
                photo = bgr_to_tk_photo(self.last_frame_out, max_w=DISPLAY_MAX_W)
                self.video_label.configure(image=photo)
                self.video_label.image = photo

        except Exception as e:
            self.set_status(f"Preview error: {e}")

        self.root.after(15, self.tick)

    def _read_frame(self):
        mode = self.source.get("mode", "none")
        if mode == "image":
            img = self.source.get("image")
            return img.copy() if img is not None else None

        cap = self.source.get("cap")
        if cap is None:
            return None

        ret, frame = cap.read()
        if not ret or frame is None:
            if mode == "video":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if ret and frame is not None:
                    return frame
            return None
        return frame

    def _process_frame(self, frame_bgr):
        self.frame_i += 1
        h, w = frame_bgr.shape[:2]

        # ORIGINAL frame for classification (current tick only)
        frame_original = frame_bgr.copy()

        # Enhancement/restoration pipeline for detection/output
        proc = apply_gamma(frame_bgr, self.gamma.get(), self.lut_cache)
        if self.var_enh.get() == 1:
            proc = enhance_clahe(proc)
        if self.var_den.get() == 1:
            proc = restore_denoise(proc)
        if self.var_ush.get() == 1:
            proc = restore_unsharp(proc)

        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)

        detect_every = max(1, int(self.detect_every.get()))
        do_detect = (self.source.get("mode") == "image") or (self.frame_i % detect_every == 0) or (len(self.tracks) == 0)

        if do_detect:
            dets = detector(gray, 0)
            self.tracks = []
            for d in dets:
                tr = dlib.correlation_tracker()
                tr.start_track(rgb, d)
                self.tracks.append(Track(self.next_id, tr))
                self.next_id += 1
        else:
            for t in self.tracks:
                t.tracker.update(rgb)

        auto_is_nv = estimate_night_vision_mode(frame_original)
        if self.nv_mode == "nv":
            use_nv = True
        elif self.nv_mode == "color":
            use_nv = False
        else:
            use_nv = auto_is_nv

        thr_mouth_c = float(self.thr_mouth_color.get())
        thr_nose_c = float(self.thr_nose_color.get())
        thr_mouth_nv = float(self.thr_mouth_nv.get())
        thr_nose_nv = float(self.thr_nose_nv.get())

        blur_mode = int(self.blur_mode.get())
        blur_k = int(self.blur_k.get())
        feather_k = int(self.feather_k.get())

        faces_total = masked = incorrect = nomask = 0
        out = proc.copy()

        for t in self.tracks:
            pos = t.tracker.get_position()
            x1, y1, x2, y2 = rect_from_pos(pos, w, h)
            if x2 <= x1 or y2 <= y1:
                continue

            faces_total += 1
            rect = dlib.rectangle(x1, y1, x2, y2)
            shape = predictor(gray, rect)
            pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=np.int32)

            # Classification computed from ORIGINAL frame only
            if use_nv:
                cls = classify_mask_nightvision(frame_original, pts, thr_mouth_ratio=thr_mouth_nv, thr_nose_ratio=thr_nose_nv)
            else:
                cls = classify_mask_color(frame_original, pts, thr_mouth=thr_mouth_c, thr_nose=thr_nose_c)

            if cls == "MASK":
                masked += 1
            elif cls == "INCORRECT":
                incorrect += 1
            else:
                nomask += 1

            do_blur = False
            if blur_mode == 1:
                do_blur = True
            elif blur_mode == 2 and cls in ("NO_MASK", "INCORRECT"):
                do_blur = True

            if do_blur:
                poly = face_polygon_from_landmarks(pts)
                m = polygon_mask((h, w), poly)
                m = feather_mask(m, feather_k)
                out = blur_with_mask(out, m, blur_k)

            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, f"ID {t.tid} {cls}", (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if self.show_landmarks.get() == 1:
                for (px, py) in pts:
                    cv2.circle(out, (int(px), int(py)), 1, (255, 0, 0), -1)

        mode_txt = "NV" if use_nv else "COLOR"
        src_txt = self.source.get("label", self.source.get("mode", ""))
        cv2.putText(out, f"[{mode_txt}] Faces:{faces_total} Mask:{masked} Incorrect:{incorrect} NoMask:{nomask}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(out, f"Source: {src_txt}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if self.recording:
            cv2.putText(out, "REC", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        return out

def main():
    root = tk.Tk()
    _ = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
