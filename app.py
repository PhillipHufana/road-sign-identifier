import time
import cv2
import dlib
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from collections import Counter, deque

# App config and GUI helpers
from config import APP_TITLE, DISPLAY_MAX_W
from tk_image import bgr_to_tk_photo, make_placeholder

# Dlib model loading (face detector + landmark predictor)
from dlib_models import load_dlib_models

# Image enhancement / restoration pipeline stages
from ivp_enhance import apply_gamma, enhance_clahe, restore_denoise, restore_unsharp

# Privacy blur helpers (polygon mask from landmarks + feathered blur)
from anonymize import face_polygon_from_landmarks, polygon_mask, feather_mask, blur_with_mask

# Mask classification + NV mode detection
from classify import estimate_night_vision_mode, classify_mask_color, classify_mask_nightvision

# Face tracking wrapper and geometry conversion
from tracking import Track, rect_from_pos

# Media input helpers (webcam and file open)
from sources import open_camera, open_media


class ScrollableFrame(ttk.Frame):
    """
    ttk.Frame that becomes scrollable by embedding an inner Frame inside a Canvas.
    This is used for the Advanced settings panel so it can hold many controls.
    """
    def __init__(self, parent, height=260, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # Canvas hosts the inner frame; scrollbar controls the canvas view.
        self.canvas = tk.Canvas(self, highlightthickness=0, bg="#F6F7F9")
        self.vscroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vscroll.set)

        # Layout: canvas on the left, scrollbar on the right.
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vscroll.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Set an initial visible height while still allowing container resizing.
        self.canvas.configure(height=height)

        # Inner frame that holds actual widgets.
        self.inner = ttk.Frame(self.canvas)
        self.inner_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        def _sync_scrollregion():
            """
            Update scrollregion based on inner content height.
            Also force inner frame width to match the canvas width so widgets wrap nicely.
            """
            self.inner.update_idletasks()
            req_w = self.canvas.winfo_width()
            req_h = self.inner.winfo_reqheight()
            self.canvas.itemconfigure(self.inner_id, width=req_w)
            self.canvas.configure(scrollregion=(0, 0, req_w, req_h))

        # Keep scrollregion correct when content or canvas size changes.
        self.inner.bind("<Configure>", lambda e: _sync_scrollregion())
        self.canvas.bind("<Configure>", lambda e: _sync_scrollregion())

        # Enable mouse wheel scrolling when the cursor is over the frame.
        self._bind_mousewheel(self.canvas)
        self._bind_mousewheel(self.inner)

    def _bind_mousewheel(self, widget):
        # Bind on enter/leave so the wheel scroll only applies while hovering this scroller.
        widget.bind("<Enter>", lambda e: self._activate_mousewheel())
        widget.bind("<Leave>", lambda e: self._deactivate_mousewheel())

    def _activate_mousewheel(self):
        # Windows/macOS mouse wheel
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        # Linux mouse wheel (button events)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _deactivate_mousewheel(self):
        # Remove global wheel bindings when leaving the scrollable area.
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, e):
        # Tk reports wheel delta in units of 120 per notch on many platforms.
        self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

    def _on_mousewheel_linux(self, e):
        # Linux: Button-4 scrolls up, Button-5 scrolls down.
        self.canvas.yview_scroll(-1 if e.num == 4 else 1, "units")


class App:
    """
    Main application:
    - Captures frames from webcam / video / still image
    - Applies optional enhancement (gamma, CLAHE, denoise, unsharp)
    - Detects faces periodically and tracks between detections
    - Predicts landmarks and classifies mask status per face
    - Applies privacy blur based on settings
    - Renders preview and optional recording/snapshot outputs
    """
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)

        # Configure ttk theme and widget styles.
        self._setup_style()

        # Load dlib face detector and landmark predictor.
        self.detector, self.predictor = load_dlib_models()

        # Current input source descriptor.
        # mode: "none" | "webcam" | "video" | "image"
        # cap: cv2.VideoCapture for webcam/video
        # image: still frame for image mode
        self.source = {"mode": "none", "cap": None, "image": None, "fps": 20.0, "label": "No source"}

        # Video recording state.
        self.writer = None
        self.recording = False
        self.out_video_path = None

        # Playback and tracking state.
        self.paused = False
        self.frame_i = 0
        self.tracks = []      # list[Track] using dlib.correlation_tracker
        self.next_id = 1      # monotonically increasing track ID

        # Classification stability settings:
        # - cls_hist_len: number of recent raw classifications used for voting
        # - cls_change_ratio: fraction of votes needed to switch stable label
        # - hyst_color/hyst_nv: threshold hysteresis to reduce flip-flopping
        self.cls_hist_len = 9
        self.cls_change_ratio = 0.70
        self.hyst_color = 0.03
        self.hyst_nv = 0.05

        # Per-track state keyed by tid:
        # {"hist": deque(...), "stable": str|None, "last_seen": frame_index}
        self.track_states = {}

        # Denoise performance controls:
        # - denoise_every_n: run heavy denoise once every N frames and reuse cached output
        # - denoise_scale: downscale before denoise then upscale back (speed/quality tradeoff)
        # - denoise_only_when_nv: skip denoise unless NV mode is active (speed optimization)
        self.denoise_every_n = 4
        self._denoise_cache = None
        self._denoise_cache_frame_i = -10**9
        self.denoise_scale = 0.5
        self.denoise_only_when_nv = True

        # Gamma LUT cache to avoid rebuilding LUT every frame unless gamma changes.
        self.lut_cache = {"gamma": None, "lut": None}

        # Night-vision behavior:
        # - "auto": use estimate_night_vision_mode(frame)
        # - "nv": force NV classification path
        # - "color": force color classification path
        self.nv_mode = "auto"

        # Last processed frame used for preview/record/snapshot.
        self.last_frame_out = make_placeholder("Opening webcam...")

        # UI state variables.
        self.show_adv = tk.BooleanVar(value=False)
        self.blur_mode_str = tk.StringVar(value="Blur all faces")

        # Enhancement toggles controlled by UI checkbuttons.
        self.var_enh = tk.IntVar(value=1)  # CLAHE
        self.var_den = tk.IntVar(value=0)  # Denoise
        self.var_ush = tk.IntVar(value=0)  # Unsharp

        # UI-controlled parameters.
        self.gamma = tk.DoubleVar(value=1.00)
        self.detect_every = tk.DoubleVar(value=10)  # run face detector every N frames

        self.blur_k = tk.DoubleVar(value=31)        # blur kernel size
        self.feather_k = tk.DoubleVar(value=11)     # mask feathering radius

        # Mask classification thresholds for visible light (color) mode.
        self.thr_mouth_color = tk.DoubleVar(value=0.15)
        self.thr_nose_color = tk.DoubleVar(value=0.15)

        # Mask classification thresholds for night vision (NV) mode.
        self.thr_mouth_nv = tk.DoubleVar(value=0.60)
        self.thr_nose_nv = tk.DoubleVar(value=0.60)

        # Debug/utility toggles.
        self.show_landmarks = tk.IntVar(value=0)
        self.jpeg_q = tk.DoubleVar(value=75)

        # Header/status bar text variables.
        self.header_source_var = tk.StringVar(value="Source: Webcam")
        self.header_mode_var = tk.StringVar(value="Mode: AUTO")
        self.header_rec_var = tk.StringVar(value="")

        self.nv_label_var = tk.StringVar(value="AUTO")
        self.status_var = tk.StringVar(value="Starting...")

        # Build the UI and hook window close.
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

        # Start with webcam and begin the main loop.
        self.on_webcam(silent=True)
        self.root.after(15, self.tick)

    def _get_track_state(self, tid: int):
        """
        Get (or create) per-track classification state used for smoothing:
        - hist: last N raw classes
        - stable: current stable class after voting/hysteresis
        - last_seen: last frame index where this track was processed
        """
        st = self.track_states.get(tid)
        if st is None:
            st = {"hist": deque(maxlen=self.cls_hist_len), "stable": None, "last_seen": self.frame_i}
            self.track_states[tid] = st
        return st

    def _stable_class_update(self, tid: int, raw_cls: str) -> str:
        """
        Update and return stable class for a track using:
        - Majority vote over the last cls_hist_len raw classes
        - Tie-breaker: keep previous stable label if still tied for first
        - Vote-hysteresis: require cls_change_ratio of votes to switch labels
        """
        st = self._get_track_state(tid)
        st["hist"].append(raw_cls)
        st["last_seen"] = self.frame_i

        counts = Counter(st["hist"])
        maxc = max(counts.values())
        winners = [k for k, v in counts.items() if v == maxc]

        prev = st["stable"]

        # If there is a tie, prefer keeping the previous stable label to reduce flicker.
        maj = winners[0] if len(winners) == 1 else (prev if prev in winners else winners[0])

        # Only switch stable label if we have a strong majority.
        if prev is None:
            st["stable"] = maj
        elif maj != prev:
            need = max(1, int(len(st["hist"]) * self.cls_change_ratio + 0.999))
            if maxc >= need:
                st["stable"] = maj

        return st["stable"]

    def _purge_track_states(self):
        """
        Remove per-track history for tracks that have not been seen recently.
        This prevents unbounded growth if track IDs keep increasing over time.
        """
        ttl = max(30, int(round(self.detect_every.get())) * 3)  # frames
        dead = [tid for tid, st in self.track_states.items() if (self.frame_i - st.get("last_seen", 0)) > ttl]
        for tid in dead:
            self.track_states.pop(tid, None)

    # -------------------------
    # Styling
    # -------------------------
    def _setup_style(self):
        # Apply a consistent background and basic ttk styling.
        self.root.configure(bg="#F6F7F9")
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("TFrame", background="#F6F7F9")
        style.configure("TLabelframe", background="#F6F7F9")
        style.configure("TLabelframe.Label", background="#F6F7F9", font=("Segoe UI", 10, "bold"))
        style.configure("TLabel", background="#F6F7F9", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("TCheckbutton", background="#F6F7F9", font=("Segoe UI", 10))

    # -------------------------
    # UI Construction
    # -------------------------
    def _build_ui(self):
        PAD = 10
        GAP = 10

        # Root container uses a 2-column layout:
        # - left: preview
        # - right: controls
        self.container = ttk.Frame(self.root, padding=(PAD, PAD, PAD, PAD))
        self.container.grid(row=0, column=0, sticky="nsew")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # ---- Header row ----
        header = ttk.Frame(self.container, padding=(0, 0, 0, 8))
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.columnconfigure(0, weight=1)
        header.columnconfigure(1, weight=1)
        header.columnconfigure(2, weight=0)

        self.header_left = ttk.Label(header, textvariable=self.header_source_var, font=("Segoe UI", 10, "bold"))
        self.header_left.grid(row=0, column=0, sticky="w")

        self.header_mid = ttk.Label(header, textvariable=self.header_mode_var)
        self.header_mid.grid(row=0, column=1, sticky="e")

        self.header_right = ttk.Label(header, textvariable=self.header_rec_var)
        self.header_right.grid(row=0, column=2, sticky="e", padx=(12, 0))

        # ---- Preview (left column) ----
        self.video_box = ttk.LabelFrame(self.container, text="Preview", padding=(8, 8, 8, 8))
        self.video_box.grid(row=1, column=0, sticky="nsew", padx=(0, GAP))
        self.container.grid_rowconfigure(1, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        # Use tk.Label (not ttk.Label) for fast image updates.
        self.video_label = tk.Label(self.video_box, bg="#000000")
        self.video_label.grid(row=0, column=0, sticky="nsew")
        self.video_box.grid_rowconfigure(0, weight=1)
        self.video_box.grid_columnconfigure(0, weight=1)

        # ---- Control panel (right column) ----
        self.ctrl_col = ttk.Frame(self.container)
        self.ctrl_col.grid(row=1, column=1, sticky="ns")
        self.ctrl_col.columnconfigure(0, weight=1)

        # ===== Source / Playback (basic controls) =====
        box_source = ttk.LabelFrame(self.ctrl_col, text="Source", padding=(10, 10, 10, 10))
        box_source.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        box_source.columnconfigure(0, weight=1)

        self.btn_webcam = ttk.Button(box_source, text="Use Webcam", command=self.on_webcam)
        self.btn_webcam.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        self.btn_upload = ttk.Button(box_source, text="Upload Photo/Video", command=self.on_upload)
        self.btn_upload.grid(row=1, column=0, sticky="ew", pady=(0, 6))

        self.btn_pause = ttk.Button(box_source, text="Pause / Resume", command=self.on_pause)
        self.btn_pause.grid(row=2, column=0, sticky="ew", pady=(0, 6))

        self.btn_exit = ttk.Button(box_source, text="Exit", command=self.on_exit)
        self.btn_exit.grid(row=3, column=0, sticky="ew")

        # ===== Privacy / Output (basic controls) =====
        box_priv = ttk.LabelFrame(self.ctrl_col, text="Privacy / Output", padding=(10, 10, 10, 10))
        box_priv.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        box_priv.columnconfigure(0, weight=1)
        box_priv.columnconfigure(1, weight=1)

        ttk.Label(box_priv, text="Privacy Blur").grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.cmb_blur = ttk.Combobox(
            box_priv,
            textvariable=self.blur_mode_str,
            state="readonly",
            values=["Off", "Blur all faces", "Blur only NoMask/Incorrect"],
        )
        self.cmb_blur.grid(row=0, column=1, sticky="ew", pady=(0, 6))

        # Blur kernel controls how strong/large the blur appears.
        self._scale_row(
            parent=box_priv,
            row=1,
            label="Blur Strength",
            var=self.blur_k,
            from_=3,
            to=101,
            step=2,
            fmt="{:0.0f}",
            help_text=None,
        )

        self.btn_snapshot = ttk.Button(box_priv, text="Snapshot (JPG)", command=self.on_snapshot)
        self.btn_snapshot.grid(row=2, column=0, sticky="ew", pady=(8, 0))

        self.rec_btn = ttk.Button(box_priv, text="Start Recording", command=self.on_toggle_record)
        self.rec_btn.grid(row=2, column=1, sticky="ew", pady=(8, 0))
        self.ctrl_col.grid_rowconfigure(2, weight=1)

        # ===== Detection (basic controls + Advanced toggle) =====
        box_det = ttk.LabelFrame(self.ctrl_col, text="Detection", padding=(10, 10, 10, 10))
        box_det.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        box_det.columnconfigure(0, weight=1)
        box_det.columnconfigure(1, weight=1)
        box_det.grid_rowconfigure(3, weight=1)  # advanced scroller goes here

        ttk.Label(box_det, text="Night Vision").grid(row=0, column=0, sticky="w")
        self.btn_nv = ttk.Button(box_det, text="Toggle NV Mode", command=self.on_toggle_nv)
        self.btn_nv.grid(row=0, column=1, sticky="ew")

        ttk.Label(box_det, text="Current").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.lbl_nv = ttk.Label(box_det, textvariable=self.nv_label_var)
        self.lbl_nv.grid(row=1, column=1, sticky="e", pady=(6, 0))

        # Advanced panel is hidden by default and contains thresholds and enhancements.
        self.adv_btn = ttk.Button(box_det, text="Advanced ▸", command=self.toggle_advanced)
        self.adv_btn.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        # Advanced scroller holds the advanced controls.
        self.advanced_scroller = ScrollableFrame(box_det, height=300)
        self.advanced_scroller.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        self.advanced_scroller.grid_remove()

        self.advanced_frame = self.advanced_scroller.inner
        self.advanced_frame.columnconfigure(0, weight=1)
        self.advanced_frame.columnconfigure(1, weight=1)

        # Enhancement toggles (apply in processing pipeline).
        row0 = ttk.Frame(self.advanced_frame)
        row0.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Checkbutton(row0, text="CLAHE", variable=self.var_enh).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(row0, text="Denoise", variable=self.var_den).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(row0, text="Unsharp", variable=self.var_ush).pack(side="left")

        # Sliders for enhancement and detection cadence.
        self._scale_row(self.advanced_frame, 1, "Gamma", self.gamma, 0.4, 2.5, 0.05, "{:0.2f}")
        self._scale_row(self.advanced_frame, 2, "Detect Every N", self.detect_every, 1, 30, 1, "{:0.0f}")
        self._scale_row(self.advanced_frame, 3, "Feather Edge", self.feather_k, 1, 51, 2, "{:0.0f}")

        # Classification thresholds for color and NV modes.
        self._scale_row(self.advanced_frame, 4, "Thr Mouth (Color)", self.thr_mouth_color, 0.05, 0.60, 0.01, "{:0.2f}")
        self._scale_row(self.advanced_frame, 5, "Thr Nose  (Color)", self.thr_nose_color, 0.05, 0.60, 0.01, "{:0.2f}")
        self._scale_row(self.advanced_frame, 6, "Thr Mouth (NV)", self.thr_mouth_nv, 0.20, 1.00, 0.02, "{:0.2f}")
        self._scale_row(self.advanced_frame, 7, "Thr Nose  (NV)", self.thr_nose_nv, 0.20, 1.00, 0.02, "{:0.2f}")

        # Debug landmarks and snapshot JPEG quality.
        ttk.Checkbutton(self.advanced_frame, text="Show Landmarks", variable=self.show_landmarks).grid(
            row=8, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        self._scale_row(self.advanced_frame, 9, "JPEG Quality", self.jpeg_q, 10, 95, 1, "{:0.0f}")

        # ---- Status bar ----
        self.status_bar = ttk.Label(self.container, textvariable=self.status_var, anchor="w")
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))

    def _scale_row(self, parent, row, label, var, from_, to, step, fmt="{:0.2f}", help_text=None):
        """
        Build a slider row: Label | Scale | Value.
        ttk.Scale is continuous; this callback quantizes values to a given step.
        """
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text=label).grid(row=0, column=0, sticky="w")

        value_var = tk.StringVar()

        def quantize_and_update(val):
            # Convert to float and quantize to "step" increments.
            try:
                f = float(val)
            except Exception:
                f = float(var.get())

            if step and step > 0:
                q = round((f - from_) / step) * step + from_
            else:
                q = f

            # Clamp to slider bounds.
            q = max(from_, min(to, q))

            # Update backing variable and the displayed value label.
            var.set(q)
            value_var.set(fmt.format(q))

        # Initialize displayed value.
        value_var.set(fmt.format(var.get()))

        scale = ttk.Scale(frame, from_=from_, to=to, command=quantize_and_update)
        scale.set(var.get())
        scale.grid(row=0, column=1, sticky="ew", padx=8)

        ttk.Label(frame, textvariable=value_var, width=8, anchor="e").grid(row=0, column=2, sticky="e")

        if help_text:
            ttk.Label(frame, text=help_text).grid(row=1, column=0, columnspan=3, sticky="w")

    # -------------------------
    # UI actions
    # -------------------------
    def set_status(self, s):
        # Update status bar text.
        self.status_var.set(s)

    def toggle_advanced(self):
        # Show/hide advanced settings panel.
        self.show_adv.set(not self.show_adv.get())
        if self.show_adv.get():
            self.advanced_scroller.grid()
            self.adv_btn.config(text="Advanced ▾")
        else:
            self.advanced_scroller.grid_remove()
            self.adv_btn.config(text="Advanced ▸")

    def on_toggle_nv(self):
        # Cycle NV mode: auto -> forced NV -> forced color -> auto.
        if self.nv_mode == "auto":
            self.nv_mode = "nv"
            self.nv_label_var.set("FORCED NV")
        elif self.nv_mode == "nv":
            self.nv_mode = "color"
            self.nv_label_var.set("FORCED COLOR")
        else:
            self.nv_mode = "auto"
            self.nv_label_var.set("AUTO")

    def on_pause(self):
        # Pause/resume processing loop while keeping last frame visible.
        self.paused = not self.paused
        self.set_status(f"{'Paused' if self.paused else 'Running'} | {self.source.get('label','')}")

    def on_upload(self):
        # Open file dialog and load an image or video.
        path = filedialog.askopenfilename(
            title="Select an image or video",
            filetypes=[
                ("Media files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp *.mp4 *.avi *.mov *.mkv *.wmv *.webm *.m4v"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            # Reset runtime state for a new source.
            self.paused = False
            self._stop_recording()
            self._release_source()

            # Display placeholder while loading and clear cached denoise output.
            self.last_frame_out = make_placeholder("Loading media...")
            self._refresh_preview()
            self._denoise_cache = None
            self._denoise_cache_frame_i = -10**9

            # Load media and reset tracking.
            self.source = open_media(path)
            self._reset_tracking()

            # Prime the preview immediately (process one frame right away).
            if self.source["mode"] == "image":
                self.last_frame_out = self._process_frame(self.source["image"].copy())
            else:
                f = self._read_frame()
                if f is not None:
                    self.last_frame_out = self._process_frame(f)

            self.set_status(f"Loaded: {self.source['label']}")
        except Exception as e:
            messagebox.showerror("Open failed", str(e))

    def _denoise_fast(self, img_bgr):
        """
        Speed-optimized denoise:
        - Optionally downscale the image for denoise (denoise_scale < 1)
        - Apply the heavy restore_denoise() on the smaller image
        - Upscale back to original resolution

        This trades fine detail for speed; it is mainly intended for noisy NV frames.
        """
        s = float(getattr(self, "denoise_scale", 1.0) or 1.0)
        if s >= 0.999:
            return restore_denoise(img_bgr)

        # Downscale for speed.
        small = cv2.resize(img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        small_dn = restore_denoise(small)

        # Upscale to original size for downstream processing and display.
        return cv2.resize(small_dn, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

    def on_webcam(self, silent=False):
        # Switch source to webcam capture.
        self.paused = False
        self._stop_recording()
        self._release_source()

        # Show placeholder while opening webcam and clear cached denoise output.
        self.last_frame_out = make_placeholder("Opening webcam...")
        self._denoise_cache = None
        self._denoise_cache_frame_i = -10**9
        self._refresh_preview()

        cap, label = open_camera()
        if cap is None:
            # Webcam open failed; keep app usable by allowing Upload.
            self.source = {"mode": "none", "cap": None, "image": None, "fps": 20.0, "label": "No webcam"}
            self.last_frame_out = make_placeholder("Webcam unavailable. Upload media.")
            if not silent:
                messagebox.showwarning("Webcam", "Could not open webcam. Use Upload Photo/Video.")
            self.set_status("Webcam unavailable. Use Upload.")
            return

        # Store new webcam source and reset tracking for the new stream.
        self.source = {"mode": "webcam", "cap": cap, "image": None, "fps": 20.0, "label": label}
        self._reset_tracking()

        # Process one frame immediately so the UI updates quickly.
        f = self._read_frame()
        self.last_frame_out = self._process_frame(f) if f is not None else make_placeholder("Webcam opened, no frame yet...")
        self.set_status(f"Using: {label}")

    def on_snapshot(self):
        # Save current output frame to disk as JPG.
        if self.last_frame_out is None:
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = f"snapshot_{ts}.jpg"
        cv2.imwrite(path, self.last_frame_out, [int(cv2.IMWRITE_JPEG_QUALITY), int(round(self.jpeg_q.get()))])
        self.set_status(f"Saved snapshot: {path}")

    def on_toggle_record(self):
        # Toggle video recording for webcam/video sources.
        if self.recording:
            self._stop_recording()
            return

        # Disallow recording for still images.
        if self.source.get("mode") == "image":
            messagebox.showinfo("Recording", "Recording is disabled for still images. Upload a video or use webcam.")
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
        # Clean shutdown: stop recording, release camera/video, close window.
        self._stop_recording()
        self._release_source()
        self.root.destroy()

    # -------------------------
    # Internals
    # -------------------------
    def _reset_tracking(self):
        # Reset trackers and per-track classification state for a new stream/file.
        self.tracks = []
        self.next_id = 1
        self.frame_i = 0
        self.track_states.clear()

    def _release_source(self):
        # Release the OpenCV capture handle if present.
        cap = self.source.get("cap")
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        self.source["cap"] = None

    def _stop_recording(self):
        # Stop and finalize any active recording.
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

    def _update_header(self, use_nv: bool):
        # Update top header text (source label, mode label, recording indicator).
        src = self.source.get("label", self.source.get("mode", ""))
        self.header_source_var.set(f"Source: {src}")

        if self.nv_mode == "nv":
            mode_txt = "FORCED NV"
        elif self.nv_mode == "color":
            mode_txt = "FORCED COLOR"
        else:
            mode_txt = "AUTO (NV)" if use_nv else "AUTO (COLOR)"
        self.header_mode_var.set(f"Mode: {mode_txt}")

        self.header_rec_var.set("● REC" if self.recording else "")

        # Mirror NV mode state in the control panel label.
        if self.nv_mode == "nv":
            self.nv_label_var.set("FORCED NV")
        elif self.nv_mode == "color":
            self.nv_label_var.set("FORCED COLOR")
        else:
            self.nv_label_var.set("AUTO")

    def _refresh_preview(self):
        # Convert BGR frame to a Tk PhotoImage and display it.
        if self.last_frame_out is None:
            return
        photo = bgr_to_tk_photo(self.last_frame_out, max_w=DISPLAY_MAX_W)
        self.video_label.configure(image=photo)
        self.video_label.image = photo  # keep reference to prevent garbage collection

    # -------------------------
    # Main loop
    # -------------------------
    def tick(self):
        """
        Periodic UI loop:
        - Read/process frames when not paused
        - Write output to recorder if active
        - Refresh displayed preview image
        """
        try:
            if not self.paused:
                frame = self._read_frame()
                if frame is not None:
                    out, use_nv = self._process_frame(frame, return_use_nv=True)
                    self.last_frame_out = out

                    if self.recording and self.writer is not None:
                        self.writer.write(out)

                    self._update_header(use_nv)
                else:
                    # For webcam/video, missing frames usually means capture issue or end-of-stream.
                    if self.source.get("mode") in ("webcam", "video"):
                        self.set_status(f"No frame read ({self.source.get('label','')}).")

            # Always redraw last frame so the UI stays responsive.
            if self.last_frame_out is not None:
                self._refresh_preview()

            # Disable record button when showing a still image (unless already recording).
            if self.source.get("mode") == "image" and not self.recording:
                self.rec_btn.state(["disabled"])
            else:
                self.rec_btn.state(["!disabled"])

        except Exception as e:
            self.set_status(f"Preview error: {e}")

        # Schedule next tick.
        self.root.after(15, self.tick)

    def _read_frame(self):
        """
        Read one frame from the active source.
        - For images: return a copy of the stored image
        - For video/webcam: cap.read()
        - For videos: rewind when end-of-file is reached
        """
        mode = self.source.get("mode", "none")

        if mode == "image":
            img = self.source.get("image")
            return img.copy() if img is not None else None

        cap = self.source.get("cap")
        if cap is None:
            return None

        ret, frame = cap.read()
        if ret and frame is not None:
            return frame

        # If a video ends, rewind and replay.
        if mode == "video":
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            except Exception:
                pass
            self._reset_tracking()
            self.set_status("Replaying video...")
            ret2, frame2 = cap.read()
            if ret2 and frame2 is not None:
                return frame2

        return None

    # -------------------------
    # Processing
    # -------------------------
    def _blur_mode_int(self):
        # Map UI dropdown string to internal integer mode.
        mode_map = {
            "Off": 0,
            "Blur all faces": 1,
            "Blur only NoMask/Incorrect": 2,
        }
        return mode_map.get(self.blur_mode_str.get(), 1)

    def _process_frame(self, frame_bgr, return_use_nv=False):
        """
        Process one frame:
        1) Decide NV vs color mode (auto or forced)
        2) Apply gamma + optional CLAHE + optional denoise + optional unsharp
        3) Detect faces periodically and track between detections
        4) For each face: landmarks -> classify mask -> optionally blur
        5) Render overlays and return output frame
        """
        self.frame_i += 1
        h, w = frame_bgr.shape[:2]

        # Preserve original frame for any logic that should depend on raw appearance.
        frame_original = frame_bgr.copy()

        # Decide NV early so denoise can be skipped when it is not needed.
        auto_is_nv = estimate_night_vision_mode(frame_original)
        use_nv = True if self.nv_mode == "nv" else False if self.nv_mode == "color" else auto_is_nv

        # Enhancement pipeline: gamma first, then CLAHE.
        proc = apply_gamma(frame_bgr, self.gamma.get(), self.lut_cache)
        if self.var_enh.get() == 1:
            proc = enhance_clahe(proc)

        # Run denoise only if enabled and allowed by NV gating.
        # When denoise_only_when_nv is True, denoise runs only on NV frames for performance.
        do_denoise = (self.var_den.get() == 1) and ((not self.denoise_only_when_nv) or use_nv)

        if do_denoise:
            if self.source.get("mode") == "image":
                # Still image: compute denoise once and reuse for all updates.
                if self._denoise_cache is None:
                    self._denoise_cache = self._denoise_fast(proc)
                proc = self._denoise_cache.copy()
            else:
                # Video/webcam: compute denoise every N frames and reuse cached output in between.
                if (self._denoise_cache is None) or ((self.frame_i - self._denoise_cache_frame_i) >= self.denoise_every_n):
                    self._denoise_cache = self._denoise_fast(proc)
                    self._denoise_cache_frame_i = self.frame_i
                proc = self._denoise_cache.copy()
        else:
            # If denoise is not used, clear cache so the next enable starts fresh.
            self._denoise_cache = None
            self._denoise_cache_frame_i = -10**9

        # Unsharp after denoise (typical ordering: reduce noise then restore edges).
        if self.var_ush.get() == 1:
            proc = restore_unsharp(proc)

        # dlib detector uses grayscale; correlation tracker uses RGB.
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)

        # Run face detection periodically; between detections, update trackers.
        detect_every = max(1, int(round(self.detect_every.get())))
        do_detect = (self.source.get("mode") == "image") or (self.frame_i % detect_every == 0) or (len(self.tracks) == 0)

        if do_detect:
            dets = self.detector(gray, 0)
            self.tracks = []
            for d in dets:
                tr = dlib.correlation_tracker()
                tr.start_track(rgb, d)
                self.tracks.append(Track(self.next_id, tr))
                self.next_id += 1
        else:
            for t in self.tracks:
                t.tracker.update(rgb)

        # Thresholds used by the mask classifiers (read from UI).
        thr_mouth_c = float(self.thr_mouth_color.get())
        thr_nose_c = float(self.thr_nose_color.get())
        thr_mouth_nv = float(self.thr_mouth_nv.get())
        thr_nose_nv = float(self.thr_nose_nv.get())

        # Blur parameters.
        blur_mode = self._blur_mode_int()
        blur_k = int(round(self.blur_k.get()))
        feather_k = int(round(self.feather_k.get()))

        # Counters for overlay summary.
        faces_total = masked = incorrect = nomask = 0

        # Output starts as processed frame, then blur and overlays are drawn on top.
        out = proc.copy()

        for t in self.tracks:
            # Convert tracker position into valid integer pixel rectangle.
            pos = t.tracker.get_position()
            x1, y1, x2, y2 = rect_from_pos(pos, w, h)
            if x2 <= x1 or y2 <= y1:
                continue

            faces_total += 1

            # Landmarks are predicted within the tracked face rectangle.
            rect = dlib.rectangle(x1, y1, x2, y2)
            shape = self.predictor(gray, rect)
            pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=np.int32)

            # Per-track classification state for hysteresis and smoothing.
            st = self._get_track_state(t.tid)
            prev_stable = st.get("stable") or ""

            # Apply hysteresis to thresholds:
            # - If previously MASK, increase thresholds to make it harder to switch away from MASK
            # - If previously not MASK, decrease thresholds to make it easier to switch into MASK
            if prev_stable == "MASK":
                mC = thr_mouth_c + self.hyst_color
                nC = thr_nose_c + self.hyst_color
                mN = thr_mouth_nv + self.hyst_nv
                nN = thr_nose_nv + self.hyst_nv
            else:
                mC = max(0.00, thr_mouth_c - self.hyst_color)
                nC = max(0.00, thr_nose_c - self.hyst_color)
                mN = max(0.00, thr_mouth_nv - self.hyst_nv)
                nN = max(0.00, thr_nose_nv - self.hyst_nv)

            # Classify on the processed frame so gamma/CLAHE/denoise/unsharp affect the features consistently.
            frame_for_cls = proc

            # Mask classification:
            # - In NV mode: use NV classifier directly
            # - In color mode: use color classifier, with NV classifier as a fallback texture cue
            if use_nv:
                raw_cls = classify_mask_nightvision(frame_for_cls, pts, mN, nN)
            else:
                skin_cls = classify_mask_color(frame_for_cls, pts, mC, nC)
                tex_cls = classify_mask_nightvision(frame_for_cls, pts, mN, nN)

                if skin_cls == "MASK":
                    raw_cls = "MASK"
                elif tex_cls == "MASK":
                    # If texture indicates coverage but color indicates skin,
                    # treat as INCORRECT when skin_cls suggests uncovered/ambiguous.
                    raw_cls = "INCORRECT" if skin_cls in ("NO_MASK", "INCORRECT") else "MASK"
                else:
                    raw_cls = skin_cls

            # Smooth the raw class over time using per-track vote history.
            cls = self._stable_class_update(t.tid, raw_cls)

            # Update counters used by on-screen overlay.
            if cls == "MASK":
                masked += 1
            elif cls == "INCORRECT":
                incorrect += 1
            else:
                nomask += 1

            # Decide whether to blur this face region based on UI setting and class.
            do_blur = (blur_mode == 1) or (blur_mode == 2 and cls in ("NO_MASK", "INCORRECT"))
            if do_blur:
                poly = face_polygon_from_landmarks(pts)
                m = polygon_mask((h, w), poly)
                m = feather_mask(m, feather_k)
                out = blur_with_mask(out, m, blur_k)

            # Debug overlays: bounding box and per-face label.
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                out,
                f"ID {t.tid} {cls}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # Optional landmark drawing for debugging/tuning.
            if self.show_landmarks.get() == 1:
                for (px, py) in pts:
                    cv2.circle(out, (int(px), int(py)), 1, (255, 0, 0), -1)

        # Cleanup old per-track histories.
        self._purge_track_states()

        # Global overlay: mode, counts, source label, recording indicator.
        mode_txt = "NV" if use_nv else "COLOR"
        src_txt = self.source.get("label", self.source.get("mode", ""))
        cv2.putText(
            out,
            f"[{mode_txt}] Faces:{faces_total} Mask:{masked} Incorrect:{incorrect} NoMask:{nomask}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            out,
            f"Source: {src_txt}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        if self.recording:
            cv2.putText(out, "REC", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        # Caller can request the NV decision for UI header updates.
        if return_use_nv:
            return out, use_nv
        return out
