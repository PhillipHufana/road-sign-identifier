import time
import cv2
import dlib
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from collections import Counter, deque

from config import APP_TITLE, DISPLAY_MAX_W
from dlib_models import load_dlib_models
from ivp_enhance import apply_gamma, enhance_clahe, restore_denoise, restore_unsharp
from anonymize import face_polygon_from_landmarks, polygon_mask, feather_mask, blur_with_mask
from classify import estimate_night_vision_mode, classify_mask_color, classify_mask_nightvision
from tracking import Track, rect_from_pos
from sources import open_camera, open_media
from tk_image import bgr_to_tk_photo, make_placeholder


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, height=260, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.canvas = tk.Canvas(self, highlightthickness=0, bg="#F6F7F9")
        self.vscroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vscroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vscroll.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # give it a starting height, but allow resize
        self.canvas.configure(height=height)

        self.inner = ttk.Frame(self.canvas)
        self.inner_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        def _sync_scrollregion():
            # critical: use requested height of inner, not bbox("all")
            self.inner.update_idletasks()
            req_w = self.canvas.winfo_width()
            req_h = self.inner.winfo_reqheight()
            self.canvas.itemconfigure(self.inner_id, width=req_w)
            self.canvas.configure(scrollregion=(0, 0, req_w, req_h))

        self.inner.bind("<Configure>", lambda e: _sync_scrollregion())
        self.canvas.bind("<Configure>", lambda e: _sync_scrollregion())

        # mouse wheel
        self._bind_mousewheel(self.canvas)
        self._bind_mousewheel(self.inner)

    def _bind_mousewheel(self, widget):
        widget.bind("<Enter>", lambda e: self._activate_mousewheel())
        widget.bind("<Leave>", lambda e: self._deactivate_mousewheel())

    def _activate_mousewheel(self):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _deactivate_mousewheel(self):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, e):
        self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

    def _on_mousewheel_linux(self, e):
        self.canvas.yview_scroll(-1 if e.num == 4 else 1, "units")


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)

        # ---- Modern ttk styling ----
        self._setup_style()

        self.detector, self.predictor = load_dlib_models()

        self.source = {"mode": "none", "cap": None, "image": None, "fps": 20.0, "label": "No source"}
        self.writer = None
        self.recording = False
        self.out_video_path = None

        self.paused = False
        self.frame_i = 0
        self.tracks = []
        self.next_id = 1

        
        # ---- Stability knobs ----
        self.cls_hist_len = 9              # vote window (7–11 is typical)
        self.cls_change_ratio = 0.70       # need >=70% votes to switch stable label
        self.hyst_color = 0.03             # +/- to thr_*_color when switching states
        self.hyst_nv = 0.05                # +/- to thr_*_nv when switching states

        # per-track state: tid -> {hist, stable, last_seen}
        self.track_states = {}

        # ---- Denoise performance knobs ----
        self.denoise_every_n = 4           # compute denoise once every N frames
        self._denoise_cache = None
        self._denoise_cache_frame_i = -10**9



        self.lut_cache = {"gamma": None, "lut": None}
        self.nv_mode = "auto"  # "auto" | "nv" | "color"

        self.last_frame_out = make_placeholder("Opening webcam...")

        # UI state
        self.show_adv = tk.BooleanVar(value=False)

        # blur dropdown (replaces radiobuttons)
        self.blur_mode_str = tk.StringVar(value="Blur all faces")

        # parameters (kept same meaning)
        self.var_enh = tk.IntVar(value=1)
        self.var_den = tk.IntVar(value=0)
        self.var_ush = tk.IntVar(value=0)

        self.gamma = tk.DoubleVar(value=1.00)
        self.detect_every = tk.DoubleVar(value=10)

        self.blur_k = tk.DoubleVar(value=31)
        self.feather_k = tk.DoubleVar(value=11)

        self.thr_mouth_color = tk.DoubleVar(value=0.15)
        self.thr_nose_color = tk.DoubleVar(value=0.15)

        self.thr_mouth_nv = tk.DoubleVar(value=0.60)
        self.thr_nose_nv = tk.DoubleVar(value=0.60)

        self.show_landmarks = tk.IntVar(value=0)
        self.jpeg_q = tk.DoubleVar(value=75)

        # header vars
        self.header_source_var = tk.StringVar(value="Source: Webcam")
        self.header_mode_var = tk.StringVar(value="Mode: AUTO")
        self.header_rec_var = tk.StringVar(value="")

        # nv display label in control panel
        self.nv_label_var = tk.StringVar(value="AUTO")

        # status bar
        self.status_var = tk.StringVar(value="Starting...")

        self._build_ui()

        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

        # default: webcam
        self.on_webcam(silent=True)
        self.root.after(15, self.tick)

    def _get_track_state(self, tid: int):
        st = self.track_states.get(tid)
        if st is None:
            st = {"hist": deque(maxlen=self.cls_hist_len), "stable": None, "last_seen": self.frame_i}
            self.track_states[tid] = st
        return st

    def _stable_class_update(self, tid: int, raw_cls: str) -> str:
        st = self._get_track_state(tid)
        st["hist"].append(raw_cls)
        st["last_seen"] = self.frame_i

        counts = Counter(st["hist"])
        maxc = max(counts.values())
        winners = [k for k, v in counts.items() if v == maxc]

        prev = st["stable"]
        # tie-break: keep previous stable if possible
        maj = winners[0] if len(winners) == 1 else (prev if prev in winners else winners[0])

        # vote-hysteresis: require strong majority to switch
        if prev is None:
            st["stable"] = maj
        elif maj != prev:
            need = max(1, int(len(st["hist"]) * self.cls_change_ratio + 0.999))
            if maxc >= need:
                st["stable"] = maj

        return st["stable"]

    def _purge_track_states(self):
        # drop states for tracks not seen recently (prevents unbounded growth)
        ttl = max(30, int(round(self.detect_every.get())) * 3)  # frames
        dead = [tid for tid, st in self.track_states.items() if (self.frame_i - st.get("last_seen", 0)) > ttl]
        for tid in dead:
            self.track_states.pop(tid, None)



    # -------------------------
    # Styling
    # -------------------------

    
    def _setup_style(self):
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

        # ---- Preview (left) ----
        self.video_box = ttk.LabelFrame(self.container, text="Preview", padding=(8, 8, 8, 8))
        self.video_box.grid(row=1, column=0, sticky="nsew", padx=(0, GAP))
        self.container.grid_rowconfigure(1, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        # Keep tk.Label for fast image updates
        self.video_label = tk.Label(self.video_box, bg="#000000")
        self.video_label.grid(row=0, column=0, sticky="nsew")
        self.video_box.grid_rowconfigure(0, weight=1)
        self.video_box.grid_columnconfigure(0, weight=1)

        # ---- Control panel (right) ----
        self.ctrl_col = ttk.Frame(self.container)
        self.ctrl_col.grid(row=1, column=1, sticky="ns")
        self.ctrl_col.columnconfigure(0, weight=1)

        # ===== Source / Playback (Basic) =====
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

        # ===== Privacy / Output (Basic) =====
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

        # Blur strength slider (basic)
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

        # ===== Detection (Basic + Advanced toggle) =====
        box_det = ttk.LabelFrame(self.ctrl_col, text="Detection", padding=(10, 10, 10, 10))
        box_det.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        box_det.columnconfigure(0, weight=1)
        box_det.columnconfigure(1, weight=1)
        box_det.grid_rowconfigure(3, weight=1)  # row where the scroller is placed

        ttk.Label(box_det, text="Night Vision").grid(row=0, column=0, sticky="w")
        self.btn_nv = ttk.Button(box_det, text="Toggle NV Mode", command=self.on_toggle_nv)
        self.btn_nv.grid(row=0, column=1, sticky="ew")

        ttk.Label(box_det, text="Current").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.lbl_nv = ttk.Label(box_det, textvariable=self.nv_label_var)
        self.lbl_nv.grid(row=1, column=1, sticky="e", pady=(6, 0))

        self.adv_btn = ttk.Button(box_det, text="Advanced ▸", command=self.toggle_advanced)
        self.adv_btn.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        # Advanced frame (hidden by default)
        self.advanced_scroller = ScrollableFrame(box_det, height=300)
        self.advanced_scroller.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        self.advanced_scroller.grid_remove()

        # put advanced controls inside this:
        self.advanced_frame = self.advanced_scroller.inner

        self.advanced_frame.columnconfigure(0, weight=1)
        self.advanced_frame.columnconfigure(1, weight=1)
        

        # Advanced: enhancement/restoration toggles
        row0 = ttk.Frame(self.advanced_frame)
        row0.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Checkbutton(row0, text="CLAHE", variable=self.var_enh).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(row0, text="Denoise", variable=self.var_den).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(row0, text="Unsharp", variable=self.var_ush).pack(side="left")

        # Advanced sliders
        self._scale_row(self.advanced_frame, 1, "Gamma", self.gamma, 0.4, 2.5, 0.05, "{:0.2f}")
        self._scale_row(self.advanced_frame, 2, "Detect Every N", self.detect_every, 1, 30, 1, "{:0.0f}")
        self._scale_row(self.advanced_frame, 3, "Feather Edge", self.feather_k, 1, 51, 2, "{:0.0f}")

        # thresholds
        self._scale_row(self.advanced_frame, 4, "Thr Mouth (Color)", self.thr_mouth_color, 0.05, 0.60, 0.01, "{:0.2f}")
        self._scale_row(self.advanced_frame, 5, "Thr Nose  (Color)", self.thr_nose_color, 0.05, 0.60, 0.01, "{:0.2f}")
        self._scale_row(self.advanced_frame, 6, "Thr Mouth (NV)", self.thr_mouth_nv, 0.20, 1.00, 0.02, "{:0.2f}")
        self._scale_row(self.advanced_frame, 7, "Thr Nose  (NV)", self.thr_nose_nv, 0.20, 1.00, 0.02, "{:0.2f}")

        # landmarks + jpeg quality
        ttk.Checkbutton(self.advanced_frame, text="Show Landmarks", variable=self.show_landmarks).grid(
            row=8, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        self._scale_row(self.advanced_frame, 9, "JPEG Quality", self.jpeg_q, 10, 95, 1, "{:0.0f}")




        # ---- Status bar ----
        self.status_bar = ttk.Label(self.container, textvariable=self.status_var, anchor="w")
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))

    def _scale_row(self, parent, row, label, var, from_, to, step, fmt="{:0.2f}", help_text=None):
        """
        Creates a clean row: Label | Scale | Value
        ttk.Scale does not natively support resolution; we quantize in the callback.
        """
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text=label).grid(row=0, column=0, sticky="w")

        value_var = tk.StringVar()

        def quantize_and_update(val):
            try:
                f = float(val)
            except Exception:
                f = float(var.get())
            # quantize
            if step and step > 0:
                q = round((f - from_) / step) * step + from_
            else:
                q = f
            # clamp
            q = max(from_, min(to, q))
            var.set(q)
            value_var.set(fmt.format(q))

        # initialize
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
        self.status_var.set(s)

    def toggle_advanced(self):
        self.show_adv.set(not self.show_adv.get())
        if self.show_adv.get():
            self.advanced_scroller.grid()
            self.adv_btn.config(text="Advanced ▾")
        else:
            self.advanced_scroller.grid_remove()
            self.adv_btn.config(text="Advanced ▸")



    def on_toggle_nv(self):
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
        self.paused = not self.paused
        self.set_status(f"{'Paused' if self.paused else 'Running'} | {self.source.get('label','')}")

    def on_upload(self):
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
            self.paused = False
            self._stop_recording()
            self._release_source()

            self.last_frame_out = make_placeholder("Loading media...")
            self._refresh_preview()
            self._denoise_cache = None
            self._denoise_cache_frame_i = -10**9

            self.source = open_media(path)
            self._reset_tracking()

            # prime preview immediately
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
        
        self.last_frame_out = make_placeholder("Opening webcam...")
        self._denoise_cache = None
        self._denoise_cache_frame_i = -10**9
        self._refresh_preview()

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

        f = self._read_frame()
        self.last_frame_out = self._process_frame(f) if f is not None else make_placeholder("Webcam opened, no frame yet...")
        self.set_status(f"Using: {label}")

    def on_snapshot(self):
        if self.last_frame_out is None:
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = f"snapshot_{ts}.jpg"
        cv2.imwrite(path, self.last_frame_out, [int(cv2.IMWRITE_JPEG_QUALITY), int(round(self.jpeg_q.get()))])
        self.set_status(f"Saved snapshot: {path}")

    def on_toggle_record(self):
        if self.recording:
            self._stop_recording()
            return

        # Optional UX: disallow recording for still images
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
        self._stop_recording()
        self._release_source()
        self.root.destroy()

    # -------------------------
    # Internals
    # -------------------------
    def _reset_tracking(self):
        self.tracks = []
        self.next_id = 1
        self.frame_i = 0
        self.track_states.clear()
        


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

    def _update_header(self, use_nv: bool):
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

        # Also reflect in small panel label
        if self.nv_mode == "nv":
            self.nv_label_var.set("FORCED NV")
        elif self.nv_mode == "color":
            self.nv_label_var.set("FORCED COLOR")
        else:
            self.nv_label_var.set("AUTO")

    def _refresh_preview(self):
        if self.last_frame_out is None:
            return
        photo = bgr_to_tk_photo(self.last_frame_out, max_w=DISPLAY_MAX_W)
        self.video_label.configure(image=photo)
        self.video_label.image = photo

    # -------------------------
    # Main loop
    # -------------------------
    def tick(self):
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
                    # keep last preview; update status for video/webcam
                    if self.source.get("mode") in ("webcam", "video"):
                        self.set_status(f"No frame read ({self.source.get('label','')}).")

            # always refresh preview
            if self.last_frame_out is not None:
                self._refresh_preview()

            # disable record button for still images (unless already recording)
            if self.source.get("mode") == "image" and not self.recording:
                self.rec_btn.state(["disabled"])
            else:
                self.rec_btn.state(["!disabled"])

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
        if ret and frame is not None:
            return frame

        # End of video: rewind + rewatch
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
        mode_map = {
            "Off": 0,
            "Blur all faces": 1,
            "Blur only NoMask/Incorrect": 2,
        }
        return mode_map.get(self.blur_mode_str.get(), 1)

    def _process_frame(self, frame_bgr, return_use_nv=False):
        self.frame_i += 1
        h, w = frame_bgr.shape[:2]

        frame_original = frame_bgr.copy()

        proc = apply_gamma(frame_bgr, self.gamma.get(), self.lut_cache)
        if self.var_enh.get() == 1:
            proc = enhance_clahe(proc)
        if self.var_den.get() == 1:
            # For still images: compute once and reuse forever
            if self.source.get("mode") == "image":
                if self._denoise_cache is None:
                    self._denoise_cache = restore_denoise(proc)
                proc = self._denoise_cache.copy()
            else:
                # For video/webcam: compute every N frames, reuse in-between
                if (self._denoise_cache is None) or ((self.frame_i - self._denoise_cache_frame_i) >= self.denoise_every_n):
                    self._denoise_cache = restore_denoise(proc)
                    self._denoise_cache_frame_i = self.frame_i
                proc = self._denoise_cache.copy()
        else:
            # denoise disabled -> clear cache so it re-inits cleanly next time
            self._denoise_cache = None
            self._denoise_cache_frame_i = -10**9

        if self.var_ush.get() == 1:
            proc = restore_unsharp(proc)

        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)

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

        auto_is_nv = estimate_night_vision_mode(frame_original)
        use_nv = True if self.nv_mode == "nv" else False if self.nv_mode == "color" else auto_is_nv

        thr_mouth_c = float(self.thr_mouth_color.get())
        thr_nose_c = float(self.thr_nose_color.get())
        thr_mouth_nv = float(self.thr_mouth_nv.get())
        thr_nose_nv = float(self.thr_nose_nv.get())

        blur_mode = self._blur_mode_int()
        blur_k = int(round(self.blur_k.get()))
        feather_k = int(round(self.feather_k.get()))

        faces_total = masked = incorrect = nomask = 0
        out = proc.copy()

        for t in self.tracks:
            pos = t.tracker.get_position()
            x1, y1, x2, y2 = rect_from_pos(pos, w, h)
            if x2 <= x1 or y2 <= y1:
                continue

            faces_total += 1
            rect = dlib.rectangle(x1, y1, x2, y2)
            shape = self.predictor(gray, rect)
            pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=np.int32)

            # -----------------------------
            # NEW: hysteresis + smoothing
            # -----------------------------
            st = self._get_track_state(t.tid)
            prev_stable = st.get("stable") or ""

            # base thresholds from sliders (already computed above, but safe to use here too)
            # thr_mouth_c, thr_nose_c, thr_mouth_nv, thr_nose_nv should exist from above.

            if prev_stable == "MASK":
                # if we're already MASK, make it harder to leave MASK
                mC = thr_mouth_c + self.hyst_color
                nC = thr_nose_c  + self.hyst_color
                mN = thr_mouth_nv + self.hyst_nv
                nN = thr_nose_nv  + self.hyst_nv
            else:
                # if we're not MASK, make it easier to become MASK
                mC = max(0.00, thr_mouth_c - self.hyst_color)
                nC = max(0.00, thr_nose_c  - self.hyst_color)
                mN = max(0.00, thr_mouth_nv - self.hyst_nv)
                nN = max(0.00, thr_nose_nv  - self.hyst_nv)

            raw_cls = (
                classify_mask_nightvision(frame_original, pts, mN, nN)
                if use_nv
                else classify_mask_color(frame_original, pts, mC, nC)
            )

            # vote smoothing + vote-hysteresis
            cls = self._stable_class_update(t.tid, raw_cls)

            

            # -----------------------------
            # rest of your code unchanged
            # -----------------------------
            if cls == "MASK":
                masked += 1
            elif cls == "INCORRECT":
                incorrect += 1
            else:
                nomask += 1

            do_blur = (blur_mode == 1) or (blur_mode == 2 and cls in ("NO_MASK", "INCORRECT"))
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
        
        self._purge_track_states()

        mode_txt = "NV" if use_nv else "COLOR"
        src_txt = self.source.get("label", self.source.get("mode", ""))
        cv2.putText(out, f"[{mode_txt}] Faces:{faces_total} Mask:{masked} Incorrect:{incorrect} NoMask:{nomask}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(out, f"Source: {src_txt}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if self.recording:
            cv2.putText(out, "REC", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        if return_use_nv:
            return out, use_nv
        return out
    