import time
import cv2
import dlib
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

from config import APP_TITLE, DISPLAY_MAX_W
from dlib_models import load_dlib_models
from ivp_enhance import apply_gamma, enhance_clahe, restore_denoise, restore_unsharp
from anonymize import face_polygon_from_landmarks, polygon_mask, feather_mask, blur_with_mask
from classify import estimate_night_vision_mode, classify_mask_color, classify_mask_nightvision
from tracking import Track, rect_from_pos
from sources import open_camera, open_media
from tk_image import bgr_to_tk_photo, make_placeholder

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)

        self.detector, self.predictor = load_dlib_models()

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

        self._build_ui()

        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

        # default: webcam
        self.on_webcam(silent=True)
        self.root.after(15, self.tick)

    def _build_ui(self):
        self.container = tk.Frame(self.root, padx=10, pady=10)
        self.container.grid(row=0, column=0, sticky="nsew")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.video_box = tk.LabelFrame(self.container, text="Preview", padx=8, pady=8)
        self.video_box.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.video_label = tk.Label(self.video_box)
        self.video_label.grid(row=0, column=0, sticky="nsew")
        self.video_box.grid_rowconfigure(0, weight=1)
        self.video_box.grid_columnconfigure(0, weight=1)

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

        self.blur_mode = tk.IntVar(value=1)
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

    # ---- UI actions ----
    def set_status(self, s): self.status_var.set(s)

    def on_toggle_nv(self):
        if self.nv_mode == "auto":
            self.nv_mode = "nv"; self.nv_label_var.set("Mode: FORCED NV")
        elif self.nv_mode == "nv":
            self.nv_mode = "color"; self.nv_label_var.set("Mode: FORCED COLOR")
        else:
            self.nv_mode = "auto"; self.nv_label_var.set("Mode: AUTO")

    def on_pause(self):
        self.paused = not self.paused
        self.set_status(f"{'Paused' if self.paused else 'Running'} | {self.source.get('label','')}")

    def on_upload(self):
        path = filedialog.askopenfilename(
            title="Select an image or video",
            filetypes=[("Media files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp *.mp4 *.avi *.mov *.mkv *.wmv *.webm *.m4v"),
                       ("All files", "*.*")]
        )
        if not path:
            return
        try:
            self.paused = False
            self._stop_recording()
            self._release_source()
            self.source = open_media(path)
            self._reset_tracking()

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

        f = self._read_frame()
        self.last_frame_out = self._process_frame(f) if f is not None else make_placeholder("Webcam opened, no frame yet...")
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

    # ---- internals ----
    def _reset_tracking(self):
        self.tracks = []
        self.next_id = 1
        self.frame_i = 0

    def _release_source(self):
        cap = self.source.get("cap")
        if cap is not None:
            try: cap.release()
            except Exception: pass
        self.source["cap"] = None

    def _stop_recording(self):
        if self.writer is not None:
            try: self.writer.release()
            except Exception: pass
        self.writer = None
        self.recording = False
        self.out_video_path = None
        self.rec_btn.config(text="Start Recording")

    def tick(self):
        try:
            if not self.paused:
                frame = self._read_frame()
                if frame is not None:
                    out = self._process_frame(frame)
                    self.last_frame_out = out
                    if self.recording and self.writer is not None:
                        self.writer.write(out)

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
        if ret and frame is not None:
            return frame

        # ---- End of video: rewind + rewatch ----
        if mode == "video":
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            except Exception:
                pass

            # reset tracking so it re-detects cleanly from start
            self._reset_tracking()

            # try reading first frame again
            ret2, frame2 = cap.read()
            if ret2 and frame2 is not None:
                return frame2

        return None

    def _process_frame(self, frame_bgr):
        self.frame_i += 1
        h, w = frame_bgr.shape[:2]

        frame_original = frame_bgr.copy()

        proc = apply_gamma(frame_bgr, self.gamma.get(), self.lut_cache)
        if self.var_enh.get() == 1: proc = enhance_clahe(proc)
        if self.var_den.get() == 1: proc = restore_denoise(proc)
        if self.var_ush.get() == 1: proc = restore_unsharp(proc)

        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)

        detect_every = max(1, int(self.detect_every.get()))
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
            shape = self.predictor(gray, rect)
            pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=np.int32)

            cls = classify_mask_nightvision(frame_original, pts, thr_mouth_nv, thr_nose_nv) if use_nv else \
                  classify_mask_color(frame_original, pts, thr_mouth_c, thr_nose_c)

            if cls == "MASK": masked += 1
            elif cls == "INCORRECT": incorrect += 1
            else: nomask += 1

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

        mode_txt = "NV" if use_nv else "COLOR"
        src_txt = self.source.get("label", self.source.get("mode", ""))
        cv2.putText(out, f"[{mode_txt}] Faces:{faces_total} Mask:{masked} Incorrect:{incorrect} NoMask:{nomask}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(out, f"Source: {src_txt}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if self.recording:
            cv2.putText(out, "REC", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        return out
