import os
import time
import threading
from collections import deque

import tkinter as tk
from tkinter import filedialog, ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

# -----------------------------
# Windows beep alarm (optional)
# -----------------------------
try:
    import winsound  # Windows-only
except ImportError:
    winsound = None


# ============================================================
# Fire + Smoke + Fog Early Detection (Feature-Based)
# Tkinter UI + OpenCV pipeline
#
# INPUTS SUPPORTED:
# 1) Live Camera feed
# 2) Upload Video file
# 3) Upload Image file (single-frame analysis)
#
# OUTPUT CLASSES:
# - FIRE   (visible flames): warm-color + bright + saturated + flicker/motion
# - SMOKE  (early risk): low-sat + motion + soft edges + persistence + growth
# - FOG    (false-alarm blocker): global coverage + low motion + soft edges
# - NORMAL
#
# NEW (THIS VERSION):
# - Score Thresholds (editable via Entry + slider):
#   * FireScore threshold (fire_score >= threshold)
#   * SmokeScore threshold (smoke_score >= threshold)
#   * FogScore threshold (optional guard; fog gate uses area/motion + score)
#
# - Any parameter change triggers "state reset" so detection stays responsive:
#   * resets BackgroundSubtractor MOG2
#   * resets optical flow prev frame
#   * resets persistence counters
#   * resets smoke growth reference
#   * resets stabilization accumulators
# ============================================================


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def safe_int(s: str, fallback: int) -> int:
    try:
        return int(float(s))
    except Exception:
        return fallback


def safe_float(s: str, fallback: float) -> float:
    try:
        return float(s)
    except Exception:
        return fallback


class AlarmManager:
    """Non-blocking alarm with cooldown. Uses winsound.Beep on Windows."""
    def __init__(self, cooldown_sec: float = 3.0):
        self.last_trigger_time = 0.0
        self.cooldown_sec = float(cooldown_sec)
        self.enabled = True

    def trigger(self, level: str):
        if not self.enabled:
            return
        now = time.time()
        if now - self.last_trigger_time < self.cooldown_sec:
            return
        self.last_trigger_time = now
        threading.Thread(target=self._beep, args=(level,), daemon=True).start()

    def _beep(self, level: str):
        if winsound is None:
            return
        if level == "FIRE":
            winsound.Beep(1400, 600)
            winsound.Beep(1400, 600)
        else:  # SMOKE
            winsound.Beep(1000, 400)


class FireSmokeFogDetector:
    def __init__(self):
        self.reset_state()

        # Event recording (prebuffer + post seconds)
        self.prebuffer = deque(maxlen=180)
        self.is_recording = False
        self.writer = None
        self.record_frames_left = 0
        self.last_saved_path = None

    def reset_state(self):
        """Reset ALL temporal state so changing thresholds doesn't get stuck."""
        self.bgs = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=False)

        self.smoke_persist = 0
        self.fire_persist = 0
        self.last_smoke_area_ratio = 0.0

        self.prev_for_flow = None

        self.prev_stab_gray = None
        self.acc_dx = 0.0
        self.acc_dy = 0.0

    # ---------------------------
    # Enhancement (safe low-light)
    # ---------------------------
    def enhance(self, bgr: np.ndarray, clahe_clip=2.0, clahe_grid=8, denoise_ksize=3) -> np.ndarray:
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(int(clahe_grid), int(clahe_grid)))
        y2 = clahe.apply(y)
        out = cv2.cvtColor(cv2.merge([y2, cr, cb]), cv2.COLOR_YCrCb2BGR)

        k = int(denoise_ksize)
        if k >= 3:
            if k % 2 == 0:
                k += 1
            out = cv2.GaussianBlur(out, (k, k), 0)
        return out

    # -----------------------------------
    # Stabilization (translation-only)
    # -----------------------------------
    def stabilize_translation(self, bgr: np.ndarray, smooth=0.85) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self.prev_stab_gray is None:
            self.prev_stab_gray = gray
            return bgr

        p0 = cv2.goodFeaturesToTrack(self.prev_stab_gray, maxCorners=120, qualityLevel=0.01, minDistance=10)
        if p0 is None:
            self.prev_stab_gray = gray
            return bgr

        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_stab_gray, gray, p0, None)
        if p1 is None or st is None:
            self.prev_stab_gray = gray
            return bgr

        good0 = p0[st.flatten() == 1]
        good1 = p1[st.flatten() == 1]
        if len(good0) < 10:
            self.prev_stab_gray = gray
            return bgr

        shifts = (good1 - good0).reshape(-1, 2)
        dx = float(np.median(shifts[:, 0]))
        dy = float(np.median(shifts[:, 1]))

        self.acc_dx = smooth * self.acc_dx + (1 - smooth) * dx
        self.acc_dy = smooth * self.acc_dy + (1 - smooth) * dy

        h, w = bgr.shape[:2]
        M = np.float32([[1, 0, -self.acc_dx], [0, 1, -self.acc_dy]])
        stabilized = cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        self.prev_stab_gray = gray
        return stabilized

    # ---------------------------
    # Smoke candidate segmentation
    # ---------------------------
    def smoke_mask(self, bgr: np.ndarray, sat_max: int, motion_sens: int, morph_ksize: int, min_area_px: int):
        h, w = bgr.shape[:2]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        S = hsv[:, :, 1]

        appearance = (S < sat_max).astype(np.uint8) * 255
        fg = self.bgs.apply(bgr)

        thr = int(200 - (motion_sens * 1.7))
        thr = max(30, min(200, thr))
        _, motion = cv2.threshold(fg, thr, 255, cv2.THRESH_BINARY)

        combined = cv2.bitwise_and(appearance, motion)

        k = int(morph_ksize)
        if k < 3:
            k = 3
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        kept = []
        mask = np.zeros((h, w), dtype=np.uint8)
        for c in contours:
            if cv2.contourArea(c) >= float(min_area_px):
                kept.append(c)
                cv2.drawContours(mask, [c], -1, 255, -1)

        return mask, kept

    # ---------------------------
    # Fire candidate segmentation
    # ---------------------------
    def fire_mask(self, bgr: np.ndarray, fire_v_min: int, fire_s_min: int,
                  h1_min: int, h1_max: int, h2_min: int, h2_max: int,
                  morph_ksize: int, min_area_px: int):
        h, w = bgr.shape[:2]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        H = hsv[:, :, 0]
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]

        bright = (V >= fire_v_min)
        sat = (S >= fire_s_min)

        m1 = (H >= h1_min) & (H <= h1_max)
        m2 = (H >= h2_min) & (H <= h2_max)
        hue = m1 | m2

        mask = (bright & sat & hue).astype(np.uint8) * 255

        k = int(morph_ksize)
        if k < 3:
            k = 3
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        kept = []
        out = np.zeros((h, w), dtype=np.uint8)
        for c in contours:
            if cv2.contourArea(c) >= float(min_area_px):
                kept.append(c)
                cv2.drawContours(out, [c], -1, 255, -1)

        return out, kept

    # ---------------------------
    # Feature extraction
    # ---------------------------
    def features(self, bgr: np.ndarray, mask: np.ndarray):
        h, w = bgr.shape[:2]
        area_ratio = float(np.count_nonzero(mask)) / float(h * w + 1e-9)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        S = hsv[:, :, 1]
        masked = S[mask > 0]
        mean_sat = float(masked.mean()) if masked.size > 0 else 255.0

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
        edge_mag = np.abs(edges)
        edge_density = float((edge_mag > 40).mean())

        motion_mag = 0.0
        if self.prev_for_flow is not None:
            prev_gray = cv2.cvtColor(self.prev_for_flow, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                pyr_scale=0.5, levels=2, winsize=15,
                                                iterations=2, poly_n=5, poly_sigma=1.2, flags=0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            if np.count_nonzero(mask) > 0:
                motion_mag = float(mag[mask > 0].mean())
            else:
                motion_mag = float(mag.mean())

        self.prev_for_flow = bgr.copy()
        return dict(area_ratio=area_ratio, mean_sat=mean_sat, edge_density=edge_density, motion_mag=motion_mag)

    # ---------------------------
    # Classification logic
    # ---------------------------
    def classify(
        self,
        smoke_feats: dict,
        fire_feats: dict,
        # gates
        fog_global_ratio: float,
        fog_motion_max: float,
        smoke_min_ratio: float,
        smoke_motion_min: float,
        smoke_persist_frames: int,
        smoke_growth_min: float,
        fire_min_ratio: float,
        fire_motion_min: float,
        fire_persist_frames: int,
        # score thresholds (NEW)
        fire_score_thresh: float,
        smoke_score_thresh: float,
        fog_score_thresh: float,
        use_fog_score_guard: bool = True
    ):
        """
        Priority: FIRE > FOG > SMOKE > NORMAL
        - FIRE: fire_score >= fire_score_thresh for fire_persist_frames
        - FOG:  global+low-motion and (optional) fog_score >= fog_score_thresh
        - SMOKE: smoke_score >= smoke_score_thresh for smoke_persist_frames AND grew
        """
        dbg = {}

        # ---- FIRE score ----
        fire_area = fire_feats["area_ratio"]
        fire_motion = fire_feats["motion_mag"]
        fire_edge = fire_feats["edge_density"]

        fire_score = 0.0
        fire_score += clamp01((fire_area - fire_min_ratio) / 0.05)
        fire_score += clamp01((fire_motion - fire_motion_min) / 0.8)
        fire_score += clamp01((fire_edge - 0.05) / 0.15)
        fire_score = clamp01(fire_score / 3.0)

        fire_like = fire_score >= fire_score_thresh
        self.fire_persist = self.fire_persist + 1 if fire_like else max(0, self.fire_persist - 1)

        dbg["fire_score"] = fire_score
        dbg["fire_persist"] = self.fire_persist

        if self.fire_persist >= int(fire_persist_frames):
            conf = clamp01(0.65 + 0.35 * fire_score)
            return "FIRE", conf, dbg

        # ---- FOG score + gate ----
        smoke_area = smoke_feats["area_ratio"]
        smoke_motion = smoke_feats["motion_mag"]
        smoke_edge = smoke_feats["edge_density"]

        fog_score = 0.0
        fog_score += clamp01((smoke_area - fog_global_ratio) / 0.35)
        fog_score += clamp01((fog_motion_max - smoke_motion) / (fog_motion_max + 1e-6))
        fog_score += clamp01((0.08 - smoke_edge) / 0.08)
        fog_score = clamp01(fog_score / 3.0)

        dbg["fog_score"] = fog_score

        fog_gate = (smoke_area >= fog_global_ratio) and (smoke_motion <= fog_motion_max)
        if use_fog_score_guard:
            fog_gate = fog_gate and (fog_score >= fog_score_thresh)

        if fog_gate:
            return "FOG", fog_score, dbg

        # ---- SMOKE score + temporal+growth gate ----
        smoke_sat = smoke_feats["mean_sat"]

        smoke_score = 0.0
        smoke_score += clamp01((smoke_area - smoke_min_ratio) / 0.10)
        smoke_score += clamp01((smoke_motion - smoke_motion_min) / 0.8)
        smoke_score += clamp01((90.0 - smoke_sat) / 90.0)
        smoke_score += clamp01((0.10 - smoke_edge) / 0.10)
        smoke_score = clamp01(smoke_score / 4.0)

        grew = (smoke_area - self.last_smoke_area_ratio) >= smoke_growth_min
        self.last_smoke_area_ratio = smoke_area

        smoke_like = smoke_score >= smoke_score_thresh
        self.smoke_persist = self.smoke_persist + 1 if smoke_like else max(0, self.smoke_persist - 1)

        dbg["smoke_score"] = smoke_score
        dbg["smoke_persist"] = self.smoke_persist
        dbg["smoke_grew"] = grew

        if self.smoke_persist >= int(smoke_persist_frames) and grew:
            conf = clamp01(0.60 + 0.40 * smoke_score)
            return "SMOKE", conf, dbg

        return "NORMAL", clamp01(smoke_score * 0.6), dbg

    # ---------------------------
    # Recording (event-triggered)
    # ---------------------------
    def push_prebuffer(self, frame_bgr: np.ndarray):
        self.prebuffer.append(frame_bgr.copy())

    def start_recording(self, out_dir: str, frame_size: tuple[int, int], fps: float, post_seconds: int = 10):
        ensure_dir(out_dir)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"event_{ts}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(out_path, fourcc, float(fps), frame_size)

        for fr in list(self.prebuffer):
            self.writer.write(fr)

        self.is_recording = True
        self.record_frames_left = int(max(1, post_seconds) * fps)
        self.last_saved_path = out_path
        return out_path

    def write_record_frame(self, frame_bgr: np.ndarray):
        if not self.is_recording or self.writer is None:
            return
        self.writer.write(frame_bgr)
        self.record_frames_left -= 1
        if self.record_frames_left <= 0:
            self.stop_recording()

    def stop_recording(self):
        if self.writer is not None:
            self.writer.release()
        self.writer = None
        self.is_recording = False
        self.record_frames_left = 0


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Fire + Smoke + Fog Early Detection (Feature-Based)")

        self.det = FireSmokeFogDetector()
        self.alarm = AlarmManager(cooldown_sec=3.0)

        self.cap = None
        self.running = False
        self.mode = "NONE"
        self.source_fps = 20.0

        self.output_dir = os.path.join(os.getcwd(), "events_output")
        ensure_dir(self.output_dir)

        # --- state reset flag (NEW) ---
        self.reset_requested = False

        # ---------- toggles ----------
        self.var_enhance = tk.BooleanVar(value=True)
        self.var_stabilize = tk.BooleanVar(value=False)
        self.var_smoke_overlay = tk.BooleanVar(value=True)
        self.var_fire_overlay = tk.BooleanVar(value=True)
        self.var_record = tk.BooleanVar(value=True)
        self.var_alarm = tk.BooleanVar(value=True)
        self.var_use_fog_score_guard = tk.BooleanVar(value=True)  # NEW

        # ---------- smoke segmentation ----------
        self.smoke_sat_max = tk.IntVar(value=85)
        self.motion_sens = tk.IntVar(value=55)
        self.smoke_morph_k = tk.IntVar(value=7)
        self.smoke_min_area = tk.IntVar(value=1200)

        # ---------- fog gate thresholds ----------
        self.fog_global_ratio = tk.DoubleVar(value=0.55)
        self.fog_motion_max = tk.DoubleVar(value=0.35)

        # ---------- smoke gate thresholds ----------
        self.smoke_min_ratio = tk.DoubleVar(value=0.03)
        self.smoke_motion_min = tk.DoubleVar(value=0.15)
        self.smoke_persist_frames = tk.IntVar(value=12)
        self.smoke_growth_min = tk.DoubleVar(value=0.004)

        # ---------- fire segmentation ----------
        self.fire_v_min = tk.IntVar(value=180)
        self.fire_s_min = tk.IntVar(value=120)
        self.h1_min = tk.IntVar(value=0)
        self.h1_max = tk.IntVar(value=35)
        self.h2_min = tk.IntVar(value=160)
        self.h2_max = tk.IntVar(value=179)
        self.fire_morph_k = tk.IntVar(value=7)
        self.fire_min_area = tk.IntVar(value=800)

        # ---------- fire gate thresholds ----------
        self.fire_min_ratio = tk.DoubleVar(value=0.01)
        self.fire_motion_min = tk.DoubleVar(value=0.12)
        self.fire_persist_frames = tk.IntVar(value=8)

        # ---------- enhancement params ----------
        self.clahe_clip = tk.DoubleVar(value=2.0)
        self.clahe_grid = tk.IntVar(value=8)
        self.denoise_k = tk.IntVar(value=3)

        # ---------- SCORE THRESHOLDS (NEW) ----------
        # These directly control "fire_like" and "smoke_like"
        self.fire_score_thresh = tk.DoubleVar(value=0.55)
        self.smoke_score_thresh = tk.DoubleVar(value=0.55)
        self.fog_score_thresh = tk.DoubleVar(value=0.60)

        self._build_ui()
        self._attach_change_watchers()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------------------------------
    # Change watchers: reset state (NEW)
    # ---------------------------------
    def _attach_change_watchers(self):
        watched = [
            # toggles that affect temporal state
            self.var_stabilize, self.var_enhance,

            # smoke seg
            self.smoke_sat_max, self.motion_sens, self.smoke_morph_k, self.smoke_min_area,

            # fog
            self.fog_global_ratio, self.fog_motion_max, self.fog_score_thresh, self.var_use_fog_score_guard,

            # smoke gates + smoke score
            self.smoke_min_ratio, self.smoke_motion_min, self.smoke_persist_frames, self.smoke_growth_min,
            self.smoke_score_thresh,

            # fire seg
            self.fire_v_min, self.fire_s_min, self.h1_max, self.h2_min, self.fire_morph_k, self.fire_min_area,

            # fire gates + fire score
            self.fire_min_ratio, self.fire_motion_min, self.fire_persist_frames, self.fire_score_thresh,

            # enhancement params
            self.clahe_clip, self.clahe_grid, self.denoise_k,
        ]

        def mark_reset(*_):
            # This is the key requirement: after threshold changes, don't get stuck on previous state.
            self.reset_requested = True

        for v in watched:
            try:
                v.trace_add("write", mark_reset)
            except Exception:
                # older Tk versions
                v.trace("w", mark_reset)

    def _apply_state_reset_if_needed(self):
        if not self.reset_requested:
            return
        self.det.reset_state()
        self.reset_requested = False

    # ---------------- UI ----------------
    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        top = ttk.Frame(main)
        top.grid(row=0, column=0, sticky="ew")

        ttk.Button(top, text="Open Camera", command=self.open_camera).grid(row=0, column=0, padx=4)
        ttk.Button(top, text="Open Video", command=self.open_video).grid(row=0, column=1, padx=4)
        ttk.Button(top, text="Open Image", command=self.open_image).grid(row=0, column=2, padx=4)
        ttk.Button(top, text="Stop", command=self.stop).grid(row=0, column=3, padx=4)

        ttk.Checkbutton(top, text="Enhancement", variable=self.var_enhance).grid(row=0, column=4, padx=10)
        ttk.Checkbutton(top, text="Stabilization", variable=self.var_stabilize).grid(row=0, column=5, padx=10)
        ttk.Checkbutton(top, text="Smoke Overlay", variable=self.var_smoke_overlay).grid(row=0, column=6, padx=10)
        ttk.Checkbutton(top, text="Fire Overlay", variable=self.var_fire_overlay).grid(row=0, column=7, padx=10)
        ttk.Checkbutton(top, text="Record Events", variable=self.var_record).grid(row=0, column=8, padx=10)
        ttk.Checkbutton(top, text="Alarm", variable=self.var_alarm).grid(row=0, column=9, padx=10)

        ttk.Button(top, text="Open Output Folder", command=self.open_output_folder).grid(row=0, column=10, padx=4)
        ttk.Button(top, text="Reset Detector State", command=self.force_reset).grid(row=0, column=11, padx=4)

        mid = ttk.Frame(main)
        mid.grid(row=1, column=0, sticky="nsew", pady=(10, 10))
        main.rowconfigure(1, weight=1)
        mid.columnconfigure(0, weight=3)
        mid.columnconfigure(1, weight=2)
        mid.rowconfigure(0, weight=1)

        self.video_label = ttk.Label(mid)
        self.video_label.grid(row=0, column=0, sticky="nsew")

        right = ttk.Frame(mid)
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        self.lbl_status = ttk.Label(right, text="Status: -", font=("Segoe UI", 13, "bold"))
        self.lbl_status.grid(row=0, column=0, sticky="w", pady=(0, 8))

        self.lbl_debug = ttk.Label(right, text="Debug: -", justify="left")
        self.lbl_debug.grid(row=1, column=0, sticky="w")

        self.lbl_saved = ttk.Label(right, text="Last saved: -", wraplength=360)
        self.lbl_saved.grid(row=2, column=0, sticky="w", pady=(8, 0))

        nb = ttk.Notebook(main)
        nb.grid(row=2, column=0, sticky="ew")

        tab_scores = ttk.Frame(nb, padding=10)
        tab_smoke = ttk.Frame(nb, padding=10)
        tab_fog = ttk.Frame(nb, padding=10)
        tab_fire = ttk.Frame(nb, padding=10)
        tab_enh = ttk.Frame(nb, padding=10)

        nb.add(tab_scores, text="Scores (NEW)")
        nb.add(tab_smoke, text="Smoke")
        nb.add(tab_fog, text="Fog")
        nb.add(tab_fire, text="Fire")
        nb.add(tab_enh, text="Enhancement")

        # ---- Scores tab (NEW) ----
        ttk.Label(tab_scores, text="Score thresholds: lower = more sensitive, higher = stricter").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        self._slider_float_with_entry(tab_scores, "FireScore Threshold", self.fire_score_thresh, 0.10, 0.90, 1, 0)
        self._slider_float_with_entry(tab_scores, "SmokeScore Threshold", self.smoke_score_thresh, 0.10, 0.90, 2, 0)
        self._slider_float_with_entry(tab_scores, "FogScore Threshold", self.fog_score_thresh, 0.10, 0.90, 3, 0)
        ttk.Checkbutton(tab_scores, text="Use FogScore Guard (recommended)", variable=self.var_use_fog_score_guard).grid(
            row=4, column=0, sticky="w", pady=(8, 0)
        )

        # ---- Smoke tab ----
        self._slider_with_entry(tab_smoke, "Smoke Sat Max (higher=more sensitive)", self.smoke_sat_max, 20, 160, 0, 0)
        self._slider_with_entry(tab_smoke, "Motion Sensitivity (higher=more sensitive)", self.motion_sens, 0, 100, 1, 0)
        self._slider_with_entry(tab_smoke, "Smoke Min Area (px) (lower=more sensitive)", self.smoke_min_area, 200, 20000, 2, 0)
        self._slider_with_entry(tab_smoke, "Smoke Morph Kernel", self.smoke_morph_k, 3, 21, 3, 0)

        self._slider_float_with_entry(tab_smoke, "Smoke Min Ratio", self.smoke_min_ratio, 0.01, 0.20, 0, 1)
        self._slider_float_with_entry(tab_smoke, "Smoke Motion Min", self.smoke_motion_min, 0.05, 1.2, 1, 1)
        self._slider_with_entry(tab_smoke, "Smoke Persist Frames (lower=faster)", self.smoke_persist_frames, 3, 60, 2, 1)
        self._slider_float_with_entry(tab_smoke, "Smoke Growth Min (lower=more sensitive)", self.smoke_growth_min, 0.001, 0.03, 3, 1)

        # ---- Fog tab ----
        self._slider_float_with_entry(tab_fog, "Fog Global Ratio (lower=more sensitive)", self.fog_global_ratio, 0.30, 0.90, 0, 0)
        self._slider_float_with_entry(tab_fog, "Fog Motion Max (higher=more fog)", self.fog_motion_max, 0.05, 0.80, 1, 0)

        # ---- Fire tab ----
        self._slider_with_entry(tab_fire, "Fire V Min (lower=more sensitive)", self.fire_v_min, 80, 255, 0, 0)
        self._slider_with_entry(tab_fire, "Fire S Min (lower=more sensitive)", self.fire_s_min, 50, 255, 1, 0)
        self._slider_with_entry(tab_fire, "Hue1 Max (0..max) (higher=more warm colors)", self.h1_max, 5, 80, 2, 0)
        self._slider_with_entry(tab_fire, "Hue2 Min (min..179) (lower=more warm colors)", self.h2_min, 120, 175, 3, 0)

        self._slider_with_entry(tab_fire, "Fire Min Area (px) (lower=more sensitive)", self.fire_min_area, 100, 20000, 0, 1)
        self._slider_with_entry(tab_fire, "Fire Morph Kernel", self.fire_morph_k, 3, 21, 1, 1)
        self._slider_float_with_entry(tab_fire, "Fire Min Ratio", self.fire_min_ratio, 0.001, 0.08, 2, 1)
        self._slider_float_with_entry(tab_fire, "Fire Motion Min", self.fire_motion_min, 0.05, 1.2, 3, 1)
        self._slider_with_entry(tab_fire, "Fire Persist Frames (lower=faster)", self.fire_persist_frames, 2, 40, 4, 0)

        # ---- Enhancement tab ----
        self._slider_float_with_entry(tab_enh, "CLAHE Clip", self.clahe_clip, 1.0, 4.0, 0, 0)
        self._slider_with_entry(tab_enh, "CLAHE Grid", self.clahe_grid, 4, 16, 1, 0)
        self._slider_with_entry(tab_enh, "Denoise K (odd)", self.denoise_k, 1, 9, 2, 0)

    # ---------- Slider + Entry helpers (NEW) ----------
    def _slider_with_entry(self, parent, label, var, vmin, vmax, r, c):
        frm = ttk.Frame(parent)
        frm.grid(row=r, column=c, padx=8, pady=6, sticky="ew")
        frm.columnconfigure(0, weight=1)

        ttk.Label(frm, text=label).grid(row=0, column=0, sticky="w")

        entry = ttk.Entry(frm, width=10)
        entry.grid(row=0, column=1, padx=(8, 0))
        entry.insert(0, str(var.get()))

        def on_scale(v):
            iv = int(float(v))
            var.set(iv)
            entry.delete(0, tk.END)
            entry.insert(0, str(iv))

        scale = ttk.Scale(frm, from_=vmin, to=vmax, orient="horizontal", command=on_scale)
        scale.set(var.get())
        scale.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        def on_enter(_evt):
            iv = safe_int(entry.get(), var.get())
            iv = max(int(vmin), min(int(vmax), iv))
            var.set(iv)
            scale.set(iv)

        entry.bind("<Return>", on_enter)

    def _slider_float_with_entry(self, parent, label, var, vmin, vmax, r, c):
        frm = ttk.Frame(parent)
        frm.grid(row=r, column=c, padx=8, pady=6, sticky="ew")
        frm.columnconfigure(0, weight=1)

        ttk.Label(frm, text=label).grid(row=0, column=0, sticky="w")

        entry = ttk.Entry(frm, width=10)
        entry.grid(row=0, column=1, padx=(8, 0))
        entry.insert(0, f"{var.get():.4f}")

        def on_scale(v):
            fv = round(float(v), 4)
            var.set(fv)
            entry.delete(0, tk.END)
            entry.insert(0, f"{fv:.4f}")

        scale = ttk.Scale(frm, from_=vmin, to=vmax, orient="horizontal", command=on_scale)
        scale.set(var.get())
        scale.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        def on_enter(_evt):
            fv = safe_float(entry.get(), var.get())
            fv = max(float(vmin), min(float(vmax), fv))
            fv = round(fv, 4)
            var.set(fv)
            scale.set(fv)

        entry.bind("<Return>", on_enter)

    # ---------------- input sources ----------------
    def open_camera(self):
        self._open_capture(0, mode="CAMERA")

    def open_video(self):
        path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if not path:
            return
        self._open_capture(path, mode="VIDEO")

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")]
        )
        if not path:
            return

        self.stop()
        self.mode = "IMAGE"
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Could not read image.")
            return
        img = self._resize_for_ui(img, max_w=1000)

        # Image: no alarm/recording (keeps it sane)
        vis, status_text, debug_text = self._process_frame(img, fps=1.0, allow_record=False, allow_alarm=False)
        self.lbl_status.config(text=status_text)
        self.lbl_debug.config(text=debug_text)
        self._show_frame(vis)

    def _open_capture(self, src, mode: str):
        self.stop()
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Unable to open source.")
            self.cap = None
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.source_fps = float(fps) if fps and 1.0 < fps < 120.0 else 20.0

        self.det = FireSmokeFogDetector()  # reset everything per run
        self.reset_requested = False
        self.running = True
        self.mode = mode
        self._loop()

    def stop(self):
        self.running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.det.stop_recording()
        self.mode = "NONE"

    def on_close(self):
        self.stop()
        self.root.destroy()

    def force_reset(self):
        self.reset_requested = True

    def open_output_folder(self):
        try:
            if os.name == "nt":
                os.startfile(self.output_dir)
            else:
                cmd = "open" if "darwin" in os.sys.platform else "xdg-open"
                os.system(f'{cmd} "{self.output_dir}"')
        except Exception:
            messagebox.showinfo("Info", f"Output folder: {self.output_dir}")

    # ---------------- main loop ----------------
    def _loop(self):
        if not self.running or self.cap is None:
            return

        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.stop()
            return

        frame = self._resize_for_ui(frame, max_w=1000)

        # IMPORTANT: apply detector state reset if any threshold changed
        self._apply_state_reset_if_needed()

        vis, status_text, debug_text = self._process_frame(
            frame, fps=self.source_fps, allow_record=True, allow_alarm=True
        )
        self.lbl_status.config(text=status_text)
        self.lbl_debug.config(text=debug_text)
        if self.det.last_saved_path:
            self.lbl_saved.config(text=f"Last saved: {self.det.last_saved_path}")

        self._show_frame(vis)

        delay_ms = int(1000 / max(5.0, self.source_fps))
        self.root.after(delay_ms, self._loop)

    # ---------------- processing pipeline ----------------
    def _process_frame(self, frame_bgr: np.ndarray, fps: float, allow_record: bool, allow_alarm: bool):
        # Always process from the ORIGINAL frame (not previously processed output)
        self.det.push_prebuffer(frame_bgr)

        proc = frame_bgr
        if self.var_enhance.get():
            proc = self.det.enhance(proc, self.clahe_clip.get(), self.clahe_grid.get(), self.denoise_k.get())

        if self.var_stabilize.get():
            proc = self.det.stabilize_translation(proc)

        smoke_mask, smoke_contours = self.det.smoke_mask(
            proc,
            sat_max=self.smoke_sat_max.get(),
            motion_sens=self.motion_sens.get(),
            morph_ksize=self.smoke_morph_k.get(),
            min_area_px=self.smoke_min_area.get()
        )
        smoke_feats = self.det.features(proc, smoke_mask)

        fire_mask, fire_contours = self.det.fire_mask(
            proc,
            fire_v_min=self.fire_v_min.get(),
            fire_s_min=self.fire_s_min.get(),
            h1_min=self.h1_min.get(),
            h1_max=self.h1_max.get(),
            h2_min=self.h2_min.get(),
            h2_max=self.h2_max.get(),
            morph_ksize=self.fire_morph_k.get(),
            min_area_px=self.fire_min_area.get()
        )
        fire_feats = self.det.features(proc, fire_mask)

        label, conf, dbg = self.det.classify(
            smoke_feats=smoke_feats,
            fire_feats=fire_feats,
            fog_global_ratio=float(self.fog_global_ratio.get()),
            fog_motion_max=float(self.fog_motion_max.get()),
            smoke_min_ratio=float(self.smoke_min_ratio.get()),
            smoke_motion_min=float(self.smoke_motion_min.get()),
            smoke_persist_frames=int(self.smoke_persist_frames.get()),
            smoke_growth_min=float(self.smoke_growth_min.get()),
            fire_min_ratio=float(self.fire_min_ratio.get()),
            fire_motion_min=float(self.fire_motion_min.get()),
            fire_persist_frames=int(self.fire_persist_frames.get()),
            fire_score_thresh=float(self.fire_score_thresh.get()),
            smoke_score_thresh=float(self.smoke_score_thresh.get()),
            fog_score_thresh=float(self.fog_score_thresh.get()),
            use_fog_score_guard=bool(self.var_use_fog_score_guard.get())
        )

        # Alarm
        self.alarm.enabled = self.var_alarm.get()
        if allow_alarm and self.var_alarm.get():
            if label == "FIRE":
                self.alarm.trigger("FIRE")
            elif label == "SMOKE":
                self.alarm.trigger("SMOKE")

        # Event-triggered recording
        if allow_record and self.var_record.get():
            if (label in ("FIRE", "SMOKE")) and (not self.det.is_recording):
                h, w = proc.shape[:2]
                self.det.start_recording(self.output_dir, (w, h), fps=fps, post_seconds=10)
            self.det.write_record_frame(proc)
        else:
            self.det.stop_recording()

        # Visualization
        vis = proc.copy()

        if self.var_smoke_overlay.get():
            overlay = np.zeros_like(vis)
            overlay[:, :, 2] = smoke_mask
            vis = cv2.addWeighted(vis, 1.0, overlay, 0.30, 0)

        if self.var_fire_overlay.get():
            overlay = np.zeros_like(vis)
            overlay[:, :, 1] = fire_mask
            vis = cv2.addWeighted(vis, 1.0, overlay, 0.30, 0)

        for c in smoke_contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)

        for c in fire_contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)

        color = (0, 255, 0)
        if label == "FOG":
            color = (255, 255, 0)
        elif label == "SMOKE":
            color = (0, 165, 255)
        elif label == "FIRE":
            color = (0, 0, 255)

        cv2.putText(vis, f"{label} ({conf:.2f})", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        # Show score thresholds on-screen (helps debugging)
        cv2.putText(
            vis,
            f"Thresh: Fire={self.fire_score_thresh.get():.2f} Smoke={self.smoke_score_thresh.get():.2f} Fog={self.fog_score_thresh.get():.2f}",
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
        )

        status_text = f"Status: {label} | Confidence: {conf:.2f}"

        debug_text = (
            f"SMOKE: area={smoke_feats['area_ratio']:.3f}, sat={smoke_feats['mean_sat']:.1f}, "
            f"edge={smoke_feats['edge_density']:.3f}, motion={smoke_feats['motion_mag']:.3f}\n"
            f"FIRE:  area={fire_feats['area_ratio']:.3f}, sat={fire_feats['mean_sat']:.1f}, "
            f"edge={fire_feats['edge_density']:.3f}, motion={fire_feats['motion_mag']:.3f}\n"
            f"scores: fire={dbg.get('fire_score', 0):.2f} (p={dbg.get('fire_persist', 0)}), "
            f"fog={dbg.get('fog_score', 0):.2f}, "
            f"smoke={dbg.get('smoke_score', 0):.2f} (p={dbg.get('smoke_persist', 0)}, grew={dbg.get('smoke_grew', False)})"
        )

        return vis, status_text, debug_text

    # ---------------- helpers ----------------
    def _resize_for_ui(self, frame, max_w=1000):
        h, w = frame.shape[:2]
        if w <= max_w:
            return frame
        s = max_w / float(w)
        nh = int(h * s)
        return cv2.resize(frame, (max_w, nh), interpolation=cv2.INTER_AREA)

    def _show_frame(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)


def main():
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.2)
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
