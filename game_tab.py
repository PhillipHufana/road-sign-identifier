#game_tab.py
import os
import random
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2

from config import FOREST_PACK_DIR, LEVELS, PARTIAL_SCORE
from metrics import compute_score, psnr, ssim_bgr
from degradations import NOISE_VARIANTS, BLUR_VARIANTS, VIS_VARIANTS, apply_all_degradations
from restoration import restore_pipeline
from helpers import to_tk, set_group_state, nice_label, draw_gradient


class GameTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.configure(style="Card.TFrame")

        self.original = None
        self.degraded = None

        self.level = 1
        self.alpha = LEVELS[self.level]["alpha"]
        self.round_time = LEVELS[self.level]["time"]

        self.truth_noise = None
        self.truth_blur = None
        self.truth_vis = None

        self.round_active = False
        self.round_start = None
        self.timer_id = None

        self._tk_orig = None
        self._tk_deg = None

        self.sel_noise = tk.StringVar(value="")
        self.sel_blur = tk.StringVar(value="")
        self.sel_vis = tk.StringVar(value="")

        self.hint_visible = False
        self.hint_text = tk.StringVar(value="")

        self.base_metrics_text = tk.StringVar(value="Baseline: SSIM — | PSNR —")
        self.round_metrics_text = tk.StringVar(value="Round: SSIM — | PSNR — | SCORE —")
        self.status = tk.StringVar(value="Load a forest image to begin.")

        self.timer_text = tk.StringVar(value=f"Time left: {self.round_time}s")

        self.build()

    def build(self):
        # 3-column layout
        left = ttk.Frame(self, style="Card.TFrame")
        mid = ttk.Frame(self, style="Card.TFrame")
        right = ttk.Frame(self, style="Card.TFrame")

        left.grid(row=0, column=0, sticky="nsew", padx=(14, 10), pady=14)
        mid.grid(row=0, column=1, sticky="nsew", padx=10, pady=14)
        right.grid(row=0, column=2, sticky="nsew", padx=(10, 14), pady=14)

        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=0)
        self.rowconfigure(0, weight=1)

        # LEFT: controls
        ttk.Label(left, text="Game Mode — Filter Frenzy", style="Title.TLabel").pack(anchor="w", pady=(0, 8))
        ttk.Label(left, text="Objective: Guess the exact variant for Noise, Blur, and Visibility.",style="Body.TLabel", wraplength=320).pack(anchor="w", pady=(0, 10))

        ttk.Button(left, text="Upload Good Image", command=self.load_image, style="Accent.TButton").pack(fill="x", pady=4)
        ttk.Button(left, text="Load Random Forest Pack", command=self.load_from_pack, style="Accent.TButton").pack(fill="x", pady=4)

        ttk.Label(left, text="Select Level:", style="Body.TLabel").pack(anchor="w", pady=(14, 4))
        self.level_var = tk.IntVar(value=1)
        for lv in [1, 2, 3]:
            ttk.Radiobutton(
                left,
                text=f'{LEVELS[lv]["name"]}  | α={LEVELS[lv]["alpha"]:.2f} | {LEVELS[lv]["time"]}s',
                variable=self.level_var,
                value=lv,
                command=self.apply_level_choice
            ).pack(anchor="w", pady=2)

        ttk.Button(left, text="Start Round", command=self.start_round, style="Accent.TButton").pack(fill="x", pady=(12, 4))
        ttk.Button(left, text="Restart Round", command=self.restart_round, style="Accent.TButton").pack(fill="x", pady=4)
        ttk.Button(left, text="Exit App", command=self.exit_app, style="Accent.TButton").pack(fill="x", pady=4)

        ttk.Label(left, textvariable=self.status, style="Body.TLabel", wraplength=320).pack(anchor="w", pady=(16, 6))
        ttk.Label(left, textvariable=self.timer_text, style="Body.TLabel").pack(anchor="w", pady=(6, 0))

        # MID: images + metrics
        ttk.Label(mid, text="Images", style="Title.TLabel").pack(anchor="w", pady=(0, 8))

        self.lbl_orig = ttk.Label(mid, text="Original")
        self.lbl_orig.pack(anchor="center", pady=(4, 2))
        ttk.Label(mid, textvariable=self.base_metrics_text, style="Body.TLabel").pack(anchor="center", pady=(0, 10))

        self.lbl_deg = ttk.Label(mid, text="Degraded (Noise + Blur + Visibility)")
        self.lbl_deg.pack(anchor="center", pady=(4, 2))
        ttk.Label(mid, textvariable=self.round_metrics_text, style="Body.TLabel").pack(anchor="center", pady=(0, 10))

        # RIGHT: groups + submit + hint
        ttk.Label(right, text="Choose Variants", style="Title.TLabel").pack(anchor="w", pady=(0, 8))
        ttk.Label(right, text="Green=Correct | Yellow=Partial | Red=Wrong", style="Body.TLabel").pack(anchor="w", pady=(0, 10))

        self.grp_noise = self.make_group(right, "Noise", self.sel_noise,[("Gaussian", "gaussian"), ("Salt&Pepper", "saltpepper"), ("Speckle", "speckle")])
        self.grp_blur = self.make_group(right, "Blur", self.sel_blur,
                                        [("Gaussian", "gaussian"), ("Motion", "motion"), ("Defocus", "defocus")])
        self.grp_vis = self.make_group(right, "Visibility", self.sel_vis, [("Low Contrast", "low_contrast"), ("Haze/Fog", "haze"), ("Underexposed", "underexpose")])

        self.grp_noise.pack(fill="x", pady=(0, 10))
        self.grp_blur.pack(fill="x", pady=(0, 10))
        self.grp_vis.pack(fill="x", pady=(0, 10))

        self.btn_submit = ttk.Button(right, text="SUBMIT", command=self.submit_answers, style="Accent.TButton")
        self.btn_submit.pack(fill="x", pady=(12, 6))

        self.btn_hint = ttk.Button(right, text="Show Hint (Toggle)", command=self.toggle_hint, style="Accent.TButton")
        self.btn_hint.pack(fill="x", pady=(0, 6))

        ttk.Label(right, textvariable=self.hint_text, style="Body.TLabel", wraplength=420).pack(fill="x", pady=(0, 6))

        self.reset_groups(neutral=True, enable=False)

    def make_group(self, parent, title, var, options):
        frame = tk.Frame(parent, bg="#171a21", bd=0, highlightthickness=0)
        canvas = tk.Canvas(frame, width=380, height=110, bd=0, highlightthickness=0)
        canvas.pack(fill="x")
        set_group_state(canvas, "neutral")

        canvas.create_text(16, 22, anchor="w", text=title, fill="white", font=("Segoe UI", 15, "bold"))
        canvas.create_text(16, 52, anchor="w", text="Pick one:", fill="#cbd5e1", font=("Segoe UI", 10))

        opts = tk.Frame(frame, bg="#171a21")
        opts.pack(fill="x", pady=(6, 0))

        radios = []
        for i, (label, value) in enumerate(options):
            rb = tk.Radiobutton(opts, text=label, variable=var, value=value,
                                bg="#171a21", fg="#e5e7eb", selectcolor="#111318",
                                activebackground="#171a21", activeforeground="#ffffff")
            rb.grid(row=0, column=i, padx=10, pady=2, sticky="w")
            radios.append(rb)

        frame._canvas = canvas
        frame._radios = radios
        return frame

    def reset_groups(self, neutral=False, enable=True):
        for g in [self.grp_noise, self.grp_blur, self.grp_vis]:
            if neutral:
                set_group_state(g._canvas, "neutral")
            for rb in g._radios:
                rb.config(state=("normal" if enable else "disabled"))
        self.btn_submit.config(state=("normal" if enable else "disabled"))
        self.btn_hint.config(state=("normal" if enable else "disabled"))
        if neutral:
            self.sel_noise.set("")
            self.sel_blur.set("")
            self.sel_vis.set("")

    def apply_level_choice(self):
        lv = int(self.level_var.get())
        self.level = lv
        self.alpha = LEVELS[lv]["alpha"]
        self.round_time = LEVELS[lv]["time"]
        self.timer_text.set(f"Time left: {self.round_time}s")
        self.status.set(f"Selected {LEVELS[lv]['name']}.")

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Failed to read image.")
            return
        self.set_original(img)
        self.status.set("Image loaded. Choose a level and Start Round.")

    def load_from_pack(self):
        if not os.path.isdir(FOREST_PACK_DIR):
            messagebox.showinfo("Forest Pack Missing",
                                f"Create '{FOREST_PACK_DIR}' beside app.py and add forest images.")
            return
        files = [os.path.join(FOREST_PACK_DIR, f) for f in os.listdir(FOREST_PACK_DIR)if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        if not files:
            messagebox.showinfo("No Images Found", f"No images in '{FOREST_PACK_DIR}'.")
            return
        img = cv2.imread(random.choice(files))
        if img is None:
            messagebox.showerror("Error", "Failed to read pack image.")
            return
        self.set_original(img)
        self.status.set("Loaded a forest pack image. Choose a level and Start Round.")

    def set_original(self, img_bgr):
        self.original = img_bgr
        self.degraded = None
        self.round_active = False
        self.stop_timer()

        base_ssim = ssim_bgr(self.original, self.original)
        base_psnr = psnr(self.original, self.original)
        self.base_metrics_text.set(f"Baseline (Original vs Original): SSIM={base_ssim:.3f} | PSNR={base_psnr:.2f}")
        self.round_metrics_text.set("Round: SSIM — | PSNR — | SCORE —")

        self.render()
        self.reset_groups(neutral=True, enable=False)

    def start_round(self):
        if self.original is None:
            messagebox.showinfo("Info", "Load an image first.")
            return
        self.apply_level_choice()

        self.truth_noise = random.choice(list(NOISE_VARIANTS.keys()))
        self.truth_blur = random.choice(list(BLUR_VARIANTS.keys()))
        self.truth_vis = random.choice(list(VIS_VARIANTS.keys()))

        self.degraded = apply_all_degradations(self.original, self.alpha, self.truth_noise, self.truth_blur, self.truth_vis)
        self.round_metrics_text.set("Round: SSIM — | PSNR — | SCORE —")

        self.round_active = True
        self.round_start = time.time()

        self.hint_text.set("")
        self.hint_visible = False
        self.btn_hint.config(text="Show Hint (Toggle)")

        self.reset_groups(neutral=True, enable=True)
        self.status.set(f"Round started ({LEVELS[self.level]['name']}).")
        self.render()
        self.start_timer()

    def submit_answers(self):
        if not self.round_active or self.degraded is None:
            messagebox.showinfo("Info", "Start a round first.")
            return
        if not self.sel_noise.get() or not self.sel_blur.get() or not self.sel_vis.get():
            messagebox.showinfo("Missing choice", "Pick one option in Noise, Blur, and Visibility.")
            return

        self.round_active = False
        self.stop_timer()

        restored = restore_pipeline(self.degraded)

        elapsed = time.time() - self.round_start
        remaining = max(0, self.round_time - elapsed)
        time_bonus = int(remaining * 8)

        score, s, p = compute_score(self.original, restored, time_bonus=time_bonus)
        self.round_metrics_text.set(f"Round (Original vs Restored): SSIM={s:.3f} | PSNR={p:.2f} | SCORE={score}")

        ok_noise = (self.sel_noise.get() == self.truth_noise)
        ok_blur = (self.sel_blur.get() == self.truth_blur)
        ok_vis = (self.sel_vis.get() == self.truth_vis)

        correct_count = sum([ok_noise, ok_blur, ok_vis])

        def state_for(is_correct: bool):
            if is_correct:
                return "correct"
            return "partial" if score >= PARTIAL_SCORE else "wrong"

        set_group_state(self.grp_noise._canvas, state_for(ok_noise))
        set_group_state(self.grp_blur._canvas, state_for(ok_blur))
        set_group_state(self.grp_vis._canvas, state_for(ok_vis))

        for g in [self.grp_noise, self.grp_blur, self.grp_vis]:
            for rb in g._radios:
                rb.config(state="disabled")
        self.btn_submit.config(state="disabled")
        self.btn_hint.config(state="disabled")

        passed = (correct_count == 3) or (correct_count == 2 and score >= PARTIAL_SCORE)
        self.status.set(("✅ Passed" if passed else "❌ Failed") + f" — {correct_count}/3 correct.")

    def toggle_hint(self):
        if self.degraded is None or not self.truth_noise:
            messagebox.showinfo("Hint", "Start a round first.")
            return
        if self.hint_visible:
            self.hint_visible = False
            self.hint_text.set("")
            self.btn_hint.config(text="Show Hint (Toggle)")
            return

        self.hint_visible = True
        self.btn_hint.config(text="Hide Hint (Toggle)")
        self.hint_text.set(
            "Correct variants:\n"
            f"• Noise: {nice_label(self.truth_noise)}\n"
            f"• Blur: {nice_label(self.truth_blur)}\n"
            f"• Visibility: {nice_label(self.truth_vis)}"
        )

    def start_timer(self):
        self.stop_timer()
        self.tick_timer()

    def stop_timer(self):
        if self.timer_id is not None:
            try:
                self.after_cancel(self.timer_id)
            except Exception:
                pass
        self.timer_id = None

    def tick_timer(self):
        if not self.round_active:
            return
        elapsed = time.time() - self.round_start
        remaining = max(0, int(round(self.round_time - elapsed)))
        self.timer_text.set(f"Time left: {remaining}s")

        if remaining <= 0:
            self.round_active = False
            self.status.set("Time’s up. Start a new round.")
            self.reset_groups(neutral=False, enable=False)
            messagebox.showinfo("Time Up", "Time’s up! Click Start Round again.")
            return

        self.timer_id = self.after(200, self.tick_timer)

    def render(self):
        if self.original is not None:
            self._tk_orig = to_tk(self.original, 720, 360)
            self.lbl_orig.configure(image=self._tk_orig)
        if self.degraded is not None:
            self._tk_deg = to_tk(self.degraded, 720, 360)
            self.lbl_deg.configure(image=self._tk_deg)

    def exit_app(self):
        self.stop_timer()
        self.winfo_toplevel().destroy()
    
    def restart_round(self):
        if self.original is None:
            messagebox.showinfo("Info", "Load an image first.")
            return
        self.start_round()

