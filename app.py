# app.py
import os
import random
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2

from config import FOREST_PACK_DIR, LEVELS, PARTIAL_SCORE, FOREST_FACTS
from metrics import compute_score, psnr, ssim_bgr
from degradations import NOISE_VARIANTS, BLUR_VARIANTS, VIS_VARIANTS, apply_all_degradations
from restoration import restore_pipeline
from ui_helpers import to_tk, set_group_state, nice_label
from eco_facts import show_eco_fact


class FilterFrenzyForestLevelMode:
    def __init__(self, root):
        self.root = root
        root.title("Filter Frenzy — Forest (LEVEL MODE)")
        root.geometry("1600x920")

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

        self.streak = 0
        self.total_score = 0

        self.hint_visible = False
        self.hint_text = tk.StringVar(value="")

        self.base_metrics_text = tk.StringVar(value="Baseline: SSIM — | PSNR —")
        self.round_metrics_text = tk.StringVar(value="Round: SSIM — | PSNR — | SCORE —")

        self.build_ui()

    def build_ui(self):
        main = tk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=12, pady=12)

        left = tk.Frame(main, width=380)
        left.pack(side="left", fill="y", padx=(0, 10))

        tk.Label(left, text="Controls", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 8))
        tk.Button(left, text="Upload Good Image", command=self.load_image, height=2).pack(fill="x", pady=4)
        tk.Button(left, text="Load Random Forest Pack", command=self.load_from_pack, height=2).pack(fill="x", pady=4)

        tk.Label(left, text="Select Level:", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(10, 4))
        self.level_var = tk.IntVar(value=1)
        level_box = tk.Frame(left)
        level_box.pack(fill="x")
        for lv in [1, 2, 3]:
            tk.Radiobutton(
                level_box,
                text=f'{LEVELS[lv]["name"]}  | α={LEVELS[lv]["alpha"]:.2f} | {LEVELS[lv]["time"]}s',
                variable=self.level_var,
                value=lv,
                anchor="w",
                command=self.apply_level_choice
            ).pack(fill="x", pady=2)

        tk.Button(left, text="Start Round", command=self.start_round, height=2).pack(fill="x", pady=(10, 4))
        tk.Button(left, text="Restart Round", command=self.restart_round, height=2).pack(fill="x", pady=4)
        tk.Button(left, text="Exit", command=self.exit_app, height=2).pack(fill="x", pady=(4, 10))

        self.status = tk.StringVar(value="Load a forest image to begin.")
        tk.Label(left, textvariable=self.status, font=("Segoe UI", 10),
                 wraplength=360, justify="left").pack(anchor="w", pady=(14, 6))

        self.timer_text = tk.StringVar(value=f"Time left: {self.round_time}s")
        tk.Label(left, textvariable=self.timer_text, font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(6, 0))

        self.progress_text = tk.StringVar(value="Streak: 0 | Total Score: 0")
        tk.Label(left, textvariable=self.progress_text, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(6, 0))

        center = tk.Frame(main)
        center.pack(side="left", fill="both", expand=True, padx=(0, 10))

        tk.Label(center, text="Images", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        imgs = tk.Frame(center)
        imgs.pack(fill="both", expand=True, pady=8)

        self.lbl_orig = tk.Label(imgs, text="Original", compound="top", font=("Segoe UI", 11, "bold"))
        self.lbl_orig.grid(row=0, column=0, padx=12, pady=(0, 4), sticky="n")
        tk.Label(imgs, textvariable=self.base_metrics_text, font=("Segoe UI", 10)).grid(
            row=1, column=0, padx=12, pady=(0, 12), sticky="n"
        )

        self.lbl_deg = tk.Label(imgs, text="Degraded (Noise + Blur + Visibility)", compound="top",
                                font=("Segoe UI", 11, "bold"))
        self.lbl_deg.grid(row=2, column=0, padx=12, pady=(0, 4), sticky="n")
        tk.Label(imgs, textvariable=self.round_metrics_text, font=("Segoe UI", 10)).grid(
            row=3, column=0, padx=12, pady=(0, 12), sticky="n"
        )

        right = tk.Frame(main, width=560)
        right.pack(side="right", fill="y")

        tk.Label(right, text="Choose Variants", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 8))
        tk.Label(right, text="Green = correct | Yellow = partial credit | Red = wrong",
                 font=("Segoe UI", 10)).pack(anchor="w", pady=(0, 12))

        self.grp_noise = self.make_group(right, "Noise", self.sel_noise,
                                         [("Gaussian", "gaussian"), ("Salt&Pepper", "saltpepper"), ("Speckle", "speckle")])
        self.grp_blur = self.make_group(right, "Blur", self.sel_blur,
                                        [("Gaussian", "gaussian"), ("Motion", "motion"), ("Defocus", "defocus")])
        self.grp_vis = self.make_group(right, "Visibility", self.sel_vis,
                                       [("Low Contrast", "low_contrast"), ("Haze/Fog", "haze"), ("Underexposed", "underexpose")])

        self.grp_noise["frame"].pack(fill="x", pady=(0, 10))
        self.grp_blur["frame"].pack(fill="x", pady=(0, 10))
        self.grp_vis["frame"].pack(fill="x", pady=(0, 10))

        self.btn_submit = tk.Button(right, text="SUBMIT ANSWERS", height=2, command=self.submit_answers)
        self.btn_submit.pack(fill="x", pady=(12, 6))

        self.btn_hint = tk.Button(right, text="Show Hint (Toggle)", height=1, command=self.toggle_hint)
        self.btn_hint.pack(fill="x", pady=(0, 6))

        tk.Label(right, textvariable=self.hint_text, font=("Segoe UI", 10),
                 justify="left", wraplength=520).pack(fill="x", pady=(0, 10))

        self.reset_groups(neutral=True, enable=False)
        self.set_hint_visible(False)

    def make_group(self, parent, title, var, options):
        frame = tk.Frame(parent, bd=0, highlightthickness=0)
        canvas = tk.Canvas(frame, width=380, height=110, bd=0, highlightthickness=0)
        canvas.pack(fill="x")
        set_group_state(canvas, "neutral")

        canvas.create_text(16, 22, anchor="w", text=title, fill="white", font=("Segoe UI", 15, "bold"))
        canvas.create_text(16, 52, anchor="w", text="Pick one:", fill="#dddddd", font=("Segoe UI", 10))

        opts = tk.Frame(frame)
        opts.pack(fill="x", pady=(6, 0))

        radios = []
        for i, (label, value) in enumerate(options):
            rb = tk.Radiobutton(opts, text=label, variable=var, value=value)
            rb.grid(row=0, column=i, padx=10, pady=2, sticky="w")
            radios.append(rb)

        return {"frame": frame, "canvas": canvas, "radios": radios}

    def apply_level_choice(self):
        lv = int(self.level_var.get())
        self.level = lv
        self.alpha = LEVELS[lv]["alpha"]
        self.round_time = LEVELS[lv]["time"]
        self.timer_text.set(f"Time left: {self.round_time}s")
        self.status.set(f"Selected {LEVELS[lv]['name']}: α={self.alpha:.2f}, time={self.round_time}s.")

    def reset_groups(self, neutral=False, enable=True):
        for g in [self.grp_noise, self.grp_blur, self.grp_vis]:
            if neutral:
                set_group_state(g["canvas"], "neutral")
            for rb in g["radios"]:
                rb.config(state=("normal" if enable else "disabled"))
        self.btn_submit.config(state=("normal" if enable else "disabled"))
        self.btn_hint.config(state=("normal" if enable else "disabled"))

        if neutral:
            self.sel_noise.set("")
            self.sel_blur.set("")
            self.sel_vis.set("")

    def set_hint_visible(self, visible: bool):
        self.hint_visible = visible
        if not visible:
            self.hint_text.set("")
            self.btn_hint.config(text="Show Hint (Toggle)")
        else:
            self.btn_hint.config(text="Hide Hint (Toggle)")

    def toggle_hint(self):
        if self.degraded is None or not self.truth_noise:
            messagebox.showinfo("Hint", "Start a round first.")
            return
        if self.hint_visible:
            self.set_hint_visible(False)
            return

        self.hint_text.set(
            "Correct variants:\n"
            f"• Noise: {nice_label(self.truth_noise)}\n"
            f"• Blur: {nice_label(self.truth_blur)}\n"
            f"• Visibility: {nice_label(self.truth_vis)}"
        )
        self.set_hint_visible(True)

    def exit_app(self):
        self.stop_timer()
        self.root.destroy()

    def restart_round(self):
        if self.original is None:
            messagebox.showinfo("Info", "Load an image first.")
            return
        self.start_round()

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Failed to read image.")
            return
        self.set_original(img)
        self.status.set("Image loaded. Choose a level, then Start Round.")

    def load_from_pack(self):
        if not os.path.isdir(FOREST_PACK_DIR):
            messagebox.showinfo("Forest Pack Missing", f"Create '{FOREST_PACK_DIR}' folder and add forest images.")
            return

        files = [os.path.join(FOREST_PACK_DIR, f) for f in os.listdir(FOREST_PACK_DIR)
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        if not files:
            messagebox.showinfo("No Images Found", f"No images found in '{FOREST_PACK_DIR}'.")
            return

        img = cv2.imread(random.choice(files))
        if img is None:
            messagebox.showerror("Error", "Failed to read a pack image.")
            return
        self.set_original(img)
        self.status.set("Loaded a forest pack image. Choose a level, then Start Round.")

    def set_original(self, img_bgr):
        self.original = img_bgr
        self.degraded = None
        self.round_active = False
        self.stop_timer()

        base_ssim = ssim_bgr(self.original, self.original)
        base_psnr = psnr(self.original, self.original)
        self.base_metrics_text.set(f"Baseline (Original vs Original): SSIM={base_ssim:.3f} | PSNR={base_psnr:.2f}")
        self.round_metrics_text.set("Round: SSIM — | PSNR — | SCORE —")

        self.timer_text.set(f"Time left: {self.round_time}s")
        self.reset_groups(neutral=True, enable=False)
        self.set_hint_visible(False)
        self.render()

    def start_round(self):
        if self.original is None:
            messagebox.showinfo("Info", "Load an image first.")
            return

        self.apply_level_choice()

        self.truth_noise = random.choice(list(NOISE_VARIANTS.keys()))
        self.truth_blur = random.choice(list(BLUR_VARIANTS.keys()))
        self.truth_vis = random.choice(list(VIS_VARIANTS.keys()))

        self.degraded = apply_all_degradations(
            self.original, self.alpha, self.truth_noise, self.truth_blur, self.truth_vis
        )

        self.round_metrics_text.set("Round: SSIM — | PSNR — | SCORE —")

        self.round_active = True
        self.round_start = time.time()

        self.reset_groups(neutral=True, enable=True)
        self.set_hint_visible(False)
        self.status.set(f"Round started: {LEVELS[self.level]['name']} | α={self.alpha:.2f} | time={self.round_time}s.")
        self.render()
        self.start_timer()

    def submit_answers(self):
        if not self.round_active or self.degraded is None:
            messagebox.showinfo("Info", "Start a round first.")
            return
        if not self.sel_noise.get() or not self.sel_blur.get() or not self.sel_vis.get():
            messagebox.showinfo("Missing choice", "Pick one option in Noise, Blur, and Visibility before submitting.")
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

        set_group_state(self.grp_noise["canvas"], state_for(ok_noise))
        set_group_state(self.grp_blur["canvas"], state_for(ok_blur))
        set_group_state(self.grp_vis["canvas"], state_for(ok_vis))

        for g in [self.grp_noise, self.grp_blur, self.grp_vis]:
            for rb in g["radios"]:
                rb.config(state="disabled")
        self.btn_submit.config(state="disabled")
        self.btn_hint.config(state="disabled")

        passed = (correct_count == 3) or (correct_count == 2 and score >= PARTIAL_SCORE)

        if passed:
            self.streak += 1
            if self.level < 3:
                self.level += 1
                self.level_var.set(self.level)
                self.apply_level_choice()
        else:
            self.streak = 0

        self.total_score += score
        self.progress_text.set(f"Streak: {self.streak} | Total Score: {self.total_score}")

        verdict = f"{correct_count}/3 correct"
        pass_text = "✅ Level Passed" if passed else "❌ Level Failed"
        headline = f"{pass_text} | {verdict} | SCORE={score} | SSIM={s:.3f} | PSNR={p:.2f}"
        self.status.set(f"Round complete. {headline}")

        show_eco_fact(self.root, random.choice(FOREST_FACTS), headline=headline)

    def start_timer(self):
        self.stop_timer()
        self.tick_timer()

    def stop_timer(self):
        if self.timer_id is not None:
            try:
                self.root.after_cancel(self.timer_id)
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
            self.set_hint_visible(False)
            messagebox.showinfo("Time Up", "Time’s up! Click Start Round again.")
            return

        self.timer_id = self.root.after(200, self.tick_timer)

    def render(self):
        if self.original is not None:
            self._tk_orig = to_tk(self.original)
            self.lbl_orig.configure(image=self._tk_orig)
        if self.degraded is not None:
            self._tk_deg = to_tk(self.degraded)
            self.lbl_deg.configure(image=self._tk_deg)
        else:
            self.lbl_deg.configure(image="", text="Degraded (Noise + Blur + Visibility)")


if __name__ == "__main__":
    root = tk.Tk()
    app = FilterFrenzyForestLevelMode(root)
    root.mainloop()
