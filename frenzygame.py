import os
import random
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
from skimage.metrics import structural_similarity as ssim


# ============================================================
# CONFIG
# ============================================================
FOREST_PACK_DIR = "forest_pack"

LEVELS = {
    1: {"name": "Level 1 (Easy)",   "alpha": 0.40, "time": 90},
    2: {"name": "Level 2 (Medium)", "alpha": 0.55, "time": 70},
    3: {"name": "Level 3 (Hard)",   "alpha": 0.70, "time": 50},
}

PARTIAL_SCORE = 680

FOREST_FACTS = [
    "Noisy satellite images can hide early signs of deforestation.",
    "Clear aerial images help detect illegal logging faster.",
    "Good image quality supports forest monitoring and biodiversity protection.",
    "Restored images can reveal land-clearing patterns that are easy to miss.",
    "Sharper drone footage helps map forest loss more accurately over time.",
    "Reducing blur in aerial images helps identify roads linked to forest encroachment."
]


# ============================================================
# UI helper: gradient using Canvas stripes
# ============================================================
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


def set_group_state(group_canvas: tk.Canvas, state: str):
    if state == "correct":
        draw_gradient(group_canvas, 380, 110, "#1b5e20", "#66bb6a")
    elif state == "partial":
        draw_gradient(group_canvas, 380, 110, "#f9a825", "#fff59d")
    elif state == "wrong":
        draw_gradient(group_canvas, 380, 110, "#b71c1c", "#ef5350")
    else:
        draw_gradient(group_canvas, 380, 110, "#2b2b2b", "#1f1f1f")


# ============================================================
# Image conversions for Tkinter display
# ============================================================
def to_tk(img_bgr, max_w=620, max_h=260):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    nw, nh = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    return ImageTk.PhotoImage(Image.fromarray(img_resized))


# ============================================================
# METRICS + SCORE
# ============================================================
def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse <= 1e-10:
        return 99.0
    return 20 * np.log10(255.0 / np.sqrt(mse))


def ssim_bgr(a_bgr, b_bgr):
    a = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2GRAY)
    return float(ssim(a, b, data_range=255))


def compute_score(original, restored, time_bonus=0):
    s = ssim_bgr(original, restored)
    p = psnr(original, restored)
    p_cap = min(p, 40.0)
    score = 700 * s + (p_cap * 7.5) + time_bonus
    return int(round(score)), s, p


# ============================================================
# Degradation utility
# ============================================================
def blend_strength(base_bgr, effected_bgr, alpha):
    return cv2.addWeighted(base_bgr, 1.0 - alpha, effected_bgr, alpha, 0)


# ============================================================
# DEGRADATION VARIANTS (3 per technique)
# ============================================================
def noise_gaussian(img):
    x = img.astype(np.float32)
    noise = np.random.normal(0, 22, x.shape).astype(np.float32)
    return np.clip(x + noise, 0, 255).astype(np.uint8)

def noise_saltpepper(img):
    out = img.copy()
    prob = 0.03
    rnd = np.random.rand(*img.shape[:2])
    out[rnd < prob] = 0
    out[rnd > 1 - prob] = 255
    return out

def noise_speckle(img):
    x = img.astype(np.float32)
    noise = np.random.randn(*x.shape).astype(np.float32)
    return np.clip(x + x * noise * 0.18, 0, 255).astype(np.uint8)

NOISE_VARIANTS = {"gaussian": noise_gaussian, "saltpepper": noise_saltpepper, "speckle": noise_speckle}


def blur_gaussian(img):
    return cv2.GaussianBlur(img, (13, 13), 0)

def blur_motion(img):
    k = 17
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)

def blur_defocus(img):
    return cv2.blur(img, (15, 15))

BLUR_VARIANTS = {"gaussian": blur_gaussian, "motion": blur_motion, "defocus": blur_defocus}


def vis_low_contrast(img):
    return cv2.convertScaleAbs(img, alpha=0.50, beta=40)

def vis_haze(img):
    fog = np.full(img.shape, 215, dtype=np.uint8)
    return cv2.addWeighted(img, 0.68, fog, 0.32, 0)

def vis_underexpose(img):
    return cv2.convertScaleAbs(img, alpha=0.70, beta=-35)

VIS_VARIANTS = {"low_contrast": vis_low_contrast, "haze": vis_haze, "underexpose": vis_underexpose}


# ============================================================
# RESTORATION PIPELINE (fixed)
# ============================================================
def restore_denoise(img_bgr):
    return cv2.medianBlur(img_bgr, 5)

def restore_sharpen(img_bgr):
    blur = cv2.GaussianBlur(img_bgr, (0, 0), 1.2)
    return cv2.addWeighted(img_bgr, 1.7, blur, -0.7, 0)

def restore_contrast(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge([l2, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def restore_pipeline(img_bgr):
    x = restore_denoise(img_bgr)
    x = restore_sharpen(x)
    x = restore_contrast(x)
    return x


# ============================================================
# ECO FACT CARD
# ============================================================
def show_eco_fact(parent, fact_text, headline):
    win = tk.Toplevel(parent)
    win.title("Eco Fact Card 🌳")
    win.resizable(False, False)

    frm = tk.Frame(win, padx=14, pady=12)
    frm.pack(fill="both", expand=True)

    tk.Label(frm, text="Forest Edition", font=("Segoe UI", 14, "bold")).pack(anchor="w")
    tk.Label(frm, text=headline, font=("Segoe UI", 10)).pack(anchor="w", pady=(2, 10))
    tk.Label(frm, text=f"“{fact_text}”", font=("Segoe UI", 11),
             wraplength=460, justify="left").pack(anchor="w")

    tk.Button(frm, text="Close", width=12, command=win.destroy).pack(anchor="e", pady=(12, 0))


# ============================================================
# MAIN APP
# ============================================================
class FilterFrenzyForestLevelMode:
    def __init__(self, root):
        self.root = root
        root.title("Filter Frenzy — Forest (LEVEL MODE)")
        root.geometry("1600x900")

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

        # default unclicked
        self.sel_noise = tk.StringVar(value="")
        self.sel_blur = tk.StringVar(value="")
        self.sel_vis = tk.StringVar(value="")

        self.streak = 0
        self.total_score = 0

        # Hint state
        self.hint_visible = False
        self.hint_text = tk.StringVar(value="")

        self.build_ui()

    # ---------------- UI ----------------
    def build_ui(self):
        main = tk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=12, pady=12)

        # Left controls
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

        # Center images
        center = tk.Frame(main)
        center.pack(side="left", fill="both", expand=True, padx=(0, 10))

        tk.Label(center, text="Images", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        imgs = tk.Frame(center)
        imgs.pack(fill="both", expand=True, pady=8)

        self.lbl_orig = tk.Label(imgs, text="Original", compound="top", font=("Segoe UI", 11, "bold"))
        self.lbl_orig.grid(row=0, column=0, padx=12, pady=(0, 14), sticky="n")

        self.lbl_deg = tk.Label(imgs, text="Degraded (Noise + Blur + Visibility)", compound="top",
                                font=("Segoe UI", 11, "bold"))
        self.lbl_deg.grid(row=1, column=0, padx=12, pady=(0, 14), sticky="n")

        # Right panel
        right = tk.Frame(main, width=560)
        right.pack(side="right", fill="y")

        tk.Label(right, text="Choose Variants", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 8))
        tk.Label(right, text="Green = correct | Yellow = partial credit | Red = wrong",
                 font=("Segoe UI", 10)).pack(anchor="w", pady=(0, 12))

        # groups: HORIZONTAL radios + labeled in full words
        self.grp_noise = self.make_variant_group_horizontal(
            right, "Noise", self.sel_noise,
            [("Gaussian", "gaussian"), ("Salt&Pepper", "saltpepper"), ("Speckle", "speckle")]
        )
        self.grp_blur = self.make_variant_group_horizontal(
            right, "Blur", self.sel_blur,
            [("Gaussian", "gaussian"), ("Motion", "motion"), ("Defocus", "defocus")]
        )
        self.grp_vis = self.make_variant_group_horizontal(
            right, "Visibility", self.sel_vis,
            [("Low Contrast", "low_contrast"), ("Haze/Fog", "haze"), ("Underexposed", "underexpose")]
        )

        self.grp_noise["frame"].pack(fill="x", pady=(0, 10))
        self.grp_blur["frame"].pack(fill="x", pady=(0, 10))
        self.grp_vis["frame"].pack(fill="x", pady=(0, 10))

        self.btn_submit = tk.Button(right, text="SUBMIT ANSWERS", height=2, command=self.submit_answers)
        self.btn_submit.pack(fill="x", pady=(12, 6))

        # Hint toggle button + hint display
        self.btn_hint = tk.Button(right, text="Show Hint (Toggle)", height=1, command=self.toggle_hint)
        self.btn_hint.pack(fill="x", pady=(0, 6))

        self.hint_label = tk.Label(right, textvariable=self.hint_text, font=("Segoe UI", 10),
                                   justify="left", wraplength=520)
        self.hint_label.pack(fill="x", pady=(0, 10))

        self.reset_groups(neutral=True, enable=False)
        self.set_hint_visible(False)

    def apply_level_choice(self):
        lv = int(self.level_var.get())
        self.level = lv
        self.alpha = LEVELS[lv]["alpha"]
        self.round_time = LEVELS[lv]["time"]
        self.timer_text.set(f"Time left: {self.round_time}s")
        self.status.set(f"Selected {LEVELS[lv]['name']}: α={self.alpha:.2f}, time={self.round_time}s.")

    def make_variant_group_horizontal(self, parent, title, var, options):
        frame = tk.Frame(parent, bd=0, highlightthickness=0)

        canvas = tk.Canvas(frame, width=380, height=110, bd=0, highlightthickness=0)
        canvas.pack(fill="x")
        set_group_state(canvas, "neutral")

        canvas.create_text(16, 22, anchor="w", text=title, fill="white", font=("Segoe UI", 15, "bold"))
        canvas.create_text(16, 52, anchor="w", text="Pick one:", fill="#dddddd", font=("Segoe UI", 10))

        opts_frame = tk.Frame(frame)
        opts_frame.pack(fill="x", pady=(6, 0))

        radios = []
        # HORIZONTAL layout
        for i, (label, value) in enumerate(options):
            rb = tk.Radiobutton(opts_frame, text=label, variable=var, value=value)
            rb.grid(row=0, column=i, padx=10, pady=2, sticky="w")
            radios.append(rb)

        return {"frame": frame, "canvas": canvas, "radios": radios, "var": var}

    def reset_groups(self, neutral=False, enable=True):
        for g in [self.grp_noise, self.grp_blur, self.grp_vis]:
            if neutral:
                set_group_state(g["canvas"], "neutral")
            for rb in g["radios"]:
                rb.config(state=("normal" if enable else "disabled"))
        self.btn_submit.config(state=("normal" if enable else "disabled"))
        self.btn_hint.config(state=("normal" if enable else "disabled"))

        # Default = unclicked every round
        if neutral:
            self.sel_noise.set("")
            self.sel_blur.set("")
            self.sel_vis.set("")

    # ---------------- Hint ----------------
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

        # Hint = correct variants (restore guidance)
        nice = {
            "gaussian": "Gaussian",
            "saltpepper": "Salt & Pepper",
            "speckle": "Speckle",
            "motion": "Motion",
            "defocus": "Defocus",
            "low_contrast": "Low Contrast",
            "haze": "Haze/Fog",
            "underexpose": "Underexposed"
        }

        hint_msg = (
            "Correct restore path (what to pick):\n"
            f"• Noise: {nice.get(self.truth_noise, self.truth_noise)}\n"
            f"• Blur: {nice.get(self.truth_blur, self.truth_blur)}\n"
            f"• Visibility: {nice.get(self.truth_vis, self.truth_vis)}"
        )
        self.hint_text.set(hint_msg)
        self.set_hint_visible(True)

    # ---------------- Controls ----------------
    def exit_app(self):
        self.stop_timer()
        self.root.destroy()

    def restart_round(self):
        if self.original is None:
            messagebox.showinfo("Info", "Load an image first.")
            return
        self.start_round()

    # ---------------- Loaders ----------------
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
            messagebox.showinfo("Forest Pack Missing",
                                f"Create '{FOREST_PACK_DIR}' folder and add forest images.")
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
        self.timer_text.set(f"Time left: {self.round_time}s")
        self.reset_groups(neutral=True, enable=False)
        self.set_hint_visible(False)
        self.render()

    # ---------------- Round ----------------
    def start_round(self):
        if self.original is None:
            messagebox.showinfo("Info", "Load an image first.")
            return

        self.apply_level_choice()

        # truth variants
        self.truth_noise = random.choice(list(NOISE_VARIANTS.keys()))
        self.truth_blur = random.choice(list(BLUR_VARIANTS.keys()))
        self.truth_vis = random.choice(list(VIS_VARIANTS.keys()))

        x = self.original.copy()

        x1 = NOISE_VARIANTS[self.truth_noise](x)
        x = blend_strength(x, x1, self.alpha)

        x2 = BLUR_VARIANTS[self.truth_blur](x)
        x = blend_strength(x, x2, self.alpha)

        x3 = VIS_VARIANTS[self.truth_vis](x)
        x = blend_strength(x, x3, self.alpha)

        self.degraded = x

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

    # ---------------- Timer ----------------
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

    # ---------------- Render ----------------
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
