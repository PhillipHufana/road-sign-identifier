# evidence_tab.py
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2

from config import PACKS_ROOT, REPORTS_DIR
from quality import basic_quality_diagnostics, retake_guidance, recommended_order
from restoration import restore_pipeline
from packs import SECTORS, pick_random_sector_image, random_fact
from helpers import to_tk, bgr_to_pil


class EvidenceTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.configure(style="Card.TFrame")

        self.img_uploaded = None
        self.img_restored = None

        self._tk_before = None
        self._tk_after = None

        self.diag = None
        self.tips = []
        self.order = []

        self.sector_var = tk.StringVar(value="Forest")

        self.guidance_text = tk.StringVar(value="Upload an image to get retake guidance.")
        self.diag_text = tk.StringVar(value="Diagnostics: —")
        self.order_text = tk.StringVar(value="Recommended order: —")
        self.fact_text = tk.StringVar(value="")

        self.location_var = tk.StringVar(value="")
        self.notes_var = tk.StringVar(value="")

        self.build()

    def build(self):
        # Left: controls + guidance; Mid: images; Right: report/export
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

        ttk.Label(left, text="Community Mode — Evidence Toolkit", style="Title.TLabel").pack(anchor="w", pady=(0, 8))
        ttk.Label(left, text="Upload a photo. The app checks usability, suggests retake tips, restores, and exports a report.",style="Body.TLabel", wraplength=320).pack(anchor="w", pady=(0, 12))

        # Sector selector
        ttk.Label(left, text="Sector Pack:", style="Body.TLabel").pack(anchor="w")
        sector_box = ttk.Combobox(left, textvariable=self.sector_var, values=SECTORS, state="readonly")
        sector_box.pack(fill="x", pady=(4, 10))

        row_btns = ttk.Frame(left, style="Card.TFrame")
        row_btns.pack(fill="x", pady=(0, 10))
        ttk.Button(row_btns, text="Upload Image", command=self.upload_image, style="Accent.TButton").pack(side="left", expand=True, fill="x", padx=(0, 6))
        ttk.Button(row_btns, text="Load Sample", command=self.load_sample_from_sector, style="Accent.TButton").pack(side="left", expand=True, fill="x", padx=(6, 0))

        ttk.Separator(left).pack(fill="x", pady=10)

        ttk.Label(left, text="Retake Guidance", style="Title.TLabel").pack(anchor="w", pady=(0, 6))
        ttk.Label(left, textvariable=self.guidance_text, style="Body.TLabel", wraplength=320).pack(anchor="w", pady=(0, 8))

        ttk.Label(left, textvariable=self.diag_text, style="Body.TLabel", wraplength=320).pack(anchor="w", pady=(8, 4))
        ttk.Label(left, textvariable=self.order_text, style="Body.TLabel", wraplength=320).pack(anchor="w", pady=(0, 8))

        ttk.Button(left, text="Restore Now", command=self.restore_now, style="Accent.TButton").pack(fill="x", pady=(8, 6))

        # MID: before/after
        ttk.Label(mid, text="Before / After", style="Title.TLabel").pack(anchor="w", pady=(0, 8))
        self.lbl_before = ttk.Label(mid, text="Before (Uploaded)")
        self.lbl_before.pack(anchor="center", pady=(8, 4))
        self.lbl_after = ttk.Label(mid, text="After (Restored)")
        self.lbl_after.pack(anchor="center", pady=(12, 4))

        ttk.Label(mid, textvariable=self.fact_text, style="Body.TLabel", wraplength=800).pack(anchor="center", pady=(18, 0))

        # RIGHT: report builder
        ttk.Label(right, text="Report Builder", style="Title.TLabel").pack(anchor="w", pady=(0, 8))
        ttk.Label(right, text="Optional details:", style="Body.TLabel").pack(anchor="w")

        ttk.Label(right, text="Location (optional):", style="Body.TLabel").pack(anchor="w", pady=(10, 2))
        loc_entry = ttk.Entry(right, textvariable=self.location_var)
        loc_entry.pack(fill="x")

        ttk.Label(right, text="Notes (optional):", style="Body.TLabel").pack(anchor="w", pady=(10, 2))
        self.notes_box = tk.Text(right, height=6, wrap="word", bg="#0f1218", fg="#e5e7eb", insertbackground="#e5e7eb")
        self.notes_box.pack(fill="x")

        ttk.Button(right, text="Export Report (PNG)", command=self.export_report, style="Accent.TButton").pack(fill="x", pady=(12, 6))
        ttk.Button(right, text="Save Restored Image", command=self.save_restored, style="Accent.TButton").pack(fill="x", pady=(0, 6))

        ttk.Label(right, text="Tip: Reports are saved to /reports/ for easy sharing.", style="Body.TLabel", wraplength=320).pack(anchor="w", pady=(12, 0))

    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Failed to read image.")
            return
        self.set_uploaded(img)

    def load_sample_from_sector(self):
        sector = self.sector_var.get()
        path = pick_random_sector_image(PACKS_ROOT, sector)
        if not path:
            messagebox.showinfo(
                "No sample images",
                f"No images found for sector '{sector}'.\n\nCreate:\n  {PACKS_ROOT}/"
                f"{sector.lower().replace(' ', '_')}/\nAnd add images."
            )
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Failed to read sample image.")
            return
        self.set_uploaded(img)

    def set_uploaded(self, img_bgr):
        self.img_uploaded = img_bgr
        self.img_restored = None
        self.diag = basic_quality_diagnostics(img_bgr)
        self.tips = retake_guidance(self.diag)
        self.order = recommended_order(self.diag)

        # Update text blocks
        tips_text = "\n".join([f"• {t}" for t in self.tips])
        self.guidance_text.set(tips_text)

        self.diag_text.set(
            "Diagnostics:\n"
            f"• Blur score: {self.diag['blur']:.1f} (lower=blurrier)\n"
            f"• Noise est.: {self.diag['noise']:.1f}\n"
            f"• Brightness: {self.diag['brightness']:.1f}\n"
            f"• Contrast: {self.diag['contrast']:.1f}"
        )
        self.order_text.set("Recommended order: " + " → ".join(self.order))

        # Sector fact
        self.fact_text.set("Sector note: " + random_fact(self.sector_var.get()))

        # Render
        self._tk_before = to_tk(self.img_uploaded, 900, 360)
        self.lbl_before.configure(image=self._tk_before)

        self.lbl_after.configure(image="", text="After (Restored)")
        self._tk_after = None

    def restore_now(self):
        if self.img_uploaded is None:
            messagebox.showinfo("Info", "Upload an image first.")
            return
        self.img_restored = restore_pipeline(self.img_uploaded)

        self._tk_after = to_tk(self.img_restored, 900, 360)
        self.lbl_after.configure(image=self._tk_after)

        messagebox.showinfo("Restored", "Restoration complete. You can now export a report.")

    def export_report(self):
        if self.img_uploaded is None:
            messagebox.showinfo("Info", "Upload an image first.")
            return
        if self.img_restored is None:
            messagebox.showinfo("Info", "Click Restore Now first.")
            return

        try:
            from report import export_report_png  # ✅ flat import (matches your project)
            sector = self.sector_var.get()
            location = self.location_var.get().strip()
            notes = self.notes_box.get("1.0", "end").strip()

            before_pil = bgr_to_pil(self.img_uploaded)
            after_pil = bgr_to_pil(self.img_restored)

            out_dir = os.path.abspath(REPORTS_DIR)   # ✅ absolute
            os.makedirs(out_dir, exist_ok=True)      # ✅ ensure folder exists

            path = export_report_png(
                out_dir=out_dir,
                sector=sector,
                location_text=location,
                notes_text=notes,
                original_pil=before_pil,
                restored_pil=after_pil,
                diagnostics=self.diag or {},
                recommendations=self.tips or [],
            )

            messagebox.showinfo("Report Exported ✅", f"Saved report here:\n\n{path}")

            # Optional: open folder now
            if messagebox.askyesno("Open Folder?", "Open the report folder now?"):
                os.startfile(os.path.dirname(path))  # Windows

        except Exception as e:
            messagebox.showerror("Export Failed ❌", f"Export error:\n\n{e}")


    def save_restored(self):
        if self.img_restored is None:
            messagebox.showinfo("Info", "Restore an image first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPG", "*.jpg *.jpeg")])
        if not path:
            return
        ok = cv2.imwrite(path, self.img_restored)
        if not ok:
            messagebox.showerror("Error", "Failed to save image.")
            return
        messagebox.showinfo("Saved", f"Saved restored image:\n{path}")
