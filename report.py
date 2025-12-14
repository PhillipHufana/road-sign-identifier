# core/report.py
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

def bgr_to_pil_rgb(bgr):
    # bgr is already converted outside? We'll accept RGB PIL directly in caller usually.
    # kept for future extension
    raise NotImplementedError

def export_report_png(
    out_dir: str,
    sector: str,
    location_text: str,
    notes_text: str,
    original_pil: Image.Image,
    restored_pil: Image.Image,
    diagnostics: dict,
    recommendations: list,
):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{sector.replace(' ', '_').lower()}_{ts}.png"
    path = os.path.join(out_dir, filename)

    # Canvas
    W, H = 1400, 900
    canvas = Image.new("RGB", (W, H), (245, 246, 248))
    draw = ImageDraw.Draw(canvas)

    # Fonts (fallback-safe)
    try:
        title_font = ImageFont.truetype("arial.ttf", 28)
        font = ImageFont.truetype("arial.ttf", 18)
        small = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        title_font = ImageFont.load_default()
        font = ImageFont.load_default()
        small = ImageFont.load_default()

    # Header
    draw.text((30, 20), "Evidence Report (Community Mode)", font=title_font, fill=(20, 20, 20))
    draw.text((30, 60), f"Sector: {sector}", font=font, fill=(50, 50, 50))
    draw.text((30, 85), f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", font=font, fill=(50, 50, 50))
    draw.text((30, 110), f"Location (optional): {location_text.strip() or '—'}", font=font, fill=(50, 50, 50))

    # Image boxes
    box_y = 160
    img_w = 650
    img_h = 430

    def paste_fit(im, x, y):
        im2 = im.copy()
        im2.thumbnail((img_w, img_h))
        # center inside box
        bx = x + (img_w - im2.size[0]) // 2
        by = y + (img_h - im2.size[1]) // 2
        canvas.paste(im2, (bx, by))

    # Frames
    draw.rectangle((30, box_y, 30 + img_w, box_y + img_h), outline=(210, 210, 210), width=2)
    draw.rectangle((720, box_y, 720 + img_w, box_y + img_h), outline=(210, 210, 210), width=2)
    draw.text((30, box_y - 28), "Before (Uploaded)", font=font, fill=(30, 30, 30))
    draw.text((720, box_y - 28), "After (Restored)", font=font, fill=(30, 30, 30))

    paste_fit(original_pil, 30, box_y)
    paste_fit(restored_pil, 720, box_y)

    # Diagnostics + Tips
    info_y = 610
    draw.text((30, info_y), "Diagnostics (no-reference):", font=font, fill=(20, 20, 20))
    diag_lines = [
        f"Blur score (Var Laplacian): {diagnostics.get('blur', 0):.1f} (lower = blurrier)",
        f"Noise estimate: {diagnostics.get('noise', 0):.1f} (higher = noisier)",
        f"Brightness mean: {diagnostics.get('brightness', 0):.1f}",
        f"Contrast std: {diagnostics.get('contrast', 0):.1f}",
    ]
    for i, line in enumerate(diag_lines):
        draw.text((30, info_y + 28 + i * 22), line, font=small, fill=(60, 60, 60))

    draw.text((720, info_y), "Retake / Action Tips:", font=font, fill=(20, 20, 20))
    tips = recommendations[:2] if recommendations else ["—"]
    for i, t in enumerate(tips):
        draw.text((720, info_y + 28 + i * 22), f"• {t}", font=small, fill=(60, 60, 60))

    # Notes
    notes_y = 720
    draw.text((30, notes_y), "Notes (optional):", font=font, fill=(20, 20, 20))
    notes = notes_text.strip() or "—"
    # wrap roughly
    wrap = 95
    wrapped = [notes[i:i+wrap] for i in range(0, len(notes), wrap)]
    for i, line in enumerate(wrapped[:4]):
        draw.text((30, notes_y + 28 + i * 22), line, font=small, fill=(60, 60, 60))

    # Footer integrity
    draw.text((30, 860), "Integrity: Original preserved. Enhanced output is for clarity, not proof.", font=small, fill=(120, 120, 120))

    canvas.save(path)
    return path
