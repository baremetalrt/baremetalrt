"""
Generate installer assets for Inno Setup:
  - icon.ico (multi-size app icon)
  - wizard-image.bmp (164x314 left panel — Inno Setup WizardImageFile)
  - wizard-small.bmp (55x55 top-right — Inno Setup WizardSmallImageFile)

Run: py -3.13 generate-assets.py
Requires: Pillow
"""

import os
from PIL import Image, ImageDraw, ImageFont

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# Brand palette — metallic silver
BG        = (18, 18, 22)     # dark base
SURFACE   = (28, 28, 34)     # card surface
DARK      = (38, 38, 46)     # icon bg
BORDER    = (58, 58, 68)     # borders
ACCENT_RGB= (192, 192, 210)  # metallic silver accent
SILVER_HI = (230, 230, 240)  # bright silver highlight
SILVER_MID= (160, 160, 175)  # mid silver
TEXT      = (230, 230, 238)  # bright text
TEXT_DIM  = (160, 160, 172)  # dimmed text
TEXT_DARK = (100, 100, 112)  # subtle text
WHITE     = (255, 255, 255)


def get_font(size, bold=False):
    names = []
    if bold:
        names += ["SpaceGrotesk-Bold.ttf", "SpaceGrotesk-SemiBold.ttf", "arialbd.ttf", "segoeui.ttf"]
    else:
        names += ["SpaceGrotesk-Regular.ttf", "arial.ttf", "segoeui.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def create_icon():
    """Multi-size .ico — modern flat icon. Dark bg, white geometric GPU chip shape.
    All solid fills, hard edges, no gradients. Rendered at 512px, downscaled with LANCZOS."""
    target_sizes = [16, 32, 48, 256]
    images = []

    for target in target_sizes:
        # Render at 512px for clean anti-aliasing on downscale
        S = 512
        img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        R = S // 8  # corner radius

        # Solid dark background
        draw.rounded_rectangle([0, 0, S - 1, S - 1], radius=R, fill=(22, 22, 28))

        # GPU chip shape — a centered rectangle with small "pins" on left/right edges
        # Main chip body
        chip_margin = S // 5
        chip = [chip_margin, chip_margin + 10, S - chip_margin, S - chip_margin - 10]
        draw.rectangle(chip, fill=(42, 42, 52))
        # Inner die (bright white square in center of chip)
        die_margin = S // 3
        die = [die_margin, die_margin + 10, S - die_margin, S - die_margin - 10]
        draw.rectangle(die, fill=(220, 220, 230))

        # Pins on left edge
        pin_w = 14
        pin_h = 18
        pin_gap = 8
        pin_x_left = chip[0] - pin_w
        pin_x_right = chip[2]
        num_pins = 7
        total_pin_h = num_pins * pin_h + (num_pins - 1) * pin_gap
        pin_start_y = (S - total_pin_h) // 2
        for i in range(num_pins):
            py = pin_start_y + i * (pin_h + pin_gap)
            draw.rectangle([pin_x_left, py, chip[0], py + pin_h], fill=(120, 120, 135))
            draw.rectangle([pin_x_right, py, pin_x_right + pin_w, py + pin_h], fill=(120, 120, 135))

        # "BM" on the die — bold, dark
        fsz = int(S * 0.18)
        font = get_font(fsz, bold=True)
        bbox = draw.textbbox((0, 0), "BM", font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = (S - tw) // 2 - bbox[0]
        ty = (S - th) // 2 - bbox[1]
        draw.text((tx, ty), "BM", fill=(22, 22, 28), font=font)

        # Downscale
        img = img.resize((target, target), Image.LANCZOS)
        images.append(img)
    path = os.path.join(ASSETS_DIR, "icon.ico")
    images[-1].save(path, format="ICO", sizes=[(s, s) for s in target_sizes], append_images=images[:-1])
    print(f"  icon.ico")


def create_wizard_image():
    """164x314 left-side wizard image. Clean: dark bg, metallic logo centered, copyright.
    No GPU graphic. Rendered at 8x for crisp text, saved at 2x for high-DPI."""
    TW, TH = 164, 314
    SCALE = 8
    W, H = TW * SCALE, TH * SCALE
    img = Image.new("RGB", (W, H), (10, 10, 14))
    draw = ImageDraw.Draw(img)

    # Metallic gradient stops — match site branding
    gradient_stops = [
        (0.0, (240, 240, 240)), (0.25, (168, 168, 176)),
        (0.5, (208, 208, 216)), (0.75, (128, 128, 136)), (1.0, (192, 192, 200)),
    ]

    def draw_metallic_text(text, font, x, y):
        """Render text with vertical metallic gradient."""
        bbox = draw.textbbox((0, 0), text, font=font)
        th = bbox[3] - bbox[1]
        tmp = Image.new("RGBA", (W, th + 20), (0, 0, 0, 0))
        ImageDraw.Draw(tmp).text((x, -bbox[1]), text, fill=(255, 255, 255), font=font)
        for yo in range(th + 20):
            t = yo / max(1, th)
            cr, cg, cb = gradient_stops[-1][1]
            for i in range(len(gradient_stops) - 1):
                t0, c0 = gradient_stops[i]
                t1, c1 = gradient_stops[i + 1]
                if t0 <= t <= t1:
                    f = (t - t0) / (t1 - t0)
                    cr = int(c0[0] + (c1[0] - c0[0]) * f)
                    cg = int(c0[1] + (c1[1] - c0[1]) * f)
                    cb = int(c0[2] + (c1[2] - c0[2]) * f)
                    break
            for xo in range(W):
                _, _, _, a = tmp.getpixel((xo, yo))
                if a > 0:
                    bg = img.getpixel((xo, y + yo))
                    img.putpixel((xo, y + yo), (
                        cr * a // 255 + bg[0] * (255 - a) // 255,
                        cg * a // 255 + bg[1] * (255 - a) // 255,
                        cb * a // 255 + bg[2] * (255 - a) // 255))

    # "BareMetalRT" metallic text — centered vertically, slightly larger
    font_brand = get_font(int(W * 0.08), bold=True)
    bbox = draw.textbbox((0, 0), "BareMetalRT", font=font_brand)
    tw_brand = bbox[2] - bbox[0]
    draw_metallic_text("BareMetalRT", font_brand, (W - tw_brand) // 2, (H - (bbox[3] - bbox[1])) // 2)

    # Copyright at bottom
    font_copy = get_font(24)
    copy_text = "\u00a9 2026 Bare Metal AI, Inc."
    bbox3 = draw.textbbox((0, 0), copy_text, font=font_copy)
    tw_copy = bbox3[2] - bbox3[0]
    draw.text(((W - tw_copy) // 2, H - 100), copy_text, fill=TEXT_DARK, font=font_copy)

    # Thin right edge
    draw.line([(W - 4, 0), (W - 4, H)], fill=(30, 30, 38), width=4)

    # Save at 2x native for high-DPI
    img = img.resize((TW * 2, TH * 2), Image.LANCZOS)
    path = os.path.join(ASSETS_DIR, "wizard-image.bmp")
    img.save(path, "BMP")
    print(f"  wizard-image.bmp")


def create_wizard_small():
    """55x55 top-right image. GPU chip design matching the app icon.
    Rendered at 880px (16x) for maximum crispness."""
    R = 880
    img = Image.new("RGB", (R, R), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Dark rounded background
    m = 32
    draw.rounded_rectangle([m, m, R - m, R - m], radius=96, fill=(14, 14, 18))

    # GPU chip body
    draw.rectangle([192, 208, 688, 672], fill=(42, 42, 52))
    # Die (bright center)
    draw.rectangle([288, 304, 592, 576], fill=(200, 200, 212))
    # Pins left/right
    for i in range(6):
        py = 240 + i * 72
        draw.rectangle([128, py, 192, py + 48], fill=(120, 120, 135))
        draw.rectangle([688, py, 752, py + 48], fill=(120, 120, 135))
    # "BM" on die
    font = get_font(160, bold=True)
    bbox = draw.textbbox((0, 0), "BM", font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((R - tw) // 2 - bbox[0], (R - th) // 2 - bbox[1]),
              "BM", fill=(30, 30, 38), font=font)

    img = img.resize((55, 55), Image.LANCZOS)
    path = os.path.join(ASSETS_DIR, "wizard-small.bmp")
    img.save(path, "BMP")
    print(f"  wizard-small.bmp")


if __name__ == "__main__":
    print("Generating installer assets...")
    create_icon()
    create_wizard_image()
    create_wizard_small()
    print(f"Done! Assets in {ASSETS_DIR}")
