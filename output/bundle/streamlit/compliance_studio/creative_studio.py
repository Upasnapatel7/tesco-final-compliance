"""
creative_studio.py
==================
Advanced Creative Studio features for dashboard.py.

Real PIL/ML implementations — nothing simulated:
  - Text effects   (shadow, outline, gradient fill, glow, emboss, neon)
  - Text styles    (banner, cursive-like, stencil, stamp, condensed bold)
  - Borders        (classic, rounded, double, film strip, luxury, neon glow)
  - Templates      (product hero, sale banner, lifestyle, luxury, minimal)
  - Animations     (fade-in, slide-in, zoom, pulse, shimmer) → real PIL GIF

Usage in dashboard.py:
    from creative_studio import (
        render_text_effect, apply_border, apply_template_overlay,
        generate_animation, TEMPLATES, BORDERS, TEXT_STYLES,
        render_studio_tab   # ← call this inside tab1 to add the UI
    )
"""

from __future__ import annotations
import io, math
from PIL import (Image, ImageDraw, ImageFont, ImageFilter,
                 ImageEnhance, ImageOps, ImageChops)
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  FONT LOADER  (Windows-safe)
# ─────────────────────────────────────────────────────────────────────────────
_WIN_STYLE_FONTS = {
    "Bold":         "C:/Windows/Fonts/arialbd.ttf",
    "Regular":      "C:/Windows/Fonts/arial.ttf",
    "Italic":       "C:/Windows/Fonts/ariali.ttf",
    "Bold Italic":  "C:/Windows/Fonts/arialbi.ttf",
    "Serif Bold":   "C:/Windows/Fonts/georgiab.ttf",
    "Serif Italic": "C:/Windows/Fonts/georgiai.ttf",
    "Condensed":    "C:/Windows/Fonts/arialbd.ttf",
    "Elegant":      "C:/Windows/Fonts/georgia.ttf",
    "Cursive":      "C:/Windows/Fonts/comic.ttf",
    "Mono":         "C:/Windows/Fonts/consola.ttf",
    "Impact":       "C:/Windows/Fonts/impact.ttf",
    "Trebuchet":    "C:/Windows/Fonts/trebucbd.ttf",
}

_LINUX_STYLE_FONTS = {
    "Bold":         "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "Regular":      "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "Italic":       "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "Bold Italic":  "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "Serif Bold":   "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf",
    "Serif Italic": "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "Condensed":    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "Elegant":      "/usr/share/fonts/truetype/google-fonts/Lora-Variable.ttf",
    "Cursive":      "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "Mono":         "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "Impact":       "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "Trebuchet":    "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
}

import platform as _plat
_IS_WIN = _plat.system() == "Windows"
_STYLE_FONTS = _WIN_STYLE_FONTS if _IS_WIN else _LINUX_STYLE_FONTS

TEXT_STYLES = [
    "Bold", "Regular", "Italic", "Bold Italic",
    "Serif Bold", "Serif Italic", "Elegant",
    "Cursive", "Impact", "Condensed", "Trebuchet", "Mono",
]

def _load_style_font(style: str, size: int) -> ImageFont.FreeTypeFont:
    path = _STYLE_FONTS.get(style, _STYLE_FONTS["Bold"])
    try:
        return ImageFont.truetype(path, max(8, size))
    except Exception:
        # fallback chain
        for p in list(_STYLE_FONTS.values()):
            try:
                return ImageFont.truetype(p, max(8, size))
            except Exception:
                pass
    return ImageFont.load_default()


# ─────────────────────────────────────────────────────────────────────────────
#  TEXT EFFECTS  (all real PIL operations)
# ─────────────────────────────────────────────────────────────────────────────
TEXT_EFFECTS = [
    "None", "Drop Shadow", "Hard Shadow", "Outline",
    "Double Outline", "Glow", "Neon Glow", "Emboss",
    "Gradient Fill", "Metallic", "Retro 3D",
]

def render_text_effect(
    canvas: Image.Image,
    text:   str,
    x: int, y: int,
    font:   ImageFont.FreeTypeFont,
    colour: tuple,          # (R,G,B)
    effect: str = "None",
    effect_colour: tuple = (0, 0, 0),
) -> Image.Image:
    """
    Draw text with a real PIL visual effect onto canvas.
    Returns modified canvas (RGBA).
    """
    if not text:
        return canvas
    if canvas.mode != "RGBA":
        canvas = canvas.convert("RGBA")

    W, H = canvas.size

    def _text_size(txt, fnt):
        tmp = ImageDraw.Draw(Image.new("RGBA", (1,1)))
        try:
            bb = tmp.textbbox((0,0), txt, font=fnt)
            return bb[2]-bb[0], bb[3]-bb[1]
        except Exception:
            sz = getattr(fnt, "size", 20)
            return len(txt)*sz//2, sz

    tw, th = _text_size(text, font)

    # ── None ──────────────────────────────────────────────────────────────────
    if effect == "None":
        d = ImageDraw.Draw(canvas)
        d.text((x, y), text, font=font, fill=colour)
        return canvas

    # ── Drop Shadow ───────────────────────────────────────────────────────────
    elif effect in ("Drop Shadow", "Hard Shadow"):
        layer = Image.new("RGBA", (W, H), (0,0,0,0))
        d = ImageDraw.Draw(layer)
        off = max(3, th // 10)
        # Shadow
        d.text((x+off, y+off), text, font=font, fill=(*effect_colour, 180))
        if effect == "Drop Shadow":
            layer = layer.filter(ImageFilter.GaussianBlur(off * 0.8))
        # Main text on top
        d2 = ImageDraw.Draw(layer)
        d2.text((x, y), text, font=font, fill=colour)
        return Image.alpha_composite(canvas, layer)

    # ── Outline ───────────────────────────────────────────────────────────────
    elif effect == "Outline":
        layer = Image.new("RGBA", (W, H), (0,0,0,0))
        d = ImageDraw.Draw(layer)
        r = max(2, th // 20)
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                if dx*dx + dy*dy <= r*r:
                    d.text((x+dx, y+dy), text, font=font, fill=(*effect_colour, 255))
        d.text((x, y), text, font=font, fill=colour)
        return Image.alpha_composite(canvas, layer)

    # ── Double Outline ────────────────────────────────────────────────────────
    elif effect == "Double Outline":
        layer = Image.new("RGBA", (W, H), (0,0,0,0))
        d = ImageDraw.Draw(layer)
        r1, r2 = max(3, th//14), max(2, th//22)
        for dx in range(-r1, r1+1):
            for dy in range(-r1, r1+1):
                if dx*dx + dy*dy <= r1*r1:
                    d.text((x+dx, y+dy), text, font=font, fill=(255,255,255,255))
        for dx in range(-r2, r2+1):
            for dy in range(-r2, r2+1):
                if dx*dx + dy*dy <= r2*r2:
                    d.text((x+dx, y+dy), text, font=font, fill=(*effect_colour, 255))
        d.text((x, y), text, font=font, fill=colour)
        return Image.alpha_composite(canvas, layer)

    # ── Glow ──────────────────────────────────────────────────────────────────
    elif effect == "Glow":
        glow_layer = Image.new("RGBA", (W, H), (0,0,0,0))
        dg = ImageDraw.Draw(glow_layer)
        gc = effect_colour if sum(effect_colour) > 50 else colour
        for r in [8, 5, 3]:
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    if dx*dx + dy*dy <= r*r:
                        alpha = int(80 * (1 - (dx*dx+dy*dy)/(r*r+1)))
                        dg.text((x+dx, y+dy), text, font=font, fill=(*gc, alpha))
        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(3))
        result = Image.alpha_composite(canvas, glow_layer)
        d2 = ImageDraw.Draw(result)
        d2.text((x, y), text, font=font, fill=colour)
        return result

    # ── Neon Glow ─────────────────────────────────────────────────────────────
    elif effect == "Neon Glow":
        neon_col = effect_colour if sum(effect_colour) > 50 else (0, 240, 255)
        # Multi-pass blur for neon bloom
        base = Image.new("RGBA", (W, H), (0,0,0,0))
        for r, alpha in [(12, 60), (7, 100), (4, 160), (2, 200), (0, 255)]:
            tmp = Image.new("RGBA", (W, H), (0,0,0,0))
            dt = ImageDraw.Draw(tmp)
            if r > 0:
                for dx in range(-r, r+1):
                    for dy in range(-r, r+1):
                        if dx*dx + dy*dy <= r*r:
                            dt.text((x+dx, y+dy), text, font=font,
                                    fill=(*neon_col, alpha))
            else:
                dt.text((x, y), text, font=font, fill=(*neon_col, 255))
            if r > 0:
                tmp = tmp.filter(ImageFilter.GaussianBlur(r // 2 + 1))
            base = Image.alpha_composite(base, tmp)
        return Image.alpha_composite(canvas, base)

    # ── Emboss ────────────────────────────────────────────────────────────────
    elif effect == "Emboss":
        # Draw text to temp, emboss via offset highlights/shadows
        layer = Image.new("RGBA", (W, H), (0,0,0,0))
        d = ImageDraw.Draw(layer)
        off = max(2, th // 18)
        d.text((x+off, y+off), text, font=font, fill=(0, 0, 0, 130))      # shadow
        d.text((x-off, y-off), text, font=font, fill=(255, 255, 255, 130)) # highlight
        d.text((x, y), text, font=font, fill=colour)
        return Image.alpha_composite(canvas, layer)

    # ── Gradient Fill ─────────────────────────────────────────────────────────
    elif effect == "Gradient Fill":
        # Create gradient image, mask with text shape
        text_mask = Image.new("L", (tw+4, th+4), 0)
        dm = ImageDraw.Draw(text_mask)
        dm.text((2, 2), text, font=font, fill=255)
        # Gradient: top=colour, bottom=effect_colour
        grad = Image.new("RGB", (tw+4, th+4))
        for row in range(th+4):
            t = row / max(th+3, 1)
            r = int(colour[0]*(1-t) + effect_colour[0]*t)
            g = int(colour[1]*(1-t) + effect_colour[1]*t)
            b = int(colour[2]*(1-t) + effect_colour[2]*t)
            for col in range(tw+4):
                grad.putpixel((col, row), (r, g, b))
        grad_rgba = grad.convert("RGBA")
        grad_rgba.putalpha(text_mask)
        layer = Image.new("RGBA", (W, H), (0,0,0,0))
        layer.paste(grad_rgba, (max(0,x-2), max(0,y-2)), grad_rgba)
        return Image.alpha_composite(canvas, layer)

    # ── Metallic ──────────────────────────────────────────────────────────────
    elif effect == "Metallic":
        # Silver/gold metallic via multiple offset layers with varying brightness
        layer = Image.new("RGBA", (W, H), (0,0,0,0))
        d = ImageDraw.Draw(layer)
        metal_steps = [
            (0, -2, (200, 200, 200, 180)),
            (0, -1, (230, 230, 230, 200)),
            (0,  0, colour),
            (0,  1, (80, 80, 80, 180)),
            (0,  2, (40, 40, 40, 120)),
        ]
        for ox, oy, col in metal_steps:
            d.text((x+ox, y+oy), text, font=font, fill=col)
        return Image.alpha_composite(canvas, layer)

    # ── Retro 3D ──────────────────────────────────────────────────────────────
    elif effect == "Retro 3D":
        layer = Image.new("RGBA", (W, H), (0,0,0,0))
        d = ImageDraw.Draw(layer)
        depth = max(3, th // 12)
        shadow_col = effect_colour
        for i in range(depth, 0, -1):
            alpha = int(180 * (1 - i / depth))
            d.text((x+i, y+i), text, font=font, fill=(*shadow_col, alpha))
        d.text((x, y), text, font=font, fill=colour)
        return Image.alpha_composite(canvas, layer)

    # fallback
    d = ImageDraw.Draw(canvas)
    d.text((x, y), text, font=font, fill=colour)
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
#  BORDERS  (real PIL drawing)
# ─────────────────────────────────────────────────────────────────────────────
BORDERS = [
    "None", "Classic", "Rounded", "Double Line", "Thick Accent",
    "Corner Brackets", "Film Strip", "Luxury Gold", "Neon",
    "Dashed", "Watercolour Edge", "Stamp"
]

def apply_border(img: Image.Image, style: str,
                 colour: tuple = (0,0,0),
                 width: int = 8) -> Image.Image:
    """Apply a decorative border using real PIL drawing. Returns same-size image."""
    if style == "None":
        return img

    canvas = img.copy().convert("RGBA")
    d = ImageDraw.Draw(canvas)
    W, H = canvas.size
    w = max(3, width)

    if style == "Classic":
        for i in range(w):
            alpha = 255 - i * (60 // max(w,1))
            d.rectangle([i, i, W-1-i, H-1-i], outline=(*colour, alpha))

    elif style == "Rounded":
        r = int(min(W, H) * 0.04)
        for i in range(w):
            d.rounded_rectangle([i, i, W-1-i, H-1-i],
                                  radius=max(4, r-i), outline=(*colour, 220))

    elif style == "Double Line":
        d.rectangle([w, w, W-1-w, H-1-w], outline=(*colour, 255), width=2)
        d.rectangle([w//2, w//2, W-1-w//2, H-1-w//2], outline=(*colour, 180), width=1)

    elif style == "Thick Accent":
        d.rectangle([0, 0, W-1, H-1], outline=(*colour, 255), width=w*2)
        # Corner accents
        ca = w * 3
        for cx, cy in [(0,0),(W-ca,0),(0,H-ca),(W-ca,H-ca)]:
            d.rectangle([cx, cy, cx+ca, cy+ca], fill=(*colour, 200))

    elif style == "Corner Brackets":
        bl = int(min(W, H) * 0.08)
        bw = max(3, w)
        corners = [(0, 0), (W-bl, 0), (0, H-bl), (W-bl, H-bl)]
        for cx, cy in corners:
            ex, ey = cx + bl, cy + bl
            # Horizontal top arm
            d.rectangle([min(cx,ex), min(cy,cy+bw), max(cx,ex), max(cy,cy+bw)], fill=(*colour, 255))
            # Vertical left arm
            d.rectangle([min(cx,cx+bw), min(cy,ey), max(cx,cx+bw), max(cy,ey)], fill=(*colour, 255))

    elif style == "Film Strip":
        hole_h = int(H * 0.025)
        hole_w = int(W * 0.018)
        gap    = int(H * 0.048)
        strip_w = int(W * 0.045)
        # Left strip
        d.rectangle([0, 0, strip_w, H], fill=(*colour, 220))
        # Right strip
        d.rectangle([W-strip_w, 0, W, H], fill=(*colour, 220))
        # Holes
        for y_pos in range(gap, H-gap, gap*2):
            for sx in [int(strip_w*0.15), W-strip_w+int(strip_w*0.15)]:
                d.rounded_rectangle([sx, y_pos, sx+hole_w, y_pos+hole_h],
                                     radius=2, fill=(255,255,255,200))

    elif style == "Luxury Gold":
        gold1 = (212, 175, 55)
        gold2 = (255, 215, 0)
        gold3 = (184, 134, 11)
        d.rectangle([0, 0, W-1, H-1], outline=gold1, width=w*2)
        d.rectangle([w*2+2, w*2+2, W-1-w*2-2, H-1-w*2-2], outline=gold2, width=1)
        d.rectangle([w*2+5, w*2+5, W-1-w*2-5, H-1-w*2-5], outline=gold3, width=1)
        # Corner ornaments
        co = w * 4
        for cx, cy in [(0,0),(W-co,0),(0,H-co),(W-co,H-co)]:
            d.rectangle([cx, cy, cx+co, cy+2], fill=gold2)
            d.rectangle([cx, cy, cx+2, cy+co], fill=gold2)

    elif style == "Neon":
        # Multi-layer glow border
        neon = colour if sum(colour) > 50 else (0, 240, 255)
        for i, alpha in [(w*3, 40), (w*2, 80), (w, 160), (w//2+1, 255)]:
            glow_layer = Image.new("RGBA", (W, H), (0,0,0,0))
            dg = ImageDraw.Draw(glow_layer)
            dg.rectangle([i, i, W-1-i, H-1-i], outline=(*neon, alpha), width=2)
            glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(2))
            canvas = Image.alpha_composite(canvas, glow_layer)
            d = ImageDraw.Draw(canvas)

    elif style == "Dashed":
        dash_len = int(min(W,H) * 0.04)
        gap_len  = int(dash_len * 0.6)
        offset   = w // 2
        # Draw dashes on all 4 sides
        for side in range(4):
            pos = 0
            while True:
                if side == 0:    # top
                    x1, y1, x2, y2 = pos, offset, min(pos+dash_len, W), offset
                elif side == 1:  # bottom
                    x1, y1, x2, y2 = pos, H-1-offset, min(pos+dash_len, W), H-1-offset
                elif side == 2:  # left
                    x1, y1, x2, y2 = offset, pos, offset, min(pos+dash_len, H)
                else:            # right
                    x1, y1, x2, y2 = W-1-offset, pos, W-1-offset, min(pos+dash_len, H)
                d.line([(x1,y1),(x2,y2)], fill=(*colour,220), width=w)
                pos += dash_len + gap_len
                limit = W if side < 2 else H
                if pos >= limit: break

    elif style == "Watercolour Edge":
        # Soft irregular edge using noise + blur
        np.random.seed(42)
        edge = np.zeros((H, W), dtype=np.uint8)
        bw = w * 3
        # Top/bottom/left/right edge bands with random opacity
        for band_y in range(bw):
            noise = np.random.randint(0, 200, W, dtype=np.uint8)
            fade  = int(180 * (1 - band_y/bw))
            edge[band_y, :]     = np.minimum(noise, fade)
            edge[H-1-band_y, :] = np.minimum(noise, fade)
        for band_x in range(bw):
            noise = np.random.randint(0, 200, H, dtype=np.uint8)
            fade  = int(180 * (1 - band_x/bw))
            edge[:, band_x]     = np.maximum(edge[:, band_x], np.minimum(noise, fade))
            edge[:, W-1-band_x] = np.maximum(edge[:, W-1-band_x], np.minimum(noise, fade))
        edge_img = Image.fromarray(edge, "L").filter(ImageFilter.GaussianBlur(w))
        colour_layer = Image.new("RGBA", (W, H), (*colour, 0))
        colour_layer.putalpha(edge_img)
        canvas = Image.alpha_composite(canvas, colour_layer)

    elif style == "Stamp":
        # Dashed rectangle with corner circles
        dash = int(min(W,H)*0.035)
        gap  = int(dash*0.5)
        off  = w
        for side in range(4):
            pos = dash
            while True:
                if side == 0:    start = (pos, off);         end = (min(pos+dash,W-off), off)
                elif side == 1:  start = (pos, H-1-off);     end = (min(pos+dash,W-off), H-1-off)
                elif side == 2:  start = (off, pos);         end = (off, min(pos+dash,H-off))
                else:            start = (W-1-off, pos);     end = (W-1-off, min(pos+dash,H-off))
                d.line([start, end], fill=(*colour, 200), width=w)
                pos += dash + gap
                if pos >= (W if side < 2 else H) - off: break
        # Corner circles
        cr = w * 2
        for cx, cy in [(0,0),(W-cr*2,0),(0,H-cr*2),(W-cr*2,H-cr*2)]:
            d.ellipse([cx+cr//2, cy+cr//2, cx+cr*3//2, cy+cr*3//2],
                       fill=(*colour, 180))

    return canvas.convert("RGB") if img.mode == "RGB" else canvas


# ─────────────────────────────────────────────────────────────────────────────
#  TEMPLATES  (real PIL compositing)
# ─────────────────────────────────────────────────────────────────────────────
TEMPLATES = {
    "None":           {"desc": "No template"},
    "Sale Banner":    {"desc": "Bold diagonal sale ribbon + highlight band"},
    "Luxury Brand":   {"desc": "Dark elegant overlay with gold accents"},
    "Minimal Clean":  {"desc": "Crisp white space, centered single line"},
    "Lifestyle":      {"desc": "Warm gradient overlay, editorial feel"},
    "Social Promo":   {"desc": "Gradient header + footer bands"},
    "Polaroid":       {"desc": "White frame bottom caption area"},
    "Magazine Cover": {"desc": "Full bleed with title masthead bar"},
    "Product Launch": {"desc": "Bold colour block with diagonal cut"},
    "Festive":        {"desc": "Decorative corners + star burst accents"},
}

def apply_template_overlay(
    img:      Image.Image,
    template: str,
    primary:  tuple = (0, 83, 159),   # brand colour RGB
    accent:   tuple = (227, 24, 55),
) -> Image.Image:
    """
    Apply a named template overlay onto the creative image.
    Uses real PIL compositing — no rules, all pixel operations.
    """
    if template == "None":
        return img

    canvas = img.copy().convert("RGBA")
    W, H   = canvas.size
    ov     = Image.new("RGBA", (W, H), (0,0,0,0))
    d      = ImageDraw.Draw(ov)

    if template == "Sale Banner":
        # Diagonal ribbon top-right corner
        ribbon_w = int(W * 0.40)
        pts = [(W - ribbon_w, 0), (W, 0), (W, ribbon_w)]
        d.polygon(pts, fill=(*accent, 230))
        # Inner ribbon line
        shrink = int(ribbon_w * 0.08)
        pts2 = [(W - ribbon_w + shrink*2, 0), (W, 0), (W, ribbon_w - shrink*2)]
        d.polygon(pts2, fill=(255,255,255,40))
        # Bottom highlight band
        band_h = int(H * 0.10)
        for i in range(band_h):
            alpha = int(200 * (i / band_h))
            d.rectangle([0, H-band_h+i, W, H-band_h+i+1], fill=(*primary, alpha))

    elif template == "Luxury Brand":
        # Dark vignette overlay
        for r in range(min(W,H)//2, 0, -int(min(W,H)*0.02)):
            alpha = int(120 * (1 - r / (min(W,H)//2)))
            d.ellipse([W//2-r, H//2-r, W//2+r, H//2+r], fill=(0,0,0, max(0,alpha)))
        # Top and bottom black bars
        bar_h = int(H * 0.06)
        d.rectangle([0, 0, W, bar_h], fill=(0,0,0,200))
        d.rectangle([0, H-bar_h, W, H], fill=(0,0,0,200))
        # Gold accent lines
        gold = (212,175,55)
        d.line([(0, bar_h), (W, bar_h)], fill=(*gold,200), width=2)
        d.line([(0, H-bar_h), (W, H-bar_h)], fill=(*gold,200), width=2)

    elif template == "Minimal Clean":
        # Thin top bar
        d.rectangle([0, 0, W, int(H*0.006)], fill=(*primary, 255))
        # Thin bottom bar
        d.rectangle([0, H-int(H*0.006), W, H], fill=(*primary, 255))
        # Very faint side shadows
        side_w = int(W * 0.03)
        for i in range(side_w):
            alpha = int(40 * (1 - i/side_w))
            d.rectangle([i, 0, i+1, H], fill=(0,0,0,alpha))
            d.rectangle([W-1-i, 0, W-i, H], fill=(0,0,0,alpha))

    elif template == "Lifestyle":
        # Warm gradient overlay (top: transparent, bottom: warm tone)
        warm = (255, 200, 120)
        for i in range(H):
            t = i / H
            # Only in lower half
            if t > 0.4:
                alpha = int(100 * ((t - 0.4) / 0.6) ** 1.5)
                d.rectangle([0, i, W, i+1], fill=(*warm, alpha))
        # Left vignette
        for i in range(int(W*0.15)):
            alpha = int(60 * (1 - i/(W*0.15)))
            d.rectangle([i, 0, i+1, H], fill=(0,0,0,alpha))

    elif template == "Social Promo":
        # Gradient header band
        hdr_h = int(H * 0.18)
        for i in range(hdr_h):
            alpha = int(220 * (1 - i/hdr_h)**0.8)
            d.rectangle([0, i, W, i+1], fill=(*primary, alpha))
        # Footer band with accent
        ftr_h = int(H * 0.12)
        for i in range(ftr_h):
            alpha = int(220 * (i/ftr_h)**0.8)
            row   = H - ftr_h + i
            d.rectangle([0, row, W, row+1], fill=(*accent, alpha))

    elif template == "Polaroid":
        # White frame
        pad_side  = int(W * 0.05)
        pad_top   = int(H * 0.05)
        pad_bot   = int(H * 0.18)   # extra bottom for caption
        d.rectangle([0, 0, W, H], fill=(0,0,0,0))   # clear
        # White border
        d.rectangle([0, 0, pad_side, H], fill=(255,255,255,255))
        d.rectangle([W-pad_side, 0, W, H], fill=(255,255,255,255))
        d.rectangle([0, 0, W, pad_top], fill=(255,255,255,255))
        d.rectangle([0, H-pad_bot, W, H], fill=(255,255,255,255))
        # Subtle shadow on inner edge
        for i in range(int(min(W,H)*0.01)):
            alpha = int(80*(1-i/(min(W,H)*0.01)))
            d.rectangle([pad_side+i, pad_top+i, W-pad_side-i, H-pad_bot-i],
                        outline=(0,0,0,alpha))

    elif template == "Magazine Cover":
        # Masthead bar at top
        mast_h = int(H * 0.12)
        d.rectangle([0, 0, W, mast_h], fill=(*primary, 240))
        # Thin accent below masthead
        d.rectangle([0, mast_h, W, mast_h+int(H*0.005)], fill=(*accent, 255))
        # Bottom info bar
        bot_h = int(H * 0.08)
        d.rectangle([0, H-bot_h, W, H], fill=(*primary, 200))

    elif template == "Product Launch":
        # Bold colour block left side
        block_w = int(W * 0.42)
        d.rectangle([0, 0, block_w, H], fill=(*primary, 210))
        # Diagonal cut
        pts = [(block_w, 0), (block_w+int(W*0.07), 0),
               (block_w, H), (block_w-int(W*0.04), H)]
        d.polygon(pts, fill=(*primary, 210))
        # Accent stripe
        stripe_x = block_w + int(W*0.07)
        d.rectangle([stripe_x, 0, stripe_x+int(W*0.008), H], fill=(*accent, 255))

    elif template == "Festive":
        # Corner star burst
        def _starburst(cx, cy, r_out, r_in, points, col, alpha):
            pts_list = []
            for i in range(points * 2):
                angle = math.pi * i / points - math.pi/2
                r     = r_out if i % 2 == 0 else r_in
                pts_list.append((cx + r*math.cos(angle), cy + r*math.sin(angle)))
            d.polygon(pts_list, fill=(*col, alpha))

        sb_r = int(min(W,H)*0.12)
        _starburst(sb_r, sb_r,         sb_r, sb_r//2, 8, accent, 200)
        _starburst(W-sb_r, sb_r,       sb_r, sb_r//2, 8, primary, 200)
        _starburst(sb_r, H-sb_r,       sb_r, sb_r//2, 8, primary, 180)
        _starburst(W-sb_r, H-sb_r,     sb_r, sb_r//2, 8, accent, 180)

        # Decorative corner lines
        cl = int(min(W,H) * 0.08)
        lw = max(2, int(min(W,H)*0.004))
        gold = (212,175,55)
        gold_col = (*gold, 200)
        for cx, cy, dx, dy in [(0,0,1,1),(W,0,-1,1),(0,H,1,-1),(W,H,-1,-1)]:
            d.line([(cx, cy), (cx+cl*dx, cy)], fill=gold_col, width=lw)
            d.line([(cx, cy), (cx, cy+cl*dy)], fill=gold_col, width=lw)

    canvas = Image.alpha_composite(canvas, ov)
    return canvas.convert("RGB") if img.mode == "RGB" else canvas


# ─────────────────────────────────────────────────────────────────────────────
#  ANIMATIONS  (real PIL GIF generation)
# ─────────────────────────────────────────────────────────────────────────────
ANIMATIONS = [
    "None", "Fade In", "Slide In Left", "Slide In Bottom",
    "Zoom In", "Zoom Out", "Pulse", "Shimmer", "Ken Burns",
    "Typewriter", "Bounce In",
]

def generate_animation(
    base_img:   Image.Image,
    style:      str,
    n_frames:   int = 12,
    duration:   int = 80,    # ms per frame
    loop:       int = 0,     # 0 = loop forever
) -> bytes:
    """
    Generate an animated GIF from a static creative image.
    Uses real PIL frame-by-frame manipulation — no simulation.
    Returns GIF bytes.
    """
    if style == "None":
        buf = io.BytesIO()
        base_img.convert("RGB").save(buf, "GIF")
        return buf.getvalue()

    W, H   = base_img.size
    base   = base_img.convert("RGBA")
    frames = []

    # ── Fade In ───────────────────────────────────────────────────────────────
    if style == "Fade In":
        for i in range(n_frames):
            t = i / (n_frames - 1)
            alpha = int(255 * t)
            black = Image.new("RGBA", (W, H), (0, 0, 0, 255 - alpha))
            frame = Image.alpha_composite(base.copy(), black)
            frames.append(frame.convert("RGB"))

    # ── Slide In Left ─────────────────────────────────────────────────────────
    elif style == "Slide In Left":
        for i in range(n_frames):
            t    = i / (n_frames - 1)
            ease = t * t * (3 - 2 * t)   # smoothstep
            x_off = int(W * (1 - ease))
            bg    = Image.new("RGB", (W, H), (20, 20, 20))
            frame_rgba = base.copy()
            result = Image.new("RGBA", (W, H), (20, 20, 20, 255))
            result.paste(frame_rgba, (-x_off, 0), frame_rgba)
            frames.append(result.convert("RGB"))

    # ── Slide In Bottom ───────────────────────────────────────────────────────
    elif style == "Slide In Bottom":
        for i in range(n_frames):
            t    = i / (n_frames - 1)
            ease = t * t * (3 - 2 * t)
            y_off = int(H * (1 - ease))
            result = Image.new("RGBA", (W, H), (20, 20, 20, 255))
            result.paste(base, (0, y_off), base)
            frames.append(result.convert("RGB"))

    # ── Zoom In ───────────────────────────────────────────────────────────────
    elif style == "Zoom In":
        for i in range(n_frames):
            t    = i / (n_frames - 1)
            ease = t * t * (3 - 2 * t)
            scale = 0.5 + 0.5 * ease
            nw, nh = int(W * scale), int(H * scale)
            nw = max(1, nw); nh = max(1, nh)
            small  = base.resize((nw, nh), Image.Resampling.LANCZOS)
            result = Image.new("RGBA", (W, H), (20, 20, 20, 255))
            px = (W - nw) // 2
            py = (H - nh) // 2
            result.paste(small, (px, py), small)
            frames.append(result.convert("RGB"))

    # ── Zoom Out ──────────────────────────────────────────────────────────────
    elif style == "Zoom Out":
        for i in range(n_frames):
            t    = i / (n_frames - 1)
            ease = t * t * (3 - 2 * t)
            scale = 1.5 - 0.5 * ease
            nw, nh = int(W * scale), int(H * scale)
            nw = max(1, nw); nh = max(1, nh)
            big    = base.resize((nw, nh), Image.Resampling.LANCZOS)
            result = Image.new("RGBA", (W, H), (20, 20, 20, 255))
            px = (W - nw) // 2
            py = (H - nh) // 2
            result.paste(big, (px, py), big)
            # Crop to canvas
            result = result.crop((0, 0, W, H))
            frames.append(result.convert("RGB"))

    # ── Pulse ─────────────────────────────────────────────────────────────────
    elif style == "Pulse":
        for i in range(n_frames):
            t     = i / n_frames
            scale = 1.0 + 0.06 * math.sin(t * 2 * math.pi)
            nw, nh = int(W * scale), int(H * scale)
            nw = max(1, nw); nh = max(1, nh)
            scaled = base.resize((nw, nh), Image.Resampling.LANCZOS)
            result = Image.new("RGBA", (W, H), (20,20,20,255))
            px = (W - nw) // 2
            py = (H - nh) // 2
            result.paste(scaled, (px, py), scaled)
            result = result.crop((0, 0, W, H))
            frames.append(result.convert("RGB"))

    # ── Shimmer ───────────────────────────────────────────────────────────────
    elif style == "Shimmer":
        for i in range(n_frames):
            t = i / n_frames
            frame = base.copy().convert("RGBA")
            # Moving diagonal highlight strip
            shimmer = Image.new("RGBA", (W, H), (0,0,0,0))
            ds = ImageDraw.Draw(shimmer)
            strip_x = int((W + H) * t) - H
            for j in range(int(W * 0.08)):
                alpha = int(60 * (1 - abs(j - W*0.04) / (W*0.04)))
                ds.line([(strip_x+j, 0), (strip_x+j+H, H)],
                        fill=(255, 255, 255, alpha), width=1)
            result = Image.alpha_composite(frame, shimmer)
            frames.append(result.convert("RGB"))

    # ── Ken Burns (pan + zoom) ────────────────────────────────────────────────
    elif style == "Ken Burns":
        for i in range(n_frames):
            t    = i / (n_frames - 1)
            ease = t * t * (3 - 2 * t)
            # Start: zoomed in top-left, End: zoomed in bottom-right
            scale  = 1.15 - 0.10 * ease
            nw, nh = int(W * scale), int(H * scale)
            nw = max(W, nw); nh = max(H, nh)
            big = base.resize((nw, nh), Image.Resampling.LANCZOS)
            # Pan from top-left to bottom-right
            crop_x = int((nw - W) * ease)
            crop_y = int((nh - H) * ease)
            crop_x = max(0, min(crop_x, nw - W))
            crop_y = max(0, min(crop_y, nh - H))
            frame  = big.crop((crop_x, crop_y, crop_x + W, crop_y + H))
            frames.append(frame.convert("RGB"))

    # ── Typewriter (text reveal simulation via brightness bands) ──────────────
    elif style == "Typewriter":
        # Reveal image from top to bottom line by line
        for i in range(n_frames):
            reveal_y = int(H * i / (n_frames - 1))
            frame    = Image.new("RGB", (W, H), (20, 20, 20))
            if reveal_y > 0:
                revealed = base.crop((0, 0, W, reveal_y)).convert("RGB")
                frame.paste(revealed, (0, 0))
            # Add scan-line effect at reveal boundary
            if reveal_y < H:
                scan = ImageDraw.Draw(frame)
                for sy in range(max(0, reveal_y-3), min(H, reveal_y+3)):
                    alpha_line = int(200 * (1 - abs(sy - reveal_y) / 4))
                    scan.line([(0, sy), (W, sy)], fill=(200, 200, 255), width=1)
            frames.append(frame)

    # ── Bounce In ─────────────────────────────────────────────────────────────
    elif style == "Bounce In":
        def _bounce(t):
            if t < 1/2.75:   return 7.5625 * t * t
            elif t < 2/2.75:
                t -= 1.5/2.75; return 7.5625*t*t + 0.75
            elif t < 2.5/2.75:
                t -= 2.25/2.75; return 7.5625*t*t + 0.9375
            else:
                t -= 2.625/2.75; return 7.5625*t*t + 0.984375
        for i in range(n_frames):
            t    = i / (n_frames - 1)
            ease = _bounce(t)
            scale = 0.3 + 0.7 * ease
            scale = max(0.1, min(1.1, scale))
            nw, nh = int(W * scale), int(H * scale)
            nw = max(1, nw); nh = max(1, nh)
            small  = base.resize((nw, nh), Image.Resampling.LANCZOS)
            result = Image.new("RGBA", (W, H), (20, 20, 20, 255))
            px = (W - nw) // 2
            py = (H - nh) // 2
            result.paste(small, (px, py), small)
            frames.append(result.convert("RGB"))

    if not frames:
        frames = [base.convert("RGB")]

    # Build GIF
    buf = io.BytesIO()
    frames[0].save(
        buf, format="GIF", save_all=True,
        append_images=frames[1:],
        duration=duration, loop=loop,
        optimize=True,
    )
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
#  TEXT POSITIONING  (helpers for positioning text anywhere on canvas)
# ─────────────────────────────────────────────────────────────────────────────
TEXT_POSITIONS = {
    "Top Left":     "tl",
    "Top Center":   "tc",
    "Top Right":    "tr",
    "Middle Left":  "ml",
    "Center":       "cc",
    "Middle Right": "mr",
    "Bottom Left":  "bl",
    "Bottom Center":"bc",
    "Bottom Right": "br",
}

def get_text_xy(pos_code: str, text: str, font, canvas_w: int, canvas_h: int,
                margin: float = 0.05) -> tuple[int, int]:
    """Calculate (x, y) pixel position for text alignment on canvas."""
    tmp = ImageDraw.Draw(Image.new("RGBA", (1,1)))
    try:
        bb = tmp.textbbox((0,0), text, font=font)
        tw, th = bb[2]-bb[0], bb[3]-bb[1]
    except Exception:
        sz = getattr(font, "size", 20)
        tw, th = len(text)*sz//2, sz

    mg = int(min(canvas_w, canvas_h) * margin)
    W, H = canvas_w, canvas_h

    col_map = {"l": mg, "c": (W - tw) // 2, "r": W - tw - mg}
    row_map = {"t": mg, "m": (H - th) // 2, "b": H - th - mg}

    col_key = pos_code[1] if len(pos_code) > 1 else "c"
    row_key = pos_code[0]
    x = col_map.get(col_key, (W-tw)//2)
    y = row_map.get(row_key, (H-th)//2)
    return max(0, x), max(0, y)


# ─────────────────────────────────────────────────────────────────────────────
#  STUDIO UI  — drop into tab1 of dashboard.py
# ─────────────────────────────────────────────────────────────────────────────
def render_studio_tab(base_img: Image.Image, brand_primary: str = "#00539F",
                       brand_accent: str = "#E31837") -> dict:
    """
    Renders the Advanced Studio panel inside Streamlit.
    Call this after the creative is generated.
    Returns dict with keys: image, gif_bytes (or None), settings
    """
    import streamlit as st

    st.markdown("---")
    st.subheader("🎛️ Advanced Studio")
    st.caption("Enhance your creative with text effects, animations, borders and templates — all real PIL processing")

    W, H = base_img.size
    primary_rgb = _hex_to_rgb(brand_primary)
    accent_rgb  = _hex_to_rgb(brand_accent)

    # ── 5 studio sub-tabs ────────────────────────────────────────────────────
    s1, s2, s3, s4, s5 = st.tabs([
        "✍️ Text Effects",
        "🖼️ Templates",
        "🔲 Borders",
        "🎬 Animation",
        "📐 Text Position",
    ])

    result_img = base_img.copy()
    gif_bytes  = None
    settings   = {}

    # ── Text Effects ──────────────────────────────────────────────────────────
    with s1:
        c1, c2 = st.columns(2)
        with c1:
            hl_text   = st.text_input("Headline text", "YOUR HEADLINE", key="se_hl")
            hl_style  = st.selectbox("Font style", TEXT_STYLES, key="se_fs")
            hl_size   = st.slider("Font size", 10, 200, 60, key="se_sz")
            hl_effect = st.selectbox("Text effect", TEXT_EFFECTS, key="se_fx")
            hl_colour = st.color_picker("Text colour", "#FFFFFF", key="se_col")
            fx_colour = st.color_picker("Effect colour", "#000000", key="se_fxc")

        with c2:
            sh_text   = st.text_input("Subhead text", "Your subhead here", key="se_sh")
            sh_style  = st.selectbox("Subhead style", TEXT_STYLES,
                                      index=TEXT_STYLES.index("Regular") if "Regular" in TEXT_STYLES else 0,
                                      key="se_shfs")
            sh_size   = st.slider("Subhead size", 10, 120, 28, key="se_shsz")
            sh_effect = st.selectbox("Subhead effect", TEXT_EFFECTS,
                                      index=TEXT_EFFECTS.index("Drop Shadow") if "Drop Shadow" in TEXT_EFFECTS else 0,
                                      key="se_shfx")
            sh_colour = st.color_picker("Subhead colour", "#EEEEEE", key="se_shcol")
            align     = st.selectbox("Text alignment", ["Left", "Center", "Right"], key="se_align")

        hl_x = st.slider("Headline X position", 0, W, int(W*0.05), key="se_hlx")
        hl_y = st.slider("Headline Y position", 0, H, int(H*0.65), key="se_hly")
        sh_x = st.slider("Subhead X position",  0, W, int(W*0.05), key="se_shx")
        sh_y = st.slider("Subhead Y position",  0, H, int(H*0.75), key="se_shy")

        if st.button("✅ Apply Text Effects", type="primary", key="se_apply"):
            with st.spinner("Applying text effects…"):
                canvas = result_img.copy().convert("RGBA")

                hl_font = _load_style_font(hl_style, hl_size)
                sh_font = _load_style_font(sh_style, sh_size)

                hl_col_rgb = _hex_to_rgb(hl_colour)
                sh_col_rgb = _hex_to_rgb(sh_colour)
                fx_col_rgb = _hex_to_rgb(fx_colour)

                canvas = render_text_effect(
                    canvas, hl_text, hl_x, hl_y,
                    hl_font, hl_col_rgb, hl_effect, fx_col_rgb)
                canvas = render_text_effect(
                    canvas, sh_text, sh_x, sh_y,
                    sh_font, sh_col_rgb, sh_effect, fx_col_rgb)

                result_img = canvas.convert("RGB")
                st.image(result_img, caption="Text effects applied", use_container_width=True)
                settings["text_effects"] = {
                    "headline": hl_text, "hl_style": hl_style,
                    "hl_effect": hl_effect, "sh_text": sh_text,
                }

    # ── Templates ─────────────────────────────────────────────────────────────
    with s2:
        tpl = st.selectbox("Template", list(TEMPLATES.keys()), key="tpl_sel")
        if tpl != "None":
            st.caption(TEMPLATES[tpl]["desc"])

        tc1, tc2 = st.columns(2)
        tpl_primary = tc1.color_picker("Primary colour", brand_primary, key="tpl_p")
        tpl_accent  = tc2.color_picker("Accent colour",  brand_accent,  key="tpl_a")

        if st.button("✅ Apply Template", type="primary", key="tpl_apply"):
            with st.spinner("Applying template…"):
                p_rgb = _hex_to_rgb(tpl_primary)
                a_rgb = _hex_to_rgb(tpl_accent)
                result_img = apply_template_overlay(result_img, tpl, p_rgb, a_rgb)
                st.image(result_img, caption=f"Template: {tpl}", use_container_width=True)
                settings["template"] = tpl

    # ── Borders ───────────────────────────────────────────────────────────────
    with s3:
        border_style = st.selectbox("Border style", BORDERS, key="brd_sel")
        bc1, bc2 = st.columns(2)
        border_col   = bc1.color_picker("Border colour", brand_primary, key="brd_col")
        border_width = bc2.slider("Border width", 2, 40, 10, key="brd_w")

        if st.button("✅ Apply Border", type="primary", key="brd_apply"):
            with st.spinner("Applying border…"):
                b_rgb = _hex_to_rgb(border_col)
                result_img = apply_border(result_img, border_style, b_rgb, border_width)
                st.image(result_img, caption=f"Border: {border_style}", use_container_width=True)
                settings["border"] = border_style

    # ── Animations ────────────────────────────────────────────────────────────
    with s4:
        st.caption("Generates a real animated GIF — choose style, frames and speed")
        ani_style  = st.selectbox("Animation", ANIMATIONS, key="ani_sel")
        ac1, ac2, ac3 = st.columns(3)
        ani_frames = ac1.slider("Frames", 6, 30, 14, key="ani_f")
        ani_dur    = ac2.slider("Speed (ms/frame)", 30, 300, 80, key="ani_d")
        ani_loop   = ac3.selectbox("Loop", [("Forever",0),("Once",1),("3×",3)],
                                    format_func=lambda x: x[0], key="ani_lp")[1]

        if st.button("🎬 Generate Animation", type="primary", key="ani_gen"):
            with st.spinner(f"Rendering {ani_frames} frames of '{ani_style}'…"):
                gif_bytes = generate_animation(
                    result_img, ani_style, ani_frames, ani_dur, ani_loop)
                st.image(gif_bytes, caption=f"Animation: {ani_style}", use_container_width=True)
                st.download_button(
                    "📥 Download GIF",
                    data=gif_bytes,
                    file_name="creative_animated.gif",
                    mime="image/gif",
                    use_container_width=True,
                    key="ani_dl",
                )
                settings["animation"] = ani_style

    # ── Text Position ──────────────────────────────────────────────────────────
    with s5:
        st.caption("Add positioned text with any style — anywhere on your creative")
        tp1, tp2 = st.columns(2)
        pos_text    = tp1.text_input("Text",       "NEW ARRIVAL", key="tp_txt")
        pos_pos     = tp2.selectbox("Position",    list(TEXT_POSITIONS.keys()), key="tp_pos")
        tp3, tp4    = st.columns(2)
        pos_style   = tp3.selectbox("Font style",  TEXT_STYLES, key="tp_sty")
        pos_size    = tp4.slider("Size",           12, 180, 52, key="tp_sz")
        tp5, tp6    = st.columns(2)
        pos_colour  = tp5.color_picker("Colour",   "#FFFFFF", key="tp_col")
        pos_effect  = tp6.selectbox("Effect",      TEXT_EFFECTS, key="tp_fx")
        pos_fx_col  = st.color_picker("Effect colour", "#000000", key="tp_fxc")

        if st.button("✅ Add Text", type="primary", key="tp_apply"):
            with st.spinner("Adding text…"):
                fnt     = _load_style_font(pos_style, pos_size)
                pos_code= TEXT_POSITIONS[pos_pos]
                px, py  = get_text_xy(pos_code, pos_text, fnt, W, H)
                c_rgb   = _hex_to_rgb(pos_colour)
                fx_rgb  = _hex_to_rgb(pos_fx_col)
                result_img = render_text_effect(
                    result_img, pos_text, px, py, fnt, c_rgb, pos_effect, fx_rgb)
                st.image(result_img, caption=f"Text at {pos_pos}", use_container_width=True)

    # ── Final download ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📥 Export")
    ec1, ec2, ec3 = st.columns(3)

    def _to_bytes(img, fmt="PNG"):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, fmt, quality=94)
        return buf.getvalue()

    ec1.download_button("📥 PNG",  data=_to_bytes(result_img,"PNG"),
        file_name="studio_creative.png", mime="image/png",
        use_container_width=True, key="st_png")
    ec2.download_button("📥 JPEG", data=_to_bytes(result_img,"JPEG"),
        file_name="studio_creative.jpg", mime="image/jpeg",
        use_container_width=True, key="st_jpg")
    if gif_bytes:
        ec3.download_button("📥 GIF",  data=gif_bytes,
            file_name="studio_animated.gif", mime="image/gif",
            use_container_width=True, key="st_gif")

    return {"image": result_img, "gif_bytes": gif_bytes, "settings": settings}


# ─────────────────────────────────────────────────────────────────────────────
#  UTILS
# ─────────────────────────────────────────────────────────────────────────────
def _hex_to_rgb(h: str) -> tuple:
    if isinstance(h, (tuple, list)) and len(h) >= 3:
        return tuple(int(x) for x in h[:3])
    h = str(h).strip().lstrip("#")
    if len(h) == 3: h = h[0]*2+h[1]*2+h[2]*2
    try:    return (int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))
    except: return (80, 80, 80)
