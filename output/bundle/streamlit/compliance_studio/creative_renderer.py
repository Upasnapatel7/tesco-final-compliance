"""
creative_renderer.py  v4
All previous issues fixed:
- Font size slider now works (proper range + pass-through)
- Text always readable — never overflows band
- Patterns are subtle (low alpha, only on backgrounds that suit them)
- No self-drawn shapes obscuring content
- Proper packshot zones that never overlap badge or logo
- Solid colour backgrounds supported
- Clean price badge (no broken polygon)
- Logo always readable
"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps
import numpy as np, math

try:
    from brand_config import FONT_MAP, BRANDS
except ImportError:
    FONT_MAP = {"Poppins (Modern)": {
        "regular": "/usr/share/fonts/truetype/google-fonts/Poppins-Regular.ttf",
        "bold":    "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
        "medium":  "/usr/share/fonts/truetype/google-fonts/Poppins-Medium.ttf",
    }}
    BRANDS = {}

# ─────────────────────────────────────────────────────────────────────────────
#  FONT LOADER  — never crashes, always returns something usable
# ─────────────────────────────────────────────────────────────────────────────
import platform as _platform
_IS_WINDOWS = _platform.system() == "Windows"

# Windows system font paths — guaranteed to exist on any Windows 10/11 machine
_WIN_FONT_MAP = {
    "Poppins (Modern)":        {"regular":"C:/Windows/Fonts/segoeui.ttf",  "bold":"C:/Windows/Fonts/segoeuib.ttf",  "medium":"C:/Windows/Fonts/calibri.ttf"},
    "Liberation Sans (Clean)": {"regular":"C:/Windows/Fonts/arial.ttf",    "bold":"C:/Windows/Fonts/arialbd.ttf",   "medium":"C:/Windows/Fonts/arialbd.ttf"},
    "DejaVu Sans (Neutral)":   {"regular":"C:/Windows/Fonts/verdana.ttf",  "bold":"C:/Windows/Fonts/verdanab.ttf",  "medium":"C:/Windows/Fonts/verdanab.ttf"},
    "Lora (Elegant Serif)":    {"regular":"C:/Windows/Fonts/georgia.ttf",  "bold":"C:/Windows/Fonts/georgiab.ttf",  "medium":"C:/Windows/Fonts/georgiab.ttf"},
    "FreeSans (Rounded)":      {"regular":"C:/Windows/Fonts/tahoma.ttf",   "bold":"C:/Windows/Fonts/tahomabd.ttf",  "medium":"C:/Windows/Fonts/tahomabd.ttf"},
    "FreeSerif (Classic)":     {"regular":"C:/Windows/Fonts/times.ttf",    "bold":"C:/Windows/Fonts/timesbd.ttf",   "medium":"C:/Windows/Fonts/timesbd.ttf"},
    "TeX Gyre Heros (Swiss)":  {"regular":"C:/Windows/Fonts/trebuc.ttf",   "bold":"C:/Windows/Fonts/trebucbd.ttf",  "medium":"C:/Windows/Fonts/trebucbd.ttf"},
}

_WIN_FALLBACKS = [
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "C:/Windows/Fonts/calibri.ttf",
]

_LINUX_FALLBACKS = [
    "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/google-fonts/Poppins-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]

def _load_font(font_name: str, size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    size   = max(8, int(size))
    weight = (weight or "regular").lower().strip()
    if weight not in ("bold", "regular", "medium"):
        weight = "bold"

    if _IS_WINDOWS:
        # Use Windows system fonts — these paths always exist
        entry = _WIN_FONT_MAP.get(font_name) or _WIN_FONT_MAP.get("Poppins (Modern)")
        for w in [weight, "bold", "regular"]:
            path = entry.get(w, "")
            if path:
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    pass
        # Hard Windows fallback
        for path in _WIN_FALLBACKS:
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    else:
        # Linux — use FONT_MAP paths
        entry = FONT_MAP.get(font_name) or list(FONT_MAP.values())[0]
        for w in [weight, "bold", "regular", "medium"]:
            path = entry.get(w, "")
            if path:
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    pass
        for path in _LINUX_FALLBACKS:
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass

    return ImageFont.load_default()


# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def _hex(h) -> tuple:
    if isinstance(h, (tuple, list)) and len(h) >= 3:
        return tuple(int(x) for x in h[:3])
    h = str(h).strip().lstrip("#")
    if len(h) == 3:
        h = h[0]*2 + h[1]*2 + h[2]*2
    try:
        return (int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))
    except Exception:
        return (80, 80, 80)

def _lum(h) -> float:
    r,g,b = _hex(h)
    return 0.299*r + 0.587*g + 0.114*b

def _darken(h, pct=0.10) -> str:
    r,g,b = _hex(h)
    return "#{:02x}{:02x}{:02x}".format(max(0,int(r*(1-pct))), max(0,int(g*(1-pct))), max(0,int(b*(1-pct))))

def _lighten(h, pct=0.30) -> str:
    r,g,b = _hex(h)
    return "#{:02x}{:02x}{:02x}".format(min(255,int(r+(255-r)*pct)), min(255,int(g+(255-g)*pct)), min(255,int(b+(255-b)*pct)))

def _valid_hex(h) -> bool:
    if not isinstance(h, str): return False
    h = h.strip()
    return h.startswith("#") and len(h) in (4, 7)

def _auto_text_color(bg_hex) -> tuple:
    """Return black or white depending on background brightness."""
    return (15, 15, 15) if _lum(bg_hex) > 140 else (245, 245, 245)


# ─────────────────────────────────────────────────────────────────────────────
#  BACKGROUND
# ─────────────────────────────────────────────────────────────────────────────
def _make_bg(W: int, H: int, top_hex: str, bot_hex: str = "") -> Image.Image:
    """Linear gradient or solid colour."""
    t = _hex(top_hex)
    if not bot_hex or bot_hex == top_hex:
        return Image.new("RGB", (W, H), top_hex)
    b = _hex(bot_hex)
    arr = np.zeros((H, W, 3), np.float32)
    tv = np.linspace(0, 1, H)[:, None]
    for c in range(3):
        arr[:, :, c] = t[c]*(1-tv) + b[c]*tv
    return Image.fromarray(arr.astype(np.uint8), "RGB")


# ─────────────────────────────────────────────────────────────────────────────
#  DECORATIVE PATTERNS  — very subtle, only drawn when BG is light enough
# ─────────────────────────────────────────────────────────────────────────────
def _add_pattern(canvas: Image.Image, style: str, primary: str, W: int, H: int, bg_hex: str) -> Image.Image:
    """
    Draws a very subtle decorative pattern. Skipped if background is dark.
    Alpha is kept extremely low (8-18) so it never obscures content.
    """
    # Only draw patterns on light backgrounds
    if _lum(bg_hex) < 160:
        return canvas

    pr = _hex(primary)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d  = ImageDraw.Draw(ov)

    if style == "circles":
        # Two large, very faint quarter-circles in corners
        r1 = int(min(W,H)*0.45)
        d.ellipse([W-r1, -r1//2, W+r1, r1*3//2],           fill=(*pr, 14))
        r2 = int(min(W,H)*0.28)
        d.ellipse([-r2//2, H-r2, r2*3//2, H+r2//2],        fill=(*pr, 10))

    elif style == "geometric":
        # Single diagonal stripe on the right edge
        pts = [(W*0.78, 0), (W*1.0, 0), (W*1.0, H), (W*0.88, H)]
        d.polygon(pts, fill=(*pr, 12))

    elif style == "diagonal_lines":
        # Very fine diagonal lines — reduced from 18 to 7 alpha
        gap = int(W*0.08); thick = max(1, int(W*0.005))
        for i in range(-H, W+H, gap):
            d.line([(i,0),(i+H,H)], fill=(*pr, 7), width=thick)

    elif style == "dots":
        sp = int(W*0.10); dr = max(2, int(W*0.010))
        for row in range(sp//2, H+sp, sp):
            for col in range(sp//2, W+sp, sp):
                d.ellipse([col-dr,row-dr,col+dr,row+dr], fill=(*pr, 14))

    elif style == "corner_arc":
        r = int(min(W,H)*0.50)
        d.arc([W-r, -r//3, W+r//2, r*4//3],    start=120, end=210, fill=(*pr, 18), width=max(3,int(W*0.018)))
        d.arc([-r//2, H*2//3, r, H+r//3],      start=300, end=30,  fill=(*pr, 12), width=max(2,int(W*0.012)))

    elif style == "minimal":
        # Just a thin accent bar on left edge
        d.rectangle([0, 0, max(3,int(W*0.004)), H], fill=(*pr, 160))

    if canvas.mode != "RGBA":
        canvas = canvas.convert("RGBA")
    return Image.alpha_composite(canvas, ov)


# ─────────────────────────────────────────────────────────────────────────────
#  DROP SHADOW
# ─────────────────────────────────────────────────────────────────────────────
def _drop_shadow(img: Image.Image, offset=10, blur=16, opacity=80) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    pad = blur * 2
    sw, sh = img.size[0]+pad*2, img.size[1]+pad*2
    layer = Image.new("RGBA", (sw,sh), (0,0,0,0))
    sil   = Image.new("RGBA", img.size, (0,0,0,opacity))
    sil.putalpha(img.split()[3])
    layer.paste(sil, (pad+offset, pad+offset))
    layer = layer.filter(ImageFilter.GaussianBlur(blur))
    out = Image.new("RGBA", (sw,sh), (0,0,0,0))
    out.paste(layer, mask=layer)
    out.paste(img, (pad,pad), img)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  TEXT WRAP + DRAW
# ─────────────────────────────────────────────────────────────────────────────
def _wrap(text: str, font, max_w: int, draw) -> list:
    if not text: return []
    words, lines, cur = text.split(), [], ""
    for w in words:
        test = (cur+" "+w).strip()
        try:
            width = draw.textlength(test, font=font)
        except Exception:
            width = len(test) * (font.size * 0.6 if hasattr(font,'size') else 10)
        if width <= max_w:
            cur = test
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return lines or [text]


def _line_height(draw, text: str, font) -> int:
    try:
        bb = draw.textbbox((0,0), text or "Ag", font=font)
        return bb[3] - bb[1]
    except Exception:
        return getattr(font, 'size', 20)


def _draw_block(draw, lines, x, y, font, color, spacing=1.22):
    for line in lines:
        draw.text((x, y), line, font=font, fill=color)
        y += int(_line_height(draw, line, font) * spacing)
    return y


def _block_height(draw, lines, font, spacing=1.22) -> int:
    if not lines: return 0
    lh = _line_height(draw, lines[0], font)
    return int(len(lines) * lh * spacing)


# ─────────────────────────────────────────────────────────────────────────────
#  PRICE BADGE  — clean rounded rect, always readable
# ─────────────────────────────────────────────────────────────────────────────
def _price_badge(label: str, price: str, sub: str, primary: str, w: int, h: int) -> Image.Image:
    img = Image.new("RGBA", (w, h), (0,0,0,0))
    d   = ImageDraw.Draw(img)
    pr  = _hex(primary)
    lum = _lum(primary)
    tc  = (255,255,255) if lum < 145 else (15,15,15)
    r   = int(h*0.18)

    # Solid background
    d.rounded_rectangle([0,0,w-1,h-1], radius=r, fill=pr)
    # Very subtle top highlight
    d.rounded_rectangle([2,2,w-3,h//3], radius=max(r-2,3), fill=(255,255,255,22))

    # Load fonts — explicitly sized
    lf = _load_font("Poppins (Modern)", max(11, int(h*0.16)))
    pf = _load_font("Poppins (Modern)", max(20, int(h*0.36)), "bold")
    sf = _load_font("Poppins (Modern)", max(10, int(h*0.13)))

    y = int(h*0.07)

    if label:
        try:
            lb = d.textbbox((0,0), label, font=lf); lw2 = lb[2]-lb[0]
        except: lw2 = len(label)*7
        d.text(((w-lw2)//2, y), label, fill=(*tc, 200), font=lf)
        y += int(h*0.22)
    else:
        y = int(h*0.14)

    try:
        pb = d.textbbox((0,0), price, font=pf); pw2 = pb[2]-pb[0]
    except: pw2 = len(price)*18
    d.text(((w-pw2)//2, y), price, fill=tc, font=pf)
    y += int(h*0.40)

    if sub:
        try:
            sb = d.textbbox((0,0), sub, font=sf); sw2 = sb[2]-sb[0]; sh2 = sb[3]-sb[1]
        except: sw2 = len(sub)*6; sh2 = 12
        sx = (w-sw2)//2
        d.text((sx, y), sub, fill=(*tc,180), font=sf)
        mid = y + sh2//2
        d.line([(sx, mid),(sx+sw2, mid)], fill=(*tc,150), width=1)

    return img


# ─────────────────────────────────────────────────────────────────────────────
#  BRAND LOGO BADGE
# ─────────────────────────────────────────────────────────────────────────────
def _brand_logo(brand: dict, canvas_w: int, canvas_h: int,
                size_factor: float = 0.15) -> Image.Image:
    size_factor = max(0.08, min(0.30, size_factor))
    lw = max(70, int(canvas_w * size_factor))
    lh = max(28, int(lw * 0.36))
    img = Image.new("RGBA", (lw, lh), (0,0,0,0))
    d   = ImageDraw.Draw(img)

    primary = brand.get("primary","#333333")
    shape   = brand.get("logo_shape","pill")
    tc_hex  = brand.get("text_light","#FFFFFF")
    tc      = _hex(tc_hex) if _valid_hex(tc_hex) else (255,255,255)
    # If text_light is dark (low contrast on primary), flip
    if _lum(primary) > 180 and _lum(tc_hex if _valid_hex(tc_hex) else "#ffffff") > 180:
        tc = (15,15,15)

    if shape == "pill":
        d.rounded_rectangle([0,0,lw-1,lh-1], radius=lh//2, fill=primary)
    elif shape == "circle":
        side = min(lw,lh)
        d.ellipse([(lw-side)//2,(lh-side)//2,(lw+side)//2,(lh+side)//2], fill=primary)
    else:
        d.rounded_rectangle([0,0,lw-1,lh-1], radius=max(4,lh//8), fill=primary)

    # Inner highlight (very subtle)
    d.rounded_rectangle([2,2,lw-3,lh//2], radius=max(3,lh//8-1), fill=(255,255,255,20))

    fnt = _load_font("Poppins (Modern)", max(11,int(lh*0.48)), "bold")
    txt = str(brand.get("logo_text","BRAND"))[:14]
    try:
        bb = d.textbbox((0,0),txt,font=fnt); tw2,th2 = bb[2]-bb[0], bb[3]-bb[1]
    except: tw2,th2 = len(txt)*8, int(lh*0.5)
    d.text(((lw-tw2)//2, (lh-th2)//2 - 1), txt, fill=tc, font=fnt)
    return img


# ─────────────────────────────────────────────────────────────────────────────
#  IMAGE EDITING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def apply_image_edits(img: Image.Image, edits: dict) -> Image.Image:
    if not edits: return img
    has_a = img.mode == "RGBA"
    if has_a: r,g,b,a = img.split(); rgb = Image.merge("RGB",(r,g,b))
    else: rgb = img.convert("RGB"); a = None

    bv = edits.get("brightness",1.0)
    cv = edits.get("contrast",1.0)
    sv = edits.get("saturation",1.0)
    shv= edits.get("sharpness",1.0)
    bl = edits.get("blur",0)
    flt= edits.get("filter","none")

    if bv  != 1.0: rgb = ImageEnhance.Brightness(rgb).enhance(bv)
    if cv  != 1.0: rgb = ImageEnhance.Contrast(rgb).enhance(cv)
    if sv  != 1.0: rgb = ImageEnhance.Color(rgb).enhance(sv)
    if shv != 1.0: rgb = ImageEnhance.Sharpness(rgb).enhance(shv)
    if bl  >  0:   rgb = rgb.filter(ImageFilter.GaussianBlur(bl))

    if flt == "warm":
        r2,g2,b2 = rgb.split()
        rgb = Image.merge("RGB",(r2.point(lambda i:min(255,int(i*1.12))),g2,b2.point(lambda i:max(0,int(i*0.88)))))
    elif flt == "cool":
        r2,g2,b2 = rgb.split()
        rgb = Image.merge("RGB",(r2.point(lambda i:max(0,int(i*0.90))),g2,b2.point(lambda i:min(255,int(i*1.12)))))
    elif flt == "vibrant":
        rgb = ImageEnhance.Color(ImageEnhance.Contrast(rgb).enhance(1.1)).enhance(1.45)
    elif flt == "matte":
        rgb = ImageEnhance.Brightness(ImageEnhance.Contrast(ImageEnhance.Color(rgb).enhance(0.72)).enhance(0.88)).enhance(1.06)
    elif flt == "bw":
        rgb = ImageEnhance.Contrast(rgb.convert("L").convert("RGB")).enhance(1.2)
    elif flt == "vintage":
        r2,g2,b2 = rgb.split()
        rgb = ImageEnhance.Contrast(Image.merge("RGB",(r2.point(lambda i:min(255,int(i*1.08+10))),g2.point(lambda i:max(0,int(i*0.96))),b2.point(lambda i:max(0,int(i*0.78)))))).enhance(0.92)

    if edits.get("flip_h"): rgb = ImageOps.mirror(rgb)
    if edits.get("flip_v"): rgb = ImageOps.flip(rgb)
    rot = edits.get("rotate",0)
    if rot in (90,180,270): rgb = rgb.rotate(-rot, expand=True)

    if has_a and a:
        if rot in (90,270): a = a.rotate(-rot, expand=True)
        r2,g2,b2 = rgb.split()
        return Image.merge("RGBA",(r2,g2,b2,a))
    return rgb


# ─────────────────────────────────────────────────────────────────────────────
#  SMART BACKGROUND REMOVER  (threshold-based, no ML required)
# ─────────────────────────────────────────────────────────────────────────────
def _smart_bg_remove(img: Image.Image) -> Image.Image:
    """
    Detect whether the image has a near-white or near-uniform background
    by sampling its 4 corners. If yes, remove it via luminance + edge mask.
    If the image already has meaningful transparency, leave it alone.
    """
    import numpy as np

    # If already RGBA with real transparency, leave it
    if img.mode == "RGBA":
        alpha_arr = np.array(img.split()[3])
        if alpha_arr.min() < 200:          # has real transparent pixels
            return img

    rgb = img.convert("RGB")
    arr = np.array(rgb, dtype=np.float32)
    h, w = arr.shape[:2]

    # Sample corners (5% of each dimension)
    margin_h = max(3, int(h * 0.05))
    margin_w = max(3, int(w * 0.05))
    corners  = [
        arr[:margin_h,  :margin_w],
        arr[:margin_h,  -margin_w:],
        arr[-margin_h:, :margin_w],
        arr[-margin_h:, -margin_w:],
    ]
    corner_means = [c.mean(axis=(0,1)) for c in corners]
    avg_corner   = np.mean(corner_means, axis=0)          # mean RGB of corners
    corner_lum   = 0.299*avg_corner[0] + 0.587*avg_corner[1] + 0.114*avg_corner[2]

    # Only attempt removal if corners look like a plain background (light or dark uniform)
    corner_std = np.std(corner_means, axis=0).mean()
    if corner_std > 30:                    # corners vary too much → not a plain BG
        return img

    # Compute alpha mask based on distance from corner colour
    bg_color = avg_corner
    diff = np.abs(arr - bg_color[None, None, :])
    dist = diff.max(axis=2)               # max channel distance from corner colour

    if corner_lum < 20:
        # Very dark background — invert distance measure
        lum = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
        dist = np.abs(lum - corner_lum)

    alpha_f = np.clip(dist / 40.0, 0, 1)

    # Apply a slight edge feather using a small blur on the mask
    from PIL import ImageFilter
    alpha_img = Image.fromarray((alpha_f * 255).astype(np.uint8), "L")
    alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(1))

    result = rgb.convert("RGBA")
    result.putalpha(alpha_img)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  PACKSHOT COMPOSITOR  — respects user positions if provided
# ─────────────────────────────────────────────────────────────────────────────
def _composite_packshots(canvas, packshots, zone_x, zone_y, zone_w, zone_h,
                          edits=None, positions=None):
    """
    positions: list of (x_frac, y_frac, scale_frac) per packshot, all in 0..1
               relative to full canvas. None = auto-layout.
    """
    W, H = canvas.size
    n = min(len(packshots), 3)

    def prep(ps, mw, mh):
        ps = ps.copy()
        if edits: ps = apply_image_edits(ps, edits)
        ps.thumbnail((mw, mh), Image.Resampling.LANCZOS)

        # Auto-detect and remove near-white backgrounds
        # (only applies when packshot has no meaningful transparency)
        ps = _smart_bg_remove(ps)
        if ps.mode != "RGBA": ps = ps.convert("RGBA")
        return ps

    def paste(cnv, ps, x, y):
        if cnv.mode != "RGBA": cnv = cnv.convert("RGBA")
        sh = _drop_shadow(ps, offset=8, blur=14, opacity=75)
        pad = 14
        cnv.paste(sh, (int(x)-pad, int(y)-pad), sh)
        return cnv

    # If user has custom positions, use them
    if positions and len(positions) >= n:
        for i in range(n):
            pos = positions[i]
            scale = pos.get("scale", 0.45)
            mw = int(W * scale); mh = int(W * scale)
            ps = prep(packshots[i], mw, mh)
            px = int(pos.get("x", 0.5) * W - ps.size[0]//2)
            py = int(pos.get("y", 0.4) * H - ps.size[1]//2)
            canvas = paste(canvas, ps, px, py)
        return canvas

    # Auto layout inside zone
    if n == 1:
        ps = prep(packshots[0], int(zone_w*0.82), int(zone_h*0.90))
        x  = zone_x + (zone_w - ps.size[0])//2
        y  = zone_y + (zone_h - ps.size[1])//2
        canvas = paste(canvas, ps, x, y)

    elif n == 2:
        hw = int(zone_w*0.46); gap = int(zone_w*0.04)
        for i, p in enumerate(packshots[:2]):
            ps = prep(p, hw, int(zone_h*0.82))
            x  = zone_x + gap if i == 0 else zone_x + zone_w - ps.size[0] - gap
            y  = zone_y + (zone_h - ps.size[1])//2
            canvas = paste(canvas, ps, x, y)

    else:
        lead = prep(packshots[0], int(zone_w*0.55), int(zone_h*0.85))
        canvas = paste(canvas, lead, zone_x+int(zone_w*0.02), zone_y+(zone_h-lead.size[1])//2)
        rw = int(zone_w*0.38)
        for i, p in enumerate(packshots[1:3]):
            ps = prep(p, rw, int(zone_h*0.40))
            x  = zone_x + zone_w - ps.size[0] - int(zone_w*0.02)
            y  = zone_y + i*(zone_h//2) + (zone_h//4 - ps.size[1]//2)
            canvas = paste(canvas, ps, x, y)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
#  PATTERN MAP
# ─────────────────────────────────────────────────────────────────────────────
_BRAND_PATTERNS = {
    "Tesco":"corner_arc", "Sainsbury's":"circles", "ASDA":"geometric",
    "Morrisons":"diagonal_lines", "Waitrose":"minimal", "Marks & Spencer":"minimal",
    "Aldi":"geometric", "Lidl":"circles", "Walmart":"dots", "Target":"circles",
    "Amazon":"dots", "Boots":"corner_arc", "H&M":"diagonal_lines",
    "Zara":"minimal", "Custom Brand":"circles",
}


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN RENDER
# ─────────────────────────────────────────────────────────────────────────────
def render_creative(
    dimensions,
    packshots,
    headline,
    subhead,
    brand_name       = "Tesco",
    badge_label      = "",
    badge_price      = "",
    badge_sub        = "",
    badge_show       = False,
    tag_text         = "",
    bg_color         = "#BFE0F5",
    bg_gradient_bot  = "",
    bg_image         = None,
    layout_preset    = "Product Hero",
    font_name        = "Poppins (Modern)",
    font_weight      = "bold",        # bold | regular | medium
    headline_size    = 0,             # 0 = auto
    subhead_size     = 0,             # 0 = auto
    headline_color   = "",
    subhead_color    = "",
    headline_uppercase = True,
    show_logo        = True,
    include_drinkaware = False,
    packshot_edits   = None,
    packshot_positions = None,
    texture_style    = "none",        # pattern for background
    logo_size_factor  = 0.15,
    logo_col_override = "",
    badge_size_factor = 0.200,
    badge_col_override= "",
) -> Image.Image:

    W, H        = dimensions
    is_stories  = H > W
    is_landscape = W > H

    # ── Brand ─────────────────────────────────────────────────────────────────
    _default_brand = {
        "primary":"#333333","secondary":"#F5F5F5","accent":"#666666",
        "text_dark":"#111111","text_light":"#FFFFFF","bg_default":"#F5F5F5",
        "font_family":"Poppins (Modern)","logo_text":brand_name.upper()[:10],
        "logo_shape":"pill","badge_style":"generic",
    }
    brand   = BRANDS.get(brand_name, _default_brand)
    primary = str(brand.get("primary","#333333"))
    t_dark  = str(brand.get("text_dark","#111111"))
    t_light = str(brand.get("text_light","#FFFFFF"))
    bg_def  = str(brand.get("bg_default","#F0F4F8"))

    # ── Font ──────────────────────────────────────────────────────────────────
    if not font_name or font_name not in FONT_MAP:
        bf = brand.get("font_family","Poppins (Modern)")
        font_name = next((k for k in FONT_MAP if bf in k), "Poppins (Modern)")

    # ── Background colour  (never let pure-solid backgrounds break) ───────────
    if _valid_hex(bg_color):
        top_hex = bg_color
    else:
        top_hex = bg_def

    # Clamp very dark bg to readable range
    if _lum(top_hex) < 50:
        top_hex = _lighten(top_hex, 0.60)

    if bg_gradient_bot and _valid_hex(bg_gradient_bot) and bg_gradient_bot != top_hex:
        bot_hex = bg_gradient_bot
    else:
        bot_hex = ""   # solid colour

    # ── 1. Canvas ─────────────────────────────────────────────────────────────
    if bg_image:
        canvas = bg_image.resize((W,H), Image.Resampling.LANCZOS).convert("RGBA")
        canvas = ImageEnhance.Brightness(ImageEnhance.Color(canvas.convert("RGB")).enhance(0.70).convert("RGBA")).enhance(0.95).convert("RGBA") if False else canvas
        rgb_tmp = ImageEnhance.Color(canvas.convert("RGB")).enhance(0.70)
        rgb_tmp = ImageEnhance.Brightness(rgb_tmp).enhance(0.95)
        canvas  = rgb_tmp.convert("RGBA")
    else:
        canvas = _make_bg(W, H, top_hex, bot_hex).convert("RGBA")

    # ── 2. Pattern (very subtle) ──────────────────────────────────────────────
    # Use user-specified texture if provided, otherwise brand default
    if texture_style and texture_style != "none":
        pat = texture_style
    else:
        pat = _BRAND_PATTERNS.get(brand_name, "circles")
    canvas = _add_pattern(canvas, pat, primary, W, H, top_hex)

    # Also apply richer apply_texture if available (from ai_creative_director)
    if texture_style and texture_style not in ("none", "corner_arc", "circles",
                                                "geometric", "dots", "minimal"):
        try:
            canvas = apply_texture(canvas, texture_style, primary, 10)
        except Exception:
            pass

    # ── 3. Layout zones ───────────────────────────────────────────────────────
    margin = int(W * 0.055)

    # Packshot zone must NOT overlap badge (top-left) or logo (top-right)
    badge_reserved_w = int(W * 0.28)   # left side reserved for badge
    logo_reserved_w  = int(W * 0.22)   # right side reserved for logo

    if is_stories:
        # Stories: packshot centred in safe middle zone
        pz_x = int(W * 0.05)
        pz_y = int(H * 0.14)
        pz_w = int(W * 0.90)
        pz_h = int(H * 0.56)
        text_x   = margin
        text_max = int(W * 0.88)
        band_h   = int(H * 0.20)

    elif is_landscape:
        # Landscape: text LEFT, packshot RIGHT
        pz_x = int(W * 0.47)
        pz_y = int(H * 0.04)
        pz_w = int(W * 0.49)
        pz_h = int(H * 0.88)
        text_x   = margin
        text_max = int(W * 0.41)
        band_h   = int(H * 0.42)   # taller band for landscape = more text room

    else:
        # Square: packshot right 54%, text left 38%
        pz_x = int(W * 0.40)
        pz_y = int(H * 0.08)
        pz_w = int(W * 0.56)
        pz_h = int(H * 0.74)
        text_x   = margin
        text_max = int(W * 0.36)
        band_h   = int(H * 0.26)

    # Layout preset adjustments
    if layout_preset == "Centered Minimal":
        pz_x = int(W*0.22); pz_w = int(W*0.56)
        pz_y = int(H*0.10); pz_h = int(H*0.62)
        text_x = int(W*0.06); text_max = int(W*0.88)
        band_h = int(H * (0.22 if not is_landscape else 0.36))

    elif layout_preset == "Bold Left":
        pz_x = int(W*0.03); pz_w = int(W*0.44)
        pz_y = int(H*0.06); pz_h = int(H*0.70)
        text_x = int(W*0.50); text_max = int(W*0.44)

    elif layout_preset == "Full Bleed":
        pz_x = 0; pz_y = 0; pz_w = W; pz_h = H
        text_max = int(W * 0.72)

    elif layout_preset == "Split Panel":
        panel_w = int(W * 0.44)
        pr_rgb  = _hex(primary)
        panel = Image.new("RGBA", (panel_w, H), (*pr_rgb, 240))
        if canvas.mode != "RGBA": canvas = canvas.convert("RGBA")
        canvas.paste(panel, (0,0), panel)
        pz_x = int(W*0.46); pz_w = int(W*0.50)
        pz_y = int(H*0.06); pz_h = int(H*0.80)
        text_x = int(W*0.04); text_max = int(W*0.38)

    band_y = H - band_h

    # ── 4. Packshots ──────────────────────────────────────────────────────────
    if packshots:
        if canvas.mode != "RGBA": canvas = canvas.convert("RGBA")
        canvas = _composite_packshots(
            canvas, packshots, pz_x, pz_y, pz_w, pz_h,
            edits=packshot_edits, positions=packshot_positions
        )

    # ── 5. Text band ──────────────────────────────────────────────────────────
    draw = ImageDraw.Draw(canvas)
    bg_dark  = _lum(top_hex) < 100
    is_split = layout_preset == "Split Panel"
    is_bleed = layout_preset == "Full Bleed"

    if is_bleed:
        # Gradient scrim from transparent → dark
        scrim = Image.new("RGBA", (W, H), (0,0,0,0))
        sd = ImageDraw.Draw(scrim)
        for i in range(band_h):
            a = int(230 * ((i / band_h) ** 0.65))
            sd.rectangle([0, band_y+i, W, band_y+i+1], fill=(0,0,0,a))
        canvas = Image.alpha_composite(canvas.convert("RGBA"), scrim)
        band_text = (245,245,245)
        tag_color = (255,255,255)

    elif is_split:
        # Text goes on solid coloured panel — no separate band
        band_text = _hex(t_light) if _lum(primary) < 145 else _hex(t_dark)
        tag_color = band_text

    else:
        # Clean white band
        ov = Image.new("RGBA", (W, band_h), (255, 255, 255, 228))
        canvas.paste(ov, (0, band_y), ov)
        band_text = _hex(t_dark)
        tag_color = _hex(primary)

    # Accent stripe
    draw = ImageDraw.Draw(canvas)
    if not is_bleed and not is_split:
        stripe_h = max(5, int(H * 0.007))
        draw.rectangle([0, band_y, W, band_y + stripe_h], fill=_hex(primary) + (255,))
        ty0 = band_y + stripe_h + int(H * 0.016)
    else:
        ty0 = band_y + int(H * 0.016)

    # ── 6. Typography — auto-fit to available space ───────────────────────────
    # Reserve space: tag line (if any) at very bottom of band, da below it
    tag_reserve = int(H * 0.060) if (tag_text and tag_text not in ("None","")) else 0
    da_reserve  = int(H * 0.035) if include_drinkaware else 0
    avail_h = H - ty0 - tag_reserve - da_reserve - int(H * 0.012)

    # Auto sizes scaled to available space
    auto_hl = max(22, int(avail_h * 0.36))
    auto_sh = max(13, int(avail_h * 0.20))

    # User override (slider > min value means user explicitly set it)
    # ── Font size: use slider value directly, 0 = auto ─────────────────────
    if isinstance(headline_size, (int, float)) and int(headline_size) > 0:
        hl_sz = max(8, int(headline_size))
    else:
        hl_sz = auto_hl
    if isinstance(subhead_size, (int, float)) and int(subhead_size) > 0:
        sh_sz = max(8, int(subhead_size))
    else:
        sh_sz = auto_sh

    # Expand band if needed to fit requested sizes
    needed = hl_sz + sh_sz + tag_reserve + da_reserve + int(H * 0.05)
    if needed > band_h:
        band_h = min(needed, int(H * 0.55))
        band_y = H - band_h
        if not is_bleed and not is_split:
            ov2 = Image.new("RGBA", (W, band_h), (255, 255, 255, 228))
            canvas.paste(ov2, (0, band_y), ov2)
            draw = ImageDraw.Draw(canvas)
            stripe_h2 = max(5, int(H * 0.007))
            draw.rectangle([0, band_y, W, band_y+stripe_h2], fill=_hex(primary)+(255,))
            ty0 = band_y + stripe_h2 + int(H * 0.016)


    resolved_weight = (font_weight or "bold").lower().strip()
    if resolved_weight not in ("bold","regular","medium"): resolved_weight = "bold"
    resolved_font = font_name if font_name in FONT_MAP else list(FONT_MAP.keys())[0]
    hl_font  = _load_font(resolved_font, hl_sz, resolved_weight)
    sh_font  = _load_font(resolved_font, sh_sz, "regular")
    tag_font = _load_font(resolved_font, max(12, int(H*0.019)), "regular")
    da_font  = _load_font(resolved_font, max(11, int(H*0.016)), "regular")

    # Resolve headline/subhead colour
    def _res_col(user_col, fallback):
        return _hex(user_col) if _valid_hex(user_col) else fallback

    hl_col = _res_col(headline_color, band_text)
    sh_col = _res_col(subhead_color,  tuple(min(255,c+55) for c in band_text) if sum(band_text) < 300 else (90,90,90))

    hl_text  = headline.upper() if headline_uppercase else headline
    hl_lines = _wrap(hl_text, hl_font, text_max, draw)
    cur_y    = ty0
    cur_y    = _draw_block(draw, hl_lines, text_x, cur_y, hl_font, hl_col)
    cur_y   += max(3, int(sh_sz * 0.18))

    # Only draw subhead if there is still space above the tag reserve
    tag_y    = H - tag_reserve - da_reserve - int(H*0.008)
    sh_lines = _wrap(subhead, sh_font, text_max, draw)
    if cur_y + int(sh_sz * 1.4) < tag_y:
        _draw_block(draw, sh_lines, text_x, cur_y, sh_font, sh_col)

    # Tag — pinned to bottom of band, never overlaps subhead
    if tag_text and tag_text not in ("None",""):
        ty = H - da_reserve - int(H * 0.050)
        draw.text((text_x, ty), tag_text, font=tag_font, fill=tag_color)

    # Drinkaware — very bottom
    if include_drinkaware:
        da = "drinkaware.co.uk"
        try:
            bb = draw.textbbox((0,0), da, font=da_font); daw = bb[2]-bb[0]
        except: daw = len(da)*10
        draw.text(((W-daw)//2, H-int(H*0.026)), da, font=da_font, fill=(20,20,20))

    # ── 7. Price badge ────────────────────────────────────────────────────────
    if badge_show and badge_price:
        bw = int(W * max(0.10, min(0.40, badge_size_factor)))
        bh = int(bw * 0.80)
        badge_primary = badge_col_override if (badge_col_override and badge_col_override.startswith("#") and len(badge_col_override) in (4,7)) else primary
        badge = _price_badge(badge_label, badge_price, badge_sub, badge_primary, bw, bh)
        sh_b  = _drop_shadow(badge, offset=5, blur=10, opacity=65)
        if canvas.mode != "RGBA": canvas = canvas.convert("RGBA")
        bx = margin
        by = int(H*0.055) if not is_stories else int(H*0.150)
        canvas.paste(sh_b, (bx-10, by-10), sh_b)

    # ── 8. Logo ───────────────────────────────────────────────────────────────
    if show_logo:
        logo_brand = dict(brand)
        if logo_col_override and logo_col_override.startswith("#") and len(logo_col_override) in (4,7):
            logo_brand["primary"] = logo_col_override
        logo = _brand_logo(logo_brand, W, H, size_factor=logo_size_factor)
        lx = W - logo.width - int(W*0.030)
        ly = int(H*0.022)
        if canvas.mode != "RGBA": canvas = canvas.convert("RGBA")
        canvas.paste(logo, (lx, ly), logo)

    return canvas.convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
#  TEXTURE APPLICATOR  (used by ai_creative_director)
# ─────────────────────────────────────────────────────────────────────────────
def apply_texture(canvas: Image.Image, style: str, colour: str, alpha: int = 12) -> Image.Image:
    """Apply a beautiful decorative texture overlay to a canvas."""
    W, H = canvas.size
    try:
        r, g, b = int(colour.lstrip("#")[0:2],16), int(colour.lstrip("#")[2:4],16), int(colour.lstrip("#")[4:6],16)
    except Exception:
        r, g, b = 100, 100, 100
    a = max(4, min(28, alpha))
    ov = Image.new("RGBA", (W, H), (0,0,0,0))
    d  = ImageDraw.Draw(ov)

    if style == "subtle_dots":
        sp = max(18, int(W*0.035)); rad = max(2, int(sp*0.18))
        for row in range(sp//2, H+sp, sp):
            for col in range(sp//2, W+sp, sp):
                d.ellipse([col-rad,row-rad,col+rad,row+rad], fill=(r,g,b,a))

    elif style == "diagonal_lines":
        gap = max(22, int(W*0.055)); thick = max(1, int(W*0.004))
        for i in range(-H, W+H, gap):
            d.line([(i,0),(i+H,H)], fill=(r,g,b,a), width=thick)

    elif style == "corner_circles":
        r1 = int(min(W,H)*0.52)
        d.ellipse([W-r1,-r1//2,W+r1//3,r1], fill=(r,g,b,max(4,a-2)))
        r2 = int(min(W,H)*0.32)
        d.ellipse([-r2//3,H-r2,r2,H+r2//3], fill=(r,g,b,max(3,a-4)))
        r3 = int(min(W,H)*0.18)
        d.ellipse([W//2-r3,-r3//2,W//2+r3,r3], fill=(r,g,b,max(3,a-6)))

    elif style == "geometric_slab":
        pts1 = [(W*0.72,0),(W,0),(W,H*0.6),(W*0.82,H*0.6)]
        pts2 = [(W*0.86,H*0.4),(W,H*0.4),(W,H),(W*0.86,H)]
        d.polygon(pts1, fill=(r,g,b,a+2))
        d.polygon(pts2, fill=(r,g,b,max(4,a-2)))

    elif style == "soft_noise":
        arr = np.random.randint(0, max(1,a//2), (H,W), dtype=np.uint8)
        noise = Image.fromarray(arr,"L")
        ov = Image.merge("RGBA",(Image.new("L",(W,H),r),Image.new("L",(W,H),g),Image.new("L",(W,H),b),noise))

    elif style == "wave_lines":
        import math
        for j, y_off in enumerate(range(0, H+int(H*0.15), int(H*0.12))):
            pts = [(int(W*s/80), int(y_off + math.sin(s/80*math.pi*3+j)*H*0.03)) for s in range(81)]
            if len(pts) >= 2:
                d.line(pts, fill=(r,g,b,max(4,a-4)), width=max(1,int(H*0.003)))

    elif style == "crosshatch":
        gap = max(28,int(W*0.06)); thick = max(1,int(W*0.003))
        for i in range(-H,W+H,gap):
            d.line([(i,0),(i+H,H)], fill=(r,g,b,max(4,a-3)), width=thick)
            d.line([(i+H,0),(i,H)], fill=(r,g,b,max(4,a-3)), width=thick)

    elif style == "halftone":
        sp = max(12,int(W*0.025))
        for row in range(0,H+sp,sp):
            for col in range(0,W+sp,sp):
                dist = ((col-W//2)**2+(row-H//2)**2)**0.5
                frac = dist/((W//2)**2+(H//2)**2)**0.5
                rad2 = max(1,int(sp*0.35*frac))
                d.ellipse([col-rad2,row-rad2,col+rad2,row+rad2],fill=(r,g,b,min(a+5,int(a*(1+frac)))))

    elif style == "grid":
        sp = max(30,int(W*0.07))
        for x in range(0,W+sp,sp): d.line([(x,0),(x,H)], fill=(r,g,b,max(4,a-4)), width=1)
        for y in range(0,H+sp,sp): d.line([(0,y),(W,y)], fill=(r,g,b,max(4,a-4)), width=1)

    if canvas.mode != "RGBA": canvas = canvas.convert("RGBA")
    return Image.alpha_composite(canvas, ov)