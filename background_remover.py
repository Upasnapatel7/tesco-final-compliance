"""
background_remover.py  –  UPGRADED
Uses rembg (U2-Net deep-learning model) for real AI background removal.
Falls back to threshold-based removal if rembg is not installed.

Install:  pip install rembg==2.0.50 onnxruntime==1.16.3
"""

from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import numpy as np
import io

# ── Try to import rembg (real ML background removal) ─────────────────────────
try:
    from rembg import remove as _rembg_remove, new_session
    _REMBG_SESSION = new_session("u2net")   # downloads model on first run (~170 MB)
    REMBG_AVAILABLE = True
except Exception:
    _rembg_remove = None
    _REMBG_SESSION = None
    REMBG_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
def remove_background_ai(image: Image.Image) -> Image.Image:
    """
    Remove the background from a product packshot using the U2-Net ML model.

    • If rembg is available  → real neural-net segmentation (high quality)
    • If rembg is unavailable → falls back to threshold-based removal
    """
    try:
        if REMBG_AVAILABLE:
            return _rembg_remove_background(image)
        else:
            return _threshold_remove_background(image)
    except Exception as exc:
        print(f"[background_remover] Error: {exc}")
        return image.convert("RGBA")


def _rembg_remove_background(image: Image.Image) -> Image.Image:
    """Real ML removal via rembg / U2-Net."""
    # rembg accepts PIL Images or bytes
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    result_bytes = _rembg_remove(buf.read(), session=_REMBG_SESSION)
    result = Image.open(io.BytesIO(result_bytes)).convert("RGBA")
    return result


def _threshold_remove_background(image: Image.Image) -> Image.Image:
    """
    Fallback: remove white / near-white backgrounds using
    per-pixel luminance thresholding + edge-softening.
    Works acceptably for product shots on plain white backgrounds.
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    data = np.array(image, dtype=np.float32)
    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]

    # Luminance of each pixel
    luminance = 0.299 * r + 0.587 * g + 0.114 * b

    # Colour uniformity: how close the pixel is to grey (low saturation)
    max_rgb = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation = np.where(max_rgb > 0, (max_rgb - min_rgb) / max_rgb, 0)

    # Pixels are "background" if they are bright AND low-saturation
    is_background = (luminance > 220) & (saturation < 0.15)

    # Soft alpha (transition zone for anti-aliasing)
    soft_mask = np.clip((luminance - 200) / 50, 0, 1) * (1 - saturation)
    new_alpha = np.where(is_background, (1 - soft_mask) * 255, a)
    data[:, :, 3] = new_alpha.astype(np.uint8)

    return Image.fromarray(data.astype(np.uint8), "RGBA")


# ─────────────────────────────────────────────────────────────────────────────
def enhance_image_quality(image: Image.Image) -> Image.Image:
    """
    Apply a professional quality-enhancement pipeline:
    sharpness → colour → contrast → unsharp mask.
    Preserves RGBA alpha channel if present.
    """
    has_alpha = image.mode == "RGBA"
    if has_alpha:
        r, g, b, alpha = image.split()
        rgb = Image.merge("RGB", (r, g, b))
    else:
        rgb = image.convert("RGB")
        alpha = None

    rgb = ImageEnhance.Sharpness(rgb).enhance(1.35)
    rgb = ImageEnhance.Color(rgb).enhance(1.12)
    rgb = ImageEnhance.Contrast(rgb).enhance(1.08)
    rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))

    if has_alpha and alpha is not None:
        r2, g2, b2 = rgb.split()
        return Image.merge("RGBA", (r2, g2, b2, alpha))
    return rgb


# ─────────────────────────────────────────────────────────────────────────────
def apply_creative_filters(image: Image.Image, filter_type: str) -> Image.Image:
    """
    Apply creative colour-grading filters.
    Supported: Warm | Cool | Vibrant | Matte | Black & White
    """
    mode = image.mode
    if mode == "RGBA":
        r, g, b, alpha = image.split()
        rgb = Image.merge("RGB", (r, g, b))
    else:
        rgb = image.convert("RGB")
        alpha = None

    if filter_type == "Warm":
        rgb = ImageEnhance.Color(rgb).enhance(1.2)
        r, g, b = rgb.split()
        r = r.point(lambda i: min(255, int(i * 1.08)))
        b = b.point(lambda i: max(0, int(i * 0.92)))
        rgb = Image.merge("RGB", (r, g, b))

    elif filter_type == "Cool":
        rgb = ImageEnhance.Color(rgb).enhance(1.1)
        r, g, b = rgb.split()
        b = b.point(lambda i: min(255, int(i * 1.1)))
        r = r.point(lambda i: max(0, int(i * 0.93)))
        rgb = Image.merge("RGB", (r, g, b))

    elif filter_type == "Vibrant":
        rgb = ImageEnhance.Color(rgb).enhance(1.45)
        rgb = ImageEnhance.Contrast(rgb).enhance(1.1)

    elif filter_type == "Matte":
        rgb = ImageEnhance.Contrast(rgb).enhance(0.85)
        rgb = ImageEnhance.Color(rgb).enhance(0.8)
        rgb = ImageEnhance.Brightness(rgb).enhance(1.05)

    elif filter_type == "Black & White":
        rgb = rgb.convert("L").convert("RGB")
        rgb = ImageEnhance.Contrast(rgb).enhance(1.2)

    if alpha is not None:
        r2, g2, b2 = rgb.split()
        return Image.merge("RGBA", (r2, g2, b2, alpha))
    return rgb if mode == "RGB" else rgb.convert(mode)


# ─────────────────────────────────────────────────────────────────────────────
def optimize_for_social_media(image: Image.Image, platform: str) -> Image.Image:
    """Resize and optimise image for a specific social media platform."""
    specs = {
        "instagram_square":  (1080, 1080),
        "instagram_portrait": (1080, 1350),
        "instagram_story":   (1080, 1920),
        "facebook_landscape": (1200, 630),
        "facebook_square":   (1080, 1080),
        "twitter_landscape": (1600, 900),
        "linkedin_banner":   (1584, 396),
    }
    key = platform.lower().replace(" ", "_")
    size = specs.get(key)
    if size:
        return image.resize(size, Image.Resampling.LANCZOS)
    return image


# ─────────────────────────────────────────────────────────────────────────────
def validate_image_dimensions(
    image: Image.Image, min_width: int = 500, min_height: int = 500
) -> tuple[bool, str]:
    """Validate minimum dimensions and return (ok, message)."""
    w, h = image.size
    if w < min_width or h < min_height:
        msgs = []
        if w < min_width:
            msgs.append(f"width {w}px < minimum {min_width}px")
        if h < min_height:
            msgs.append(f"height {h}px < minimum {min_height}px")
        return False, f"Image too small: {', '.join(msgs)}"
    return True, f"Dimensions OK: {w}×{h}px"


# ─────────────────────────────────────────────────────────────────────────────
def create_placeholder_image(width: int, height: int, text: str = "Upload Image") -> Image.Image:
    """Create a branded placeholder image."""
    img = Image.new("RGB", (width, height), color="#BFE0F5")
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, width - 1, height - 1], outline="#00539F", width=3)

    icon_size = min(width, height) // 3
    icon_x = (width - icon_size) // 2
    icon_y = (height - icon_size) // 2 - 20
    draw.rectangle(
        [icon_x, icon_y, icon_x + icon_size, icon_y + icon_size],
        fill="#00539F",
        outline="#003366",
        width=2,
    )

    try:
        from PIL import ImageFont
        font = ImageFont.truetype("Arial", max(14, min(24, width // 15)))
    except Exception:
        from PIL import ImageFont
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((width - tw) // 2, icon_y + icon_size + 20), text, fill="#00539F", font=font)

    return img


# ─────────────────────────────────────────────────────────────────────────────
def get_removal_status() -> dict:
    """Returns the current background-removal engine status for display in the UI."""
    return {
        "engine": "U2-Net (rembg)" if REMBG_AVAILABLE else "Threshold fallback",
        "ai_powered": REMBG_AVAILABLE,
        "quality": "High – neural segmentation" if REMBG_AVAILABLE else "Basic – luminance threshold",
        "install_hint": (
            None if REMBG_AVAILABLE
            else "pip install rembg==2.0.50 onnxruntime==1.16.3"
        ),
    }
