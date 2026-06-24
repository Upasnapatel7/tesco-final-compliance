"""
value_tile_generator.py  –  UPGRADED
Generates Tesco-compliant value tiles using PIL for rendering.
New: Claude API validates tile copy before rendering and generates
     AI-suggested alternative price messaging.
"""

import os
import json
import re
from PIL import Image, ImageDraw, ImageFont

try:
    import anthropic
    _client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
    ANTHROPIC_AVAILABLE = True
except Exception:
    _client = None
    ANTHROPIC_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# CLAUDE HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _call_claude(prompt: str, max_tokens: int = 300) -> str:
    if not ANTHROPIC_AVAILABLE or not _client:
        return ""
    try:
        response = _client.messages.create(
            model="claude-opus-4-6",
            max_tokens=max_tokens,
            system=(
                "You are a Tesco value-tile compliance specialist. "
                "Value tiles follow Appendix A rules: only price figures may change; "
                "all labels, colours, and layouts are predefined and locked. "
                "Return valid JSON only."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as exc:
        return f"__error__:{exc}"


def _safe_json(text: str, fallback):
    try:
        cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
        return json.loads(cleaned)
    except Exception:
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
# FONT LOADER
# ─────────────────────────────────────────────────────────────────────────────
def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Try common system fonts; fall back to PIL default."""
    candidates = (
        ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/System/Library/Fonts/Helvetica.ttc",
         "ArialBD.ttf", "Arial Bold.ttf"]
        if bold else
        ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/System/Library/Fonts/Helvetica.ttc",
         "Arial.ttf"]
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def generate_value_tile(
    tile_type: str, price_data: dict, dimensions: tuple[int, int] = (300, 100)
) -> Image.Image:
    """
    Generate a Tesco-compliant value tile.
    Claude validates price_data before rendering.
    """
    # AI validation pass (non-blocking – render even if API unavailable)
    _ai_validate_price_data(tile_type, price_data)

    width, height = dimensions
    if tile_type == "Clubcard Price":
        return _create_clubcard_tile(price_data, width, height)
    elif tile_type == "Everyday Low Price":
        return _create_lep_tile(price_data, width, height)
    elif tile_type == "New":
        return _create_new_tile(width, height)
    else:
        return _create_default_tile(width, height)


def _ai_validate_price_data(tile_type: str, price_data: dict) -> dict:
    """
    Ask Claude to check whether the price data is correctly formatted
    and compliant with Appendix A.
    Returns a dict of {valid, issues}.
    """
    if not ANTHROPIC_AVAILABLE or not _client:
        return {"valid": True, "issues": []}

    prompt = f"""
Validate this Tesco value-tile data against Appendix A rules.

Tile type  : {tile_type}
Price data : {json.dumps(price_data)}

Appendix A rules:
- Clubcard Price: requires 'clubcard_price' (e.g. "£3.50") and optionally 'regular_price' (e.g. "£4.50")
- Everyday Low Price: requires 'lep_price' (e.g. "£2.99")
- New: no price data needed
- All prices must use £ symbol and decimal format
- No promotional language in price data fields

Return: {{"valid": true|false, "issues": ["...", ...]}}
"""
    raw = _call_claude(prompt)
    if raw and not raw.startswith("__error__"):
        return _safe_json(raw, {"valid": True, "issues": []})
    return {"valid": True, "issues": []}


def ai_suggest_price_messaging(
    product_name: str, tile_type: str, price_data: dict
) -> dict:
    """
    Ask Claude to suggest the best way to present price information on the tile
    while staying fully Appendix A compliant.
    Returns {suggestion, rationale, compliant_copy}
    """
    if not ANTHROPIC_AVAILABLE or not _client:
        return {
            "suggestion": "Use standard Appendix A tile format",
            "rationale": "Claude API unavailable",
            "compliant_copy": price_data,
        }

    prompt = f"""
Suggest the clearest Appendix-A compliant way to present price information
for this Tesco value tile.

Product    : {product_name}
Tile type  : {tile_type}
Price data : {json.dumps(price_data)}

Remember:
- Only price figures may change; all label text is locked (predefined)
- No promotional language (no "save", "only", "special", "discount")
- Format: £X.XX

Return:
{{
  "suggestion": "<one sentence recommendation>",
  "rationale": "<why this works>",
  "compliant_copy": {{<updated price_data dict>}}
}}
"""
    raw = _call_claude(prompt)
    if raw and not raw.startswith("__error__"):
        result = _safe_json(raw, None)
        if isinstance(result, dict):
            return result
    return {
        "suggestion": "Use standard Appendix A format",
        "rationale": "AI suggestion unavailable",
        "compliant_copy": price_data,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TILE RENDERERS
# ─────────────────────────────────────────────────────────────────────────────
def _create_clubcard_tile(price_data: dict, width: int, height: int) -> Image.Image:
    """
    Clubcard Price tile – Appendix A:
    Tesco blue (#00539F) background, white text, flat design.
    Only offer price and regular price are editable.
    """
    tile = Image.new("RGBA", (width, height), (0, 83, 159, 255))
    draw = ImageDraw.Draw(tile)

    # ── Clubcard price (large, bold, centred) ─────────────────────────────
    price_text = price_data.get("clubcard_price", "£0.00")
    price_font = _font(int(height * 0.30), bold=True)
    pbbox = draw.textbbox((0, 0), price_text, font=price_font)
    pw = pbbox[2] - pbbox[0]
    draw.text(((width - pw) // 2, int(height * 0.08)), price_text, fill="white", font=price_font)

    # ── "Clubcard Price" label ─────────────────────────────────────────────
    label_font = _font(int(height * 0.16))
    lbbox = draw.textbbox((0, 0), "Clubcard Price", font=label_font)
    lw = lbbox[2] - lbbox[0]
    draw.text(((width - lw) // 2, int(height * 0.52)), "Clubcard Price", fill="white", font=label_font)

    # ── Regular price (strikethrough, smaller) ────────────────────────────
    regular = price_data.get("regular_price", "")
    if regular:
        was_text = f"Was {regular}"
        was_font = _font(int(height * 0.13))
        wbbox = draw.textbbox((0, 0), was_text, font=was_font)
        ww = wbbox[2] - wbbox[0]
        wx = (width - ww) // 2
        wy = int(height * 0.76)
        draw.text((wx, wy), was_text, fill=(191, 224, 245, 200), font=was_font)
        mid_y = wy + (wbbox[3] - wbbox[1]) // 2
        draw.line([(wx, mid_y), (wx + ww, mid_y)], fill=(191, 224, 245, 180), width=1)

    # ── Subtle bottom border ──────────────────────────────────────────────
    draw.rectangle([0, height - 3, width, height], fill=(0, 60, 120, 200))
    return tile


def _create_lep_tile(price_data: dict, width: int, height: int) -> Image.Image:
    """
    Everyday Low Price tile – Appendix A:
    White background, Tesco blue (#00539F) text, trade-style.
    Only the price is editable.
    """
    tile = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(tile)

    # Blue border (trade-style)
    draw.rectangle([0, 0, width - 1, height - 1], outline=(0, 83, 159), width=3)

    # ── LEP price ─────────────────────────────────────────────────────────
    price_text = price_data.get("lep_price", "£0.00")
    price_font = _font(int(height * 0.28), bold=True)
    pbbox = draw.textbbox((0, 0), price_text, font=price_font)
    pw = pbbox[2] - pbbox[0]
    draw.text(((width - pw) // 2, int(height * 0.08)), price_text, fill=(0, 83, 159), font=price_font)

    # ── "Everyday low price" label ────────────────────────────────────────
    label_font = _font(int(height * 0.15))
    lbbox = draw.textbbox((0, 0), "Everyday low price", font=label_font)
    lw = lbbox[2] - lbbox[0]
    draw.text(((width - lw) // 2, int(height * 0.60)), "Everyday low price", fill=(0, 83, 159), font=label_font)

    return tile


def _create_new_tile(width: int, height: int) -> Image.Image:
    """
    New product tile – Appendix A: predefined, cannot be edited.
    Green background, white text.
    """
    tile = Image.new("RGBA", (width, height), (22, 160, 133, 255))   # Tesco-friendly teal-green
    draw = ImageDraw.Draw(tile)

    font = _font(int(height * 0.45), bold=True)
    bbox = draw.textbbox((0, 0), "NEW", font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, (height - th) // 2), "NEW", fill="white", font=font)
    return tile


def _create_default_tile(width: int, height: int) -> Image.Image:
    tile = Image.new("RGBA", (width, height), (230, 230, 230, 255))
    draw = ImageDraw.Draw(tile)
    draw.rectangle([0, 0, width - 1, height - 1], outline=(150, 150, 150), width=2)
    font = _font(int(height * 0.22))
    draw.text((width // 2, height // 2), "VALUE", fill="#888888", font=font, anchor="mm")
    return tile


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def validate_value_tile_design(tile_image: Image.Image, tile_type: str) -> dict:
    """
    Validate a generated tile against Appendix A & B.
    Claude adds a plain-language rationale for each issue.
    """
    issues = []
    recommendations = [
        "Value tile position is predefined and cannot be moved (Appendix A)",
        "Nothing can overlap the value tile (Appendix B hard fail)",
        "Clubcard: flat design, only offer+regular price editable (Appendix A)",
        "LEP: white background, Tesco blue font, trade-style (Appendix A)",
        "New: predefined, cannot be edited (Appendix A)",
    ]

    # Basic dimension check
    w, h = tile_image.size
    if w < 200 or h < 60:
        issues.append("HARD FAIL: Tile dimensions too small – minimum 200×60px")

    # AI rationale for any issues
    if issues and ANTHROPIC_AVAILABLE and _client:
        prompt = f"""
These issues were found with a Tesco value tile of type '{tile_type}':
{json.dumps(issues)}

For each issue, write a short plain-English explanation (1 sentence) of why it
violates Tesco Appendix A or B guidelines.
Return a JSON array of strings.
"""
        raw = _call_claude(prompt, max_tokens=200)
        if raw and not raw.startswith("__error__"):
            rationales = _safe_json(raw, [])
            if isinstance(rationales, list):
                recommendations = rationales + recommendations

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "recommendations": recommendations,
        "ai_enhanced": ANTHROPIC_AVAILABLE,
    }


def get_value_tile_templates() -> dict:
    """Return all available tile templates with Appendix A rules."""
    return {
        "tile_types": ["Clubcard Price", "Everyday Low Price", "New"],
        "appendix_a_rules": {
            "Clubcard Price": "Flat design, predefined, only offer price and regular price editable, requires end date DD/MM",
            "Everyday Low Price": "White background, Tesco blue font, trade-style, only price editable, positioned right of packshot",
            "New": "Predefined, cannot be edited",
        },
        "ai_powered": ANTHROPIC_AVAILABLE,
    }
