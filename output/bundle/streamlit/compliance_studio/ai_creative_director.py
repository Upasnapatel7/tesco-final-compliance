"""
ai_creative_director.py
========================
Uses Claude to automatically design a complete, beautiful retail creative.
Claude chooses: layout, background colours, texture style, font family,
font sizes, headline copy, subhead copy, badge styling, and tag line.
The result is passed directly to render_creative() for pixel-perfect output.
"""

import os, json, re, random
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np


import os as _os
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path

# 🔥 Force load .env properly
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

AI_AVAILABLE = False
_client = None

try:
    import anthropic

    _api_key ="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"
    
    print("DEBUG API KEY:", repr(_api_key))

    if _api_key and _api_key.startswith("sk-ant-"):
        _client = anthropic.Anthropic(api_key=_api_key)
        AI_AVAILABLE = True
        print("✅ Claude AI ENABLED")
    else:
        print("❌ API KEY NOT DETECTED")

except Exception as e:
    print("❌ Anthropic error:", e)
    _client = None
    AI_AVAILABLE = False

try:
    from brand_config import FONT_MAP, BRANDS, LAYOUT_PRESETS
except ImportError:
    from brand_config import FONT_MAP, BRANDS
    LAYOUT_PRESETS = {}

# ─────────────────────────────────────────────────────────────────────────────
#  DESIGN SYSTEM CLAUDE KNOWS ABOUT
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a world-class retail advertising creative director.
You design stunning social media creatives for major retail brands.
Your designs always feel premium, on-brand, visually balanced and modern.

When given a brief you return a JSON object describing the complete creative design.
Never include markdown fences. Return only raw JSON.

Available font families:
- "Poppins (Modern)"       – geometric sans, modern & clean
- "Liberation Sans (Clean)"– neutral corporate sans
- "Lora (Elegant Serif)"   – elegant editorial serif, luxury/premium
- "DejaVu Sans (Neutral)"  – neutral, approachable
- "FreeSans (Rounded)"     – friendly, rounded
- "TeX Gyre Heros (Swiss)" – Swiss minimalist, high fashion

Available layout presets:
- "Product Hero"      – product right, copy left, frosted band bottom
- "Centered Minimal"  – product centered, full-width copy band
- "Bold Left"         – product left, bold copy right
- "Full Bleed"        – product fills canvas, gradient scrim for copy
- "Split Panel"       – solid brand colour left panel, product right

Available texture styles (for background decoration):
- "none"           – clean solid/gradient only
- "subtle_dots"    – tiny dot grid pattern
- "diagonal_lines" – fine angled lines
- "corner_circles" – large faint circles in corners
- "geometric_slab" – diagonal geometric accent
- "soft_noise"     – subtle grain texture

Design philosophy:
- Premium brands (M&S, Waitrose, Zara) → serif fonts, minimal decoration, elegant
- Value brands (ASDA, Aldi, Lidl) → bold sans, vibrant colours, strong badge
- Fashion (H&M, Zara) → editorial layouts, minimal colour, large typography
- Grocery (Tesco, Sainsbury's) → friendly, clear hierarchy, prominent badge
- US Retail (Walmart, Target, Amazon) → bold, energetic, high contrast
"""

_DESIGN_PROMPT_TEMPLATE = """Design a beautiful social media creative for the following brief.

Brand: {brand_name}
Brand primary colour: {primary_colour}
Brand background colour: {bg_default}
Product: {product_name}
Product category: {product_category}
Canvas format: {canvas_format}
User headline (if provided): {user_headline}
User subhead (if provided): {user_subhead}
Badge price: {badge_price}
Badge label: {badge_label}

Return a JSON object with EXACTLY these fields:
{{
  "headline": "<compelling headline, max 40 chars, brand-appropriate tone>",
  "subhead": "<supporting subhead, max 60 chars>",
  "headline_size": <integer 28-110, scale to canvas {canvas_h}px tall>,
  "subhead_size": <integer 16-55>,
  "font_name": "<one of the available font families>",
  "headline_color": "<hex colour e.g. #002858>",
  "subhead_color": "<hex colour e.g. #444444>",
  "headline_uppercase": <true or false>,
  "bg_color": "<hex — primary brand background or a beautiful complementary colour>",
  "bg_gradient_bot": "<hex — slightly darker/lighter variation for gradient bottom>",
  "layout_preset": "<one of the layout presets>",
  "texture_style": "<one of the texture styles>",
  "texture_color": "<hex for texture elements, usually a soft tint of primary>",
  "texture_alpha": <integer 6-20, how visible the texture is>,
  "badge_show": <true or false>,
  "badge_shape": "<rounded | pill | circle>",
  "badge_primary_override": "<hex or empty string — override badge colour for extra impact>",
  "tag_text": "<short tag line e.g. 'Available at Tesco' or empty>",
  "design_rationale": "<one sentence: why these choices look great for this brand>"
}}"""


# ─────────────────────────────────────────────────────────────────────────────
#  BEAUTIFUL TEXTURE RENDERER  (applied directly to canvas)
# ─────────────────────────────────────────────────────────────────────────────

def _hex_to_rgb(h):
    h = str(h).strip().lstrip("#")
    if len(h) == 3: h = h[0]*2+h[1]*2+h[2]*2
    try: return tuple(int(h[i:i+2],16) for i in (0,2,4))
    except: return (100,100,100)

def apply_texture(canvas: Image.Image, style: str, colour: str, alpha: int = 12) -> Image.Image:
    """
    Apply a beautiful decorative texture to the canvas.
    All textures are very subtle (low alpha) so they add depth without clutter.
    """
    W, H = canvas.size
    pr = _hex_to_rgb(colour)
    a  = max(4, min(30, alpha))

    ov = Image.new("RGBA", (W, H), (0,0,0,0))
    d  = ImageDraw.Draw(ov)

    if style == "subtle_dots":
        # Fine dot grid — elegant, premium
        sp = max(18, int(W * 0.035))
        r  = max(2, int(sp * 0.18))
        for row in range(sp//2, H+sp, sp):
            for col in range(sp//2, W+sp, sp):
                d.ellipse([col-r, row-r, col+r, row+r], fill=(*pr, a))

    elif style == "diagonal_lines":
        # Fine diagonal stripes — fashion-forward
        gap = max(22, int(W * 0.055))
        thick = max(1, int(W * 0.004))
        for i in range(-H, W+H, gap):
            d.line([(i,0),(i+H,H)], fill=(*pr, a), width=thick)

    elif style == "corner_circles":
        # Large overlapping circles in corners — soft, modern
        r1 = int(min(W,H) * 0.52)
        d.ellipse([W-r1, -r1//2, W+r1//3, r1], fill=(*pr, a-2))
        r2 = int(min(W,H) * 0.32)
        d.ellipse([-r2//3, H-r2, r2, H+r2//3], fill=(*pr, a-4))
        r3 = int(min(W,H) * 0.18)
        d.ellipse([W//2-r3, -r3//2, W//2+r3, r3], fill=(*pr, a-6))

    elif style == "geometric_slab":
        # Bold diagonal accent panels — dynamic, energetic
        pts1 = [(W*0.72, 0), (W*1.0, 0), (W*1.0, H*0.6), (W*0.82, H*0.6)]
        pts2 = [(W*0.86, H*0.4), (W*1.0, H*0.4), (W*1.0, H), (W*0.86, H)]
        d.polygon(pts1, fill=(*pr, a+2))
        d.polygon(pts2, fill=(*pr, a-2))

    elif style == "soft_noise":
        # Perlin-like noise grain — premium, subtle depth
        arr = np.random.randint(0, max(1, a//2), (H, W), dtype=np.uint8)
        noise = Image.fromarray(arr, "L")
        r_ch = Image.new("L", (W,H), pr[0])
        g_ch = Image.new("L", (W,H), pr[1])
        b_ch = Image.new("L", (W,H), pr[2])
        ov = Image.merge("RGBA", (r_ch, g_ch, b_ch, noise))

    elif style == "halftone":
        # Halftone dot pattern — vintage editorial
        sp = max(12, int(W * 0.025))
        for row in range(0, H+sp, sp):
            for col in range(0, W+sp, sp):
                dist = ((col - W//2)**2 + (row - H//2)**2) ** 0.5
                max_d = ((W//2)**2 + (H//2)**2) ** 0.5
                frac  = dist / max_d
                r_dot = max(1, int(sp * 0.35 * frac))
                d.ellipse([col-r_dot, row-r_dot, col+r_dot, row+r_dot],
                           fill=(*pr, min(a+5, int(a * (1 + frac)))))

    elif style == "wave_lines":
        # Curved wave lines — fluid, fresh
        import math
        for j, y_off in enumerate(range(0, H+int(H*0.15), int(H*0.12))):
            pts = []
            steps = 80
            for s in range(steps+1):
                x = int(W * s / steps)
                y = int(y_off + math.sin(s/steps * math.pi * 3 + j) * H * 0.03)
                pts.append((x, y))
            if len(pts) >= 2:
                d.line(pts, fill=(*pr, max(4, a-4)), width=max(1, int(H*0.003)))

    elif style == "crosshatch":
        # Fine crosshatch — tailored, structured
        gap = max(28, int(W*0.06)); thick = max(1, int(W*0.003))
        for i in range(-H, W+H, gap):
            d.line([(i,0),(i+H,H)], fill=(*pr,a-3), width=thick)
            d.line([(i+H,0),(i,H)], fill=(*pr,a-3), width=thick)

    elif style == "grid":
        # Subtle grid — tech, structured
        sp = max(30, int(W*0.07)); thick = max(1, 2)
        for x in range(0, W+sp, sp):
            d.line([(x,0),(x,H)], fill=(*pr, a-4), width=thick)
        for y in range(0, H+sp, sp):
            d.line([(0,y),(W,y)], fill=(*pr, a-4), width=thick)

    if canvas.mode != "RGBA": canvas = canvas.convert("RGBA")
    return Image.alpha_composite(canvas, ov)


# ─────────────────────────────────────────────────────────────────────────────
#  CLAUDE CREATIVE DIRECTOR
# ─────────────────────────────────────────────────────────────────────────────

def _safe_json(text: str, fallback: dict) -> dict:
    try:
        cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
        return json.loads(cleaned)
    except Exception:
        return fallback


def ai_design_creative(
    brand_name:       str,
    product_name:     str,
    product_category: str,
    dimensions:       tuple,
    user_headline:    str = "",
    user_subhead:     str = "",
    badge_price:      str = "",
    badge_label:      str = "",
) -> dict:
    """
    Ask Claude to design the complete creative.
    Returns a design_spec dict ready to pass straight to render_creative().
    Falls back to an algorithmically-generated spec if Claude is unavailable.
    """
    W, H = dimensions
    fmt  = ("Stories (9:16)" if H > W else "Landscape (16:9)" if W > H else "Square (1:1)")
    brand_cfg  = BRANDS.get(brand_name, {})
    primary    = brand_cfg.get("primary", "#333333")
    bg_default = brand_cfg.get("bg_default", "#F5F5F5")

    if AI_AVAILABLE and _client:
        prompt = _DESIGN_PROMPT_TEMPLATE.format(
            brand_name       = brand_name,
            primary_colour   = primary,
            bg_default       = bg_default,
            product_name     = product_name or "product",
            product_category = product_category,
            canvas_format    = fmt,
            canvas_h         = H,
            user_headline    = user_headline or "(none — generate a great one)",
            user_subhead     = user_subhead  or "(none — generate a great one)",
            badge_price      = badge_price   or "(none)",
            badge_label      = badge_label   or "(none)",
        )
        try:
            response = _client.messages.create(
                model      = "claude-3-sonnet-20240229",
                max_tokens = 900,
                system     = _SYSTEM_PROMPT,
                messages   = [{"role": "user", "content": prompt}],
            )
            raw    = response.content[0].text.strip()
            spec   = _safe_json(raw, {})
            if spec and "headline" in spec:
                spec["_source"] = "claude"
                return spec
        except Exception as e:
            import traceback
    print("❌ CLAUDE FULL ERROR:")
    traceback.print_exc() # fall through to algorithmic fallback

    # ── Algorithmic fallback (no API key) ─────────────────────────────────────
    return _algorithmic_design(brand_name, product_name, product_category,
                                dimensions, user_headline, user_subhead,
                                badge_price, badge_label)

def _algorithmic_design(brand_name, product_name, product_category,
                         dimensions, user_headline, user_subhead,
                         badge_price, badge_label) -> dict:
    """
    Generates a beautiful, brand-appropriate creative design without the API.
    Uses a rule-based system with brand-specific design tokens.
    """
    W, H = dimensions
    brand = BRANDS.get(brand_name, {})
    primary    = brand.get("primary",    "#333333")
    bg_default = brand.get("bg_default", "#F5F5F5")
    font_fam   = brand.get("font_family","Poppins")

    # Map brand font preference to FONT_MAP key
    font_map_key = next(
        (k for k in FONT_MAP if font_fam.split()[0].lower() in k.lower()),
        "Poppins (Modern)"
    )

    # Brand aesthetic profiles
    profiles = {
        "luxury":    dict(layout="Centered Minimal", texture="subtle_dots",
                          hl_upper=False, font="Lora (Elegant Serif)",
                          hl_size=int(H*0.055), sh_size=int(H*0.028)),
        "fashion":   dict(layout="Full Bleed",       texture="diagonal_lines",
                          hl_upper=True,  font="TeX Gyre Heros (Swiss)",
                          hl_size=int(H*0.070), sh_size=int(H*0.032)),
        "energetic": dict(layout="Bold Left",        texture="geometric_slab",
                          hl_upper=True,  font="Poppins (Modern)",
                          hl_size=int(H*0.060), sh_size=int(H*0.030)),
        "friendly":  dict(layout="Product Hero",     texture="corner_circles",
                          hl_upper=True,  font="Poppins (Modern)",
                          hl_size=int(H*0.052), sh_size=int(H*0.028)),
        "minimal":   dict(layout="Centered Minimal", texture="none",
                          hl_upper=False, font="Liberation Sans (Clean)",
                          hl_size=int(H*0.048), sh_size=int(H*0.026)),
    }

    brand_profile_map = {
        "Tesco":          "friendly",
        "Sainsbury's":    "friendly",
        "ASDA":           "energetic",
        "Morrisons":      "energetic",
        "Waitrose":       "luxury",
        "Marks & Spencer":"luxury",
        "Aldi":           "energetic",
        "Lidl":           "energetic",
        "Walmart":        "energetic",
        "Target":         "energetic",
        "Amazon":         "minimal",
        "Boots":          "friendly",
        "H&M":            "fashion",
        "Zara":           "luxury",
        "Custom Brand":   "friendly",
    }

    profile_key = brand_profile_map.get(brand_name, "friendly")
    p = profiles[profile_key]

    # Generate copy if user didn't provide
    copy_map = {
        "Grocery":   ("Discover Fresh Flavours",     "Shop the full range in store"),
        "Alcohol":   ("Taste the Difference",        "Award-winning wines & spirits"),
        "Electronics":("Power Your World",           "The latest tech, right here"),
        "Fashion":   ("New Season Arrivals",         "Style that moves with you"),
        "Home & Garden":("Transform Your Space",     "Beautiful home essentials"),
        "Beauty & Health":("Look & Feel Amazing",    "Premium beauty, every day"),
        "General":   ("Something New Awaits",        "Find it in stores near you"),
    }
    default_hl, default_sh = copy_map.get(product_category, copy_map["General"])
    if product_name and product_name.lower() not in (user_headline or "").lower():
        default_hl = f"Discover {product_name}"

    headline  = user_headline  or default_hl
    subhead   = user_subhead   or default_sh

    # Compute gradient bottom colour
    def _darken_hex(h, amt=0.08):
        r,g,b = _hex_to_rgb(h)
        return "#{:02x}{:02x}{:02x}".format(
            max(0,int(r*(1-amt))), max(0,int(g*(1-amt))), max(0,int(b*(1-amt))))

    def _lum(h):
        r,g,b = _hex_to_rgb(h)
        return 0.299*r + 0.587*g + 0.114*b

    hl_color = brand.get("text_dark", "#111111")
    sh_color = "#555555"
    # For dark backgrounds, flip text colours
    if _lum(bg_default) < 120:
        hl_color = "#FFFFFF"; sh_color = "#DDDDDD"

    return {
        "headline":               headline,
        "subhead":                subhead,
        "headline_size":          max(26, p["hl_size"]),
        "subhead_size":           max(15, p["sh_size"]),
        "font_name":              p["font"],
        "headline_color":         hl_color,
        "subhead_color":          sh_color,
        "headline_uppercase":     p["hl_upper"],
        "bg_color":               bg_default,
        "bg_gradient_bot":        _darken_hex(bg_default, 0.07),
        "layout_preset":          p["layout"],
        "texture_style":          p["texture"],
        "texture_color":          primary,
        "texture_alpha":          10,
        "badge_show":             bool(badge_price),
        "badge_shape":            "rounded",
        "badge_primary_override": "",
        "tag_text":               f"Available at {brand_name}" if brand_name != "Custom Brand" else "",
        "design_rationale":       f"Algorithmic {profile_key} design for {brand_name}",
        "_source":                "algorithmic",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  GENERATE MULTIPLE VARIATIONS
# ─────────────────────────────────────────────────────────────────────────────

def generate_ai_variations(
    brand_name:       str,
    product_name:     str,
    product_category: str,
    dimensions:       tuple,
    packshots:        list,
    user_headline:    str = "",
    user_subhead:     str = "",
    badge_price:      str = "",
    badge_label:      str = "",
    n_variations:     int = 3,
) -> list[dict]:
    """
    Generate n_variations complete creative designs.
    Returns list of dicts: {spec, image, source}
    """
    from creative_renderer import render_creative, apply_texture

    specs  = []
    images = []

    # First variation: full AI design (or best algorithmic)
    spec1 = ai_design_creative(
        brand_name, product_name, product_category, dimensions,
        user_headline, user_subhead, badge_price, badge_label,
    )
    specs.append(spec1)

    # Additional variations: genuinely different visual treatments
    alt_layouts  = ["Product Hero", "Split Panel", "Bold Left", "Centered Minimal", "Full Bleed"]
    alt_textures = ["subtle_dots", "corner_circles", "diagonal_lines", "soft_noise",
                    "geometric_slab", "wave_lines", "crosshatch", "halftone", "grid"]
    alt_fonts    = list(FONT_MAP.keys())

    # Colour accent variations (lighter/darker/complementary backgrounds)
    brand_cfg  = BRANDS.get(brand_name, {})
    primary    = brand_cfg.get("primary",    "#333333")
    bg_default = brand_cfg.get("bg_default", "#F5F5F5")

    def _lighten(h, p=0.18):
        r,g,b = _hex_to_rgb(h)
        return "#{:02x}{:02x}{:02x}".format(min(255,int(r+(255-r)*p)),min(255,int(g+(255-g)*p)),min(255,int(b+(255-b)*p)))
    def _tint(h, tint="#ffffff", p=0.15):
        r,g,b = _hex_to_rgb(h); tr,tg,tb = _hex_to_rgb(tint)
        return "#{:02x}{:02x}{:02x}".format(int(r*(1-p)+tr*p),int(g*(1-p)+tg*p),int(b*(1-p)+tb*p))

    alt_bgs = [
        bg_default,
        _lighten(primary, 0.82),      # very light tint of primary
        _lighten(primary, 0.60),      # medium tint
        "#FFFFFF",                     # pure white — clean
        _tint(bg_default, primary, 0.12),  # bg with primary blush
    ]
    alt_hl_sizes = [0, int(dimensions[1]*0.055), int(dimensions[1]*0.072), int(dimensions[1]*0.048), 0]
    alt_uppercase = [True, True, False, False, True]

    for i in range(1, n_variations):
        s = dict(spec1)   # copy base spec
        s["layout_preset"]      = alt_layouts[i % len(alt_layouts)]
        s["texture_style"]      = alt_textures[i % len(alt_textures)]
        s["font_name"]          = alt_fonts[(i+1) % len(alt_fonts)]
        s["texture_alpha"]      = random.randint(8, 18)
        s["bg_color"]           = alt_bgs[i % len(alt_bgs)]
        _bg = s["bg_color"]
        if _bg and _bg != "#FFFFFF":
            _r,_g,_b = _hex_to_rgb(_bg)
            s["bg_gradient_bot"] = "#{:02x}{:02x}{:02x}".format(max(0,int(_r*0.93)),max(0,int(_g*0.93)),max(0,int(_b*0.93)))
        else:
            s["bg_gradient_bot"] = "#F5F5F5"
        s["headline_size"]      = alt_hl_sizes[i % len(alt_hl_sizes)]
        s["headline_uppercase"] = alt_uppercase[i % len(alt_uppercase)]
        s["_source"]            = f"variation_{i}"
        s["design_rationale"]   = f"Variation {i}: {s['layout_preset']} layout · {s['texture_style']} texture · {s['font_name'].split('(')[0].strip()} font"
        specs.append(s)

    # Render all specs
    results = []
    for spec in specs:
        try:
            img = render_creative(
                dimensions        = dimensions,
                packshots         = packshots,
                headline          = spec.get("headline", user_headline or "New Arrival"),
                subhead           = spec.get("subhead",  user_subhead  or "Available now"),
                brand_name        = brand_name,
                badge_label       = badge_label,
                badge_price       = badge_price,
                badge_sub         = "",
                badge_show        = spec.get("badge_show", bool(badge_price)),
                tag_text          = spec.get("tag_text", ""),
                bg_color          = spec.get("bg_color", "#F0F4F8"),
                bg_gradient_bot   = spec.get("bg_gradient_bot", ""),
                bg_image          = None,
                layout_preset     = spec.get("layout_preset", "Product Hero"),
                font_name         = spec.get("font_name", "Poppins (Modern)"),
                headline_size     = spec.get("headline_size", 0),
                subhead_size      = spec.get("subhead_size", 0),
                headline_color    = spec.get("headline_color", ""),
                subhead_color     = spec.get("subhead_color", ""),
                headline_uppercase= spec.get("headline_uppercase", True),
                show_logo         = True,
                include_drinkaware= False,
                packshot_edits    = None,
                packshot_positions= None,
            )

            # Apply texture on top of rendered creative
            tex_style = spec.get("texture_style", "none")
            tex_color = spec.get("texture_color", "#888888")
            tex_alpha = spec.get("texture_alpha", 10)
            if tex_style and tex_style != "none":
                img_rgba = img.convert("RGBA")
                img_rgba = apply_texture(img_rgba, tex_style, tex_color, tex_alpha)
                img = img_rgba.convert("RGB")

            results.append({
                "spec":      spec,
                "image":     img,
                "source":    spec.get("_source", "ai"),
                "rationale": spec.get("design_rationale", ""),
            })
        except Exception as e:
            results.append({
                "spec":      spec,
                "image":     None,
                "source":    "error",
                "rationale": str(e),
            })

    return results
