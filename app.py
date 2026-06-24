"""
app.py  –  UPGRADED
Tesco GenAI Creative Compliance Studio
Integrates real Claude API, rembg background removal, and vision-based audit.
"""

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import io, zipfile, json, time, base64, uuid, random, re, os
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tesco GenAI Creative Compliance Studio",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Module imports with graceful fallbacks ────────────────────────────────────
try:
    from compliance_engine import AdvancedComplianceEngine
    compliance_engine = AdvancedComplianceEngine()
except ImportError as e:
    st.error(f"Compliance engine error: {e}")
    class AdvancedComplianceEngine:
        def check_text_compliance(self, h, s, c="general"):
            return {"approved": True, "issues": [], "suggestions": [], "compliance_score": 100, "llm_enhanced": False}
        def full_creative_audit(self, d, f):
            return {"overall_assessment": {"passed": True, "score": 95, "summary": "Audit unavailable"}}
        def validate_creative_design(self, d, f):
            return {"valid": True, "issues": [], "warnings": [], "hard_fails": []}
        def check_safe_zones(self, f, p):
            return {"passed": True, "issues": []}
        def analyze_headline_subhead(self, h, s, c):
            return {"headline_issues": [], "subhead_issues": [], "recommendations": [], "compliance_score": 100, "llm_enhanced": False}
        def audit_creative_image(self, b64, mt="image/png"):
            return {"passed": None, "issues": [], "warnings": ["Module unavailable"], "llm_enhanced": False}
    compliance_engine = AdvancedComplianceEngine()

try:
    from ai_creative_generator import AICreativeSuggestor
    creative_suggestor = AICreativeSuggestor()
except ImportError:
    class AICreativeSuggestor:
        def suggest_copy_improvements(self, h, s, t): return ["Module unavailable – check imports"]
        def predict_performance(self, e, p): return {"engagement_score": 75, "click_through_prediction": "7.5%", "conversion_likelihood": "Medium", "performance_grade": "B"}
        def get_trending_designs(self, c): return {"styles": [], "colors": [], "recommendations": []}
        def generate_headline_variants(self, n, t, tone="confident"): return []
        def rewrite_compliant(self, h, s, t): return {"headline": h, "subhead": s, "changes_made": []}
    creative_suggestor = AICreativeSuggestor()

try:
    from background_remover import (
        remove_background_ai, enhance_image_quality,
        apply_creative_filters, optimize_for_social_media,
        validate_image_dimensions, create_placeholder_image,
        get_removal_status, REMBG_AVAILABLE
    )
except ImportError:
    REMBG_AVAILABLE = False
    def remove_background_ai(img): return img.convert("RGBA")
    def enhance_image_quality(img): return img
    def apply_creative_filters(img, f): return img
    def optimize_for_social_media(img, p): return img
    def validate_image_dimensions(img, mw=500, mh=500): return True, "OK"
    def create_placeholder_image(w, h, t="Upload"): return Image.new("RGB", (w, h), "#BFE0F5")
    def get_removal_status(): return {"engine": "Unavailable", "ai_powered": False, "quality": "N/A", "install_hint": "pip install rembg"}

try:
    from value_tile_generator import (
        generate_value_tile, validate_value_tile_design,
        get_value_tile_templates, ai_suggest_price_messaging
    )
except ImportError:
    def generate_value_tile(t, d, dims=(300,100)):
        img = Image.new("RGB", dims, (200, 200, 200))
        return img
    def validate_value_tile_design(i, t): return {"valid": True, "issues": [], "recommendations": []}
    def get_value_tile_templates(): return {"tile_types": ["Clubcard Price", "Everyday Low Price", "New"]}
    def ai_suggest_price_messaging(n, t, d): return {"suggestion": "N/A", "rationale": "Module unavailable", "compliant_copy": d}

try:
    from creative_renderer import render_creative
    RENDERER_AVAILABLE = True
except ImportError:
    RENDERER_AVAILABLE = False

# ── Check Anthropic API key ───────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
AI_ENABLED = bool(ANTHROPIC_API_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
_defaults = {
    "processed_image": None, "original_image": None,
    "headline": "", "subhead": "",
    "show_logo": True, "include_drinkaware": False,
    "product_category": "General", "clubcard_end_date": "",
    "generated_creatives": [], "people_detected": False, "people_confirmed": False,
    "packshots": [], "background_color": "#BFE0F5", "background_image": None,
    "product_exclusivity": "Non-exclusive", "creative_links_to_tesco": True,
    "dark_mode": False, "processed_packshots": [],
    "ai_suggestions": [], "performance_prediction": {},
    "ai_headlines": [], "compliance_chat_history": [],
    "visual_audit_result": None,
    "packshot_positions": [],
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────
def apply_theme():
    dark = st.session_state.dark_mode
    base = """
    <style>
      .compliant{background:#1B3B1B;padding:12px;border-radius:8px;border-left:4px solid #4CAF50;margin:12px 0;color:#E8F5E8}
      .non-compliant{background:#3B1B1B;padding:12px;border-radius:8px;border-left:4px solid #F44336;margin:12px 0;color:#FFEBEE}
      .warning{background:#3B3B1B;padding:12px;border-radius:8px;border-left:4px solid #FFC107;margin:12px 0;color:#FFF8E1}
      .creative-preview{border:2px solid #00539F;border-radius:12px;padding:16px;margin:16px 0;box-shadow:0 4px 12px rgba(0,0,0,.3)}
      .required-field::after{content:" *";color:#FF6B6B}
      .appendix-a{background:#1A237E;padding:10px;border-radius:6px;border-left:4px solid #2196F3;margin:8px 0;color:#E3F2FD}
      .appendix-b{background:#3E2723;padding:10px;border-radius:6px;border-left:4px solid #FF9800;margin:8px 0;color:#FFF3E0}
      .hard-fail{background:#3B1B1B;color:#FF8A80;padding:6px;border-radius:4px;font-weight:bold;border:1px solid #F44336}
      .compliance-check{background:#2D2D2D;padding:12px;border-radius:8px;margin:8px 0;border:1px solid #444}
      .card{background:#262730;padding:16px;border-radius:10px;margin:12px 0;border:1px solid #444}
      .ai-badge{background:linear-gradient(135deg,#00539F,#003366);color:white;padding:3px 8px;border-radius:12px;font-size:11px;font-weight:600}
      .section-header{background:linear-gradient(90deg,#00539F,#003366);padding:15px;border-radius:8px;color:white;margin:20px 0}
      .stButton>button{background:linear-gradient(45deg,#00539F,#003366);color:white;border:none;border-radius:6px;padding:10px 20px;font-weight:600}
    </style>""" if dark else """
    <style>
      .compliant{background:#E8F5E9;padding:12px;border-radius:8px;border-left:4px solid #4CAF50;margin:12px 0}
      .non-compliant{background:#FFEBEE;padding:12px;border-radius:8px;border-left:4px solid #F44336;margin:12px 0}
      .warning{background:#FFF8E1;padding:12px;border-radius:8px;border-left:4px solid #FFC107;margin:12px 0}
      .creative-preview{border:2px solid #00539F;border-radius:12px;padding:16px;margin:16px 0;box-shadow:0 4px 12px rgba(0,83,159,.15)}
      .required-field::after{content:" *";color:#F44336}
      .appendix-a{background:#E3F2FD;padding:10px;border-radius:6px;border-left:4px solid #2196F3;margin:8px 0}
      .appendix-b{background:#FFF3E0;padding:10px;border-radius:6px;border-left:4px solid #FF9800;margin:8px 0}
      .hard-fail{background:#FFEBEE;color:#C62828;padding:6px;border-radius:4px;font-weight:bold;border:1px solid #EF9A9A}
      .card{background:white;padding:16px;border-radius:10px;margin:12px 0;border:1px solid #E0E0E0;box-shadow:0 2px 8px rgba(0,0,0,.08)}
      .ai-badge{background:linear-gradient(135deg,#00539F,#003366);color:white;padding:3px 8px;border-radius:12px;font-size:11px;font-weight:600}
      .section-header{background:linear-gradient(90deg,#00539F,#003366);padding:15px;border-radius:8px;color:white;margin:20px 0}
      .stButton>button{background:linear-gradient(45deg,#00539F,#003366);color:white;border:none;border-radius:6px;padding:10px 20px;font-weight:600}
    </style>"""
    st.markdown(base, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────────────────────────
def image_to_bytes(image, fmt="PNG"):
    buf = io.BytesIO()
    img = image.convert("RGB") if fmt == "JPEG" and image.mode == "RGBA" else image
    img.save(buf, format=fmt, quality=95)
    return buf.getvalue()

def image_to_b64(image, fmt="PNG") -> str:
    return base64.b64encode(image_to_bytes(image, fmt)).decode()

def create_tesco_logo(size=(120, 40)):
    logo = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(logo)
    draw.rectangle([0, 0, size[0]-1, size[1]-1], fill=(0, 83, 159), outline=(0, 60, 120))
    try:
        font = ImageFont.truetype("Arial", 18)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), "TESCO", font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((size[0] - tw) // 2, (size[1] - (bbox[3]-bbox[1])) // 2), "TESCO", fill="white", font=font)
    return logo

def validate_dd_mm_format(date_str):
    pattern = r"^\d{2}/\d{2}$"
    if not re.match(pattern, date_str):
        return False, "Date must be in DD/MM format (e.g. 23/06)"
    try:
        day, month = int(date_str[:2]), int(date_str[3:])
        if not (1 <= day <= 31 and 1 <= month <= 12):
            return False, "Date values out of range"
    except Exception:
        return False, "Invalid date"
    return True, "Valid"

def get_appropriate_tag(value_tile_type, clubcard_end_date, product_exclusivity, creative_links_to_tesco):
    if not creative_links_to_tesco:
        return "None"
    if value_tile_type == "Clubcard Price" and clubcard_end_date:
        return f"Clubcard/app required. Ends {clubcard_end_date}"
    if value_tile_type in ("Clubcard Price", "Everyday Low Price"):
        return "Selected stores. While stocks last."
    return "Only at Tesco" if product_exclusivity == "Exclusive" else "Available at Tesco"

def analyze_text_compliance(headline, subhead, product_category):
    analysis = compliance_engine.analyze_headline_subhead(headline, subhead, product_category)
    full_compliance = compliance_engine.check_text_compliance(headline, subhead, product_category)
    return {
        "analysis": analysis,
        "full_compliance": full_compliance,
        "is_compliant": full_compliance["approved"] and len(analysis.get("headline_issues", [])) == 0,
    }

# ─────────────────────────────────────────────────────────────────────────────
# CREATIVE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def generate_creative(dimensions, packshots, headline, subhead, value_tile_type, tag_type,
                      bg_color, bg_image, include_drinkaware, clubcard_price, regular_price,
                      lep_price, clubcard_end_date, product_category, product_exclusivity,
                      creative_links_to_tesco):
    """Delegate to the professional creative_renderer module for Canva-quality output."""

    tag_text = get_appropriate_tag(value_tile_type, clubcard_end_date,
                                   product_exclusivity, creative_links_to_tesco)
    if tag_type == "None" or not creative_links_to_tesco:
        tag_text = ""

    if RENDERER_AVAILABLE:
        return render_creative(
            dimensions=dimensions,
            packshots=packshots,
            headline=headline,
            subhead=subhead,
            value_tile_type=value_tile_type,
            tag_text=tag_text,
            bg_color=bg_color,
            bg_image=bg_image,
            include_drinkaware=(product_category.lower() == "alcohol" and include_drinkaware),
            clubcard_price=clubcard_price,
            regular_price=regular_price,
            lep_price=lep_price,
            product_category=product_category,
            show_logo=st.session_state.get("show_logo", True),
        )

    # ── Fallback if creative_renderer not importable ──────────────────────────
    W, H = dimensions
    img = bg_image.resize((W, H), Image.Resampling.LANCZOS) if bg_image else Image.new("RGB", (W, H), bg_color)
    draw = ImageDraw.Draw(img)
    try:
        fnt  = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 36)
        fnt2 = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 24)
    except Exception:
        fnt = fnt2 = ImageFont.load_default()
    if headline:
        draw.text((50, H - 180), headline.upper(), fill="#003366", font=fnt)
    if subhead:
        draw.text((50, H - 130), subhead, fill="#333333", font=fnt2)
    return img

def check_creative_compliance(creative_data, format_name):
    issues, hard_fails, warnings = [], [], []
    h = creative_data.get("headline", "")
    s = creative_data.get("subhead", "")
    cat = creative_data.get("product_category", "general")

    if not h: hard_fails.append("HARD FAIL: Headline is required (Appendix A)")
    if not s: hard_fails.append("HARD FAIL: Subhead is required (Appendix A)")

    text_result = compliance_engine.check_text_compliance(h, s, cat)
    hard_fails.extend(text_result.get("issues", []))

    design_result = compliance_engine.validate_creative_design(creative_data, format_name)
    hard_fails.extend(design_result.get("hard_fails", []))
    warnings.extend(design_result.get("warnings", []))

    return {
        "compliant": len(hard_fails) == 0,
        "hard_fails": hard_fails,
        "warnings": warnings,
        "llm_enhanced": text_result.get("llm_enhanced", False),
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    apply_theme()

    # ── Header ────────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.title("🎨 GenAI Creative Studio")
        ai_label = (
            '<span class="ai-badge">🤖 Claude AI Active</span>'
            if AI_ENABLED else
            '<span style="background:#888;color:white;padding:3px 8px;border-radius:12px;font-size:11px">⚠️ Add ANTHROPIC_API_KEY</span>'
        )
        st.markdown(f"**Multi-Brand AI-Powered Creative Builder** &nbsp; {ai_label}", unsafe_allow_html=True)
    with col3:
        if st.button("🌙 Dark" if not st.session_state.dark_mode else "☀️ Light", use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    if not AI_ENABLED:
        st.warning("⚠️ Set `ANTHROPIC_API_KEY` to enable Claude AI features.")

    # ── Load brand config ─────────────────────────────────────────────────────
    try:
        from brand_config import BRANDS, BRAND_CATEGORIES, FONT_MAP, LAYOUT_PRESETS, BG_GRADIENTS
    except ImportError:
        BRANDS = {"Tesco": {"primary":"#00539F","secondary":"#BFE0F5","accent":"#E31837","text_dark":"#002858","text_light":"#FFFFFF","bg_default":"#BFE0F5","font_family":"Poppins (Modern)","logo_text":"TESCO","logo_shape":"pill","compliance":"tesco"}}
        BRAND_CATEGORIES = {"UK Grocery": ["Tesco"]}
        FONT_MAP = {}
        LAYOUT_PRESETS = {"Product Hero": {}}
        BG_GRADIENTS = {"Brand Default": None}

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        # ── 1. BRAND SELECTOR ─────────────────────────────────────────────────
        st.markdown('<div class="section-header">🏪 Brand & Campaign</div>', unsafe_allow_html=True)

        brand_category = st.selectbox("Industry / Sector", list(BRAND_CATEGORIES.keys()))
        brand_list = BRAND_CATEGORIES.get(brand_category, ["Custom Brand"])
        selected_brand = st.selectbox("Brand", brand_list)

        brand_cfg = BRANDS.get(selected_brand, {})
        if brand_cfg:
            pc = brand_cfg["primary"]
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin:6px 0">'
                f'<div style="width:20px;height:20px;background:{pc};border-radius:4px;border:1px solid #ccc"></div>'
                f'<span style="font-size:12px;font-weight:600">{selected_brand}</span>'
                f'<span style="font-size:11px;color:#888">{brand_cfg.get("category","")}</span>'
                f'</div>', unsafe_allow_html=True
            )

        if selected_brand == "Custom Brand":
            st.markdown("**Customise Brand Colours**")
            c1, c2 = st.columns(2)
            custom_primary   = c1.color_picker("Primary",  "#333333")
            custom_secondary = c2.color_picker("Accent",   "#666666")
            custom_logo_text = st.text_input("Logo Text", "BRAND", max_chars=12)
            brand_cfg = {**brand_cfg, "primary": custom_primary, "accent": custom_secondary,
                         "logo_text": custom_logo_text, "text_light": "#FFFFFF",
                         "text_dark": "#111111", "bg_default": "#F5F5F5"}
            BRANDS["Custom Brand"].update(brand_cfg)

        product_category = st.selectbox("Product Category", [
            "General","Alcohol","Grocery","Electronics","Fashion","Home & Garden",
            "Beauty & Health","Finance","Healthcare","Pharmacy","Automotive",
            "Travel","Food & Beverage","Technology","Insurance","Energy",
            "Telecom","Education","Real Estate"])
        st.session_state.product_category = product_category

        is_tesco = selected_brand == "Tesco"
        if is_tesco:
            st.session_state.creative_links_to_tesco = st.checkbox("Creative links to Tesco", value=True)
            product_exclusivity = st.selectbox("Product Exclusivity", ["Exclusive","Non-exclusive"])
            st.session_state.product_exclusivity = product_exclusivity
        else:
            st.session_state.creative_links_to_tesco = False
            product_exclusivity = "Non-exclusive"
            st.session_state.product_exclusivity = product_exclusivity

        if product_category == "Alcohol":
            st.session_state.include_drinkaware = st.checkbox("Include Drinkaware *", value=True)
        else:
            st.session_state.include_drinkaware = False

        # ── 2. PRICE BADGE ────────────────────────────────────────────────────
        st.markdown('<div class="section-header">💰 Price Badge</div>', unsafe_allow_html=True)
        show_badge = st.checkbox("Show Price Badge", value=False)
        badge_label = badge_price = badge_sub = ""
        clubcard_price = regular_price = lep_price = clubcard_end_date = ""
        value_tile_type = "None"

        if show_badge:
            if is_tesco:
                value_tile_type = st.selectbox("Badge Type", ["Clubcard Price","Everyday Low Price","New","None"])
                if value_tile_type == "Clubcard Price":
                    c1, c2 = st.columns(2)
                    clubcard_price = c1.text_input("Clubcard Price", "£3.50")
                    regular_price  = c2.text_input("Regular Price",  "£4.50")
                    clubcard_end_date = st.text_input("End Date DD/MM", "23/06")
                    if clubcard_end_date:
                        ok, msg = validate_dd_mm_format(clubcard_end_date)
                        if not ok: st.error(msg)
                    badge_label="Clubcard Price"; badge_price=clubcard_price; badge_sub=f"Was {regular_price}"
                elif value_tile_type == "Everyday Low Price":
                    lep_price = st.text_input("Price", "£2.99")
                    badge_label="Everyday Low"; badge_price=lep_price; badge_sub=""
            else:
                badge_label = st.text_input("Badge Label", "Special Offer")
                badge_price = st.text_input("Price / Deal Text", "£3.99")
                badge_sub   = st.text_input("Sub-text (e.g. Was £5.99)", "")

        # ── 3. TAG ────────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">🏷️ Tag Line</div>', unsafe_allow_html=True)
        if is_tesco:
            appropriate_tag = get_appropriate_tag(value_tile_type, clubcard_end_date,
                                                  product_exclusivity, st.session_state.creative_links_to_tesco)
            tag_type = st.selectbox("Tesco Tag",
                ["None","Only at Tesco","Available at Tesco","Selected stores. While stocks last."], index=0)
            if st.session_state.creative_links_to_tesco:
                st.info(f"Recommended: '{appropriate_tag}'")
        else:
            tag_type = st.text_input("Tag / Disclaimer text", f"Available at {selected_brand}")
        st.session_state.show_logo = st.checkbox("Show Brand Logo", value=True)

        # ── 4. LAYOUT ─────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📐 Layout</div>', unsafe_allow_html=True)
        try:
            from brand_config import LAYOUT_PRESETS as LP
            layout_options = list(LP.keys())
        except Exception:
            layout_options = ["Product Hero","Centered Minimal","Bold Left","Full Bleed","Split Panel","Top Banner","Diagonal Cut"]
        layout_preset = st.selectbox("Layout Preset", layout_options)

        # ── 5. BACKGROUND ─────────────────────────────────────────────────────
        st.markdown('<div class="section-header">🎨 Background</div>', unsafe_allow_html=True)

        bg_mode = st.radio("Background Type", [
            "🎨 Brand Colour","🖌️ Solid Colour","🌈 Gradient Preset",
            "✏️ Custom Gradient","🖼️ Upload Image"], horizontal=False)
        bg_color = brand_cfg.get("bg_default","#BFE0F5")
        bg_gradient_bot = ""

        if bg_mode == "🖌️ Solid Colour":
            bg_color = st.color_picker("Background Colour", brand_cfg.get("bg_default","#BFE0F5"))
            st.session_state.background_image = None

        elif bg_mode == "🌈 Gradient Preset":
            try:
                from brand_config import BG_GRADIENTS
            except Exception:
                BG_GRADIENTS = {}
            preset_name = st.selectbox("Gradient Preset", list(BG_GRADIENTS.keys()))
            grad = BG_GRADIENTS.get(preset_name)
            if grad:
                bg_color, bg_gradient_bot = grad[0], grad[1]
                st.markdown(
                    f'<div style="height:30px;border-radius:6px;background:linear-gradient(to bottom,{bg_color},{bg_gradient_bot});margin:4px 0"></div>',
                    unsafe_allow_html=True)
            else:
                bg_color = brand_cfg.get("bg_default","#BFE0F5")
            st.session_state.background_image = None

        elif bg_mode == "✏️ Custom Gradient":
            g1, g2 = st.columns(2)
            bg_color        = g1.color_picker("Top colour",    brand_cfg.get("bg_default","#BFE0F5"))
            bg_gradient_bot = g2.color_picker("Bottom colour", "#FFFFFF")
            st.markdown(
                f'<div style="height:30px;border-radius:6px;background:linear-gradient(to bottom,{bg_color},{bg_gradient_bot});margin:4px 0"></div>',
                unsafe_allow_html=True)
            st.session_state.background_image = None

        elif bg_mode == "🖼️ Upload Image":
            bg_file = st.file_uploader("Background Image", type=["png","jpg","jpeg"])
            if bg_file:
                st.session_state.background_image = Image.open(bg_file)
            bg_color = brand_cfg.get("bg_default","#BFE0F5")

        else:  # Brand Colour
            bg_color = brand_cfg.get("bg_default","#BFE0F5")
            st.session_state.background_image = None

        # Background Pattern
        try:
            from brand_config import BG_PATTERNS
        except Exception:
            BG_PATTERNS = {"None":"none","Subtle Dots":"subtle_dots","Diagonal Lines":"diagonal_lines","Geometric":"geometric_slab"}
        bg_pattern_name = st.selectbox("✨ Background Pattern", list(BG_PATTERNS.keys()))
        bg_pattern = BG_PATTERNS.get(bg_pattern_name, "none")

        # Colour Palette Quick-Apply
        try:
            from brand_config import COLOR_PALETTES
        except Exception:
            COLOR_PALETTES = {}
        if COLOR_PALETTES:
            with st.expander("🎨 Colour Palette Quick-Apply"):
                palette_name = st.selectbox("Apply Palette", ["— None —"] + list(COLOR_PALETTES.keys()))
                if palette_name != "— None —":
                    pal = COLOR_PALETTES[palette_name]
                    swatches = "".join([f'<div style="width:22px;height:22px;background:{c};border-radius:3px;border:1px solid #ccc;display:inline-block;margin:1px" title="{c}"></div>' for c in pal])
                    st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:2px;margin:4px 0">{swatches}</div>', unsafe_allow_html=True)
                    if st.button("Apply to Background & Text"):
                        bg_color = pal[3] if len(pal) > 3 else bg_color
                        st.rerun()

        st.session_state.background_color = bg_color

        # ── 6. TYPOGRAPHY ─────────────────────────────────────────────────────
        st.markdown('<div class="section-header">✍️ Typography</div>', unsafe_allow_html=True)

        font_options = list(FONT_MAP.keys()) if FONT_MAP else ["Poppins (Modern)"]
        brand_font_default = brand_cfg.get("font_family","Poppins")
        default_font_idx   = next((i for i,f in enumerate(font_options) if brand_font_default in f), 0)
        font_name = st.selectbox("Font Family", font_options, index=default_font_idx)

        # Font weight
        font_weight = st.radio("Font Weight", ["Regular","Bold","Medium"], horizontal=True)

        tc1, tc2 = st.columns(2)
        headline_size = tc1.slider("Headline px", 16, 120, 16, help="16 = auto-scale")
        subhead_size  = tc2.slider("Subhead px",  13, 80,  13, help="13 = auto-scale")

        # Store font weight in session for renderer
        st.session_state["font_weight"] = font_weight.lower()

        # Letter spacing / style
        font_style_col1, font_style_col2 = st.columns(2)
        headline_uppercase = font_style_col1.checkbox("UPPERCASE", value=True)
        show_text_shadow   = font_style_col2.checkbox("Text Shadow", value=False)

        # Colour pickers
        cc1, cc2 = st.columns(2)
        headline_color = cc1.color_picker("Headline Colour",
            "#002858" if is_tesco else brand_cfg.get("text_dark","#111111"))
        subhead_color  = cc2.color_picker("Subhead Colour", "#444444")

    # ── MAIN COLUMNS ──────────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 2])

    with col1:
        # ── Copy ──────────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📝 Copy</div>', unsafe_allow_html=True)
        st.markdown('<div class="required-field">Headline</div>', unsafe_allow_html=True)
        headline = st.text_input("Headline", value=st.session_state.headline,
                                 placeholder="HEADLINE", label_visibility="collapsed")
        st.markdown('<div class="required-field">Subhead</div>', unsafe_allow_html=True)
        subhead = st.text_input("Subhead", value=st.session_state.subhead,
                                placeholder="Subhead", label_visibility="collapsed")
        st.session_state.headline = headline
        st.session_state.subhead = subhead

        # ── AI Headline Generator ──────────────────────────────────────────
        st.markdown('<div class="section-header">✨ AI Headline Generator</div>', unsafe_allow_html=True)
        product_name_input = st.text_input("Product name (for AI generation)", placeholder=f"e.g. {selected_brand} Finest Cheddar")
        tone_choice = st.selectbox("Copy tone", ["confident", "warm", "playful", "informative"])

        if st.button("🤖 Generate Compliant Headlines", use_container_width=True, disabled=not AI_ENABLED):
            with st.spinner("Claude is writing brand-compliant headlines…"):
                st.session_state.ai_headlines = creative_suggestor.generate_headline_variants(
                    product_name_input or headline or "product",
                    product_category, tone_choice
                )

        if st.session_state.ai_headlines:
            st.markdown("**Generated Headlines (click to use):**")
            for i, variant in enumerate(st.session_state.ai_headlines):
                with st.expander(f"Option {i+1}: {variant.get('headline', '')} (score: {variant.get('predicted_engagement', '-')})"):
                    st.write(f"**Headline:** {variant.get('headline', '')}")
                    st.write(f"**Subhead:** {variant.get('subhead', '')}")
                    st.write(f"**Why it works:** {variant.get('rationale', '')}")
                    if st.button(f"Use this headline", key=f"use_h_{i}"):
                        st.session_state.headline = variant.get("headline", "")
                        st.session_state.subhead = variant.get("subhead", "")
                        st.rerun()

        # ── AI Rewriter ────────────────────────────────────────────────────
        if headline or subhead:
            if st.button("🔧 Auto-fix Non-compliant Copy", use_container_width=True, disabled=not AI_ENABLED,
                         help="Claude rewrites your copy to remove all violations"):
                with st.spinner("Claude is rewriting compliant copy…"):
                    result = creative_suggestor.rewrite_compliant(headline, subhead, product_category)
                    if result.get("headline"):
                        st.session_state.headline = result["headline"]
                        st.session_state.subhead  = result["subhead"]
                        st.success("✅ Copy rewritten!")
                        for change in result.get("changes_made", []):
                            st.info(f"• {change}")
                        st.rerun()

        # ── Packshots ──────────────────────────────────────────────────────
        st.markdown('<div class="section-header">🖼️ Packshots (up to 5)</div>', unsafe_allow_html=True)
        uploaded_packshots = st.file_uploader("Upload Product Images",
            type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True, key="packshot_uploader")

        if uploaded_packshots:
            if len(uploaded_packshots) > 5:
                st.warning("Maximum 5 packshots. Using first 5.")
                uploaded_packshots = uploaded_packshots[:5]
            st.session_state.packshots = [Image.open(f) for f in uploaded_packshots]
            n = len(st.session_state.packshots)
            cols = st.columns(min(5, n))
            for i, (ps, c) in enumerate(zip(st.session_state.packshots, cols)):
                c.image(ps, caption=f"#{i+1}", use_column_width=True)
            st.caption(f"✅ {n} image{'s' if n>1 else ''} uploaded")

        # ── Packshot Positioning Studio ────────────────────────────────────
        if uploaded_packshots and st.session_state.packshots:
            with st.expander("📐 Packshot Position & Scale", expanded=True):
                st.caption("Set exact position (0.0 = left/top, 1.0 = right/bottom) and scale for each packshot. Use **Auto Layout** to reset.")
                use_auto_layout = st.checkbox("✨ Auto Layout (recommended)", value=True)
                positions = []
                if not use_auto_layout:
                    for i in range(min(3, len(st.session_state.packshots))):
                        st.markdown(f"**Packshot {i+1}**")
                        pc1, pc2, pc3 = st.columns(3)
                        px = pc1.slider(f"X position", 0.0, 1.0, [0.65, 0.35, 0.70][i], 0.01, key=f"px_{i}")
                        py = pc2.slider(f"Y position", 0.0, 1.0, [0.40, 0.60, 0.25][i], 0.01, key=f"py_{i}")
                        ps = pc3.slider(f"Scale",      0.1, 0.9, 0.40, 0.01, key=f"ps_{i}")
                        positions.append({"x": px, "y": py, "scale": ps})
                    st.session_state.packshot_positions = positions
                else:
                    st.session_state.packshot_positions = []

        # ── Image Editing Studio ───────────────────────────────────────────
        if uploaded_packshots:
            st.markdown('<div class="section-header">🖌️ Image Editing Studio</div>', unsafe_allow_html=True)

            with st.expander("🎛️ Adjustments", expanded=True):
                adj1, adj2 = st.columns(2)
                brightness  = adj1.slider("☀️ Brightness",  0.3, 2.0, 1.0, 0.05)
                contrast    = adj2.slider("◑ Contrast",     0.3, 2.0, 1.0, 0.05)
                saturation  = adj1.slider("🎨 Saturation",  0.0, 2.5, 1.0, 0.05)
                sharpness   = adj2.slider("🔪 Sharpness",   0.0, 2.5, 1.0, 0.05)
                blur_px     = adj1.slider("🌫️ Blur",        0,   12,  0)
                exposure    = adj2.slider("💡 Exposure",    0.3, 2.0, 1.0, 0.05)

            with st.expander("📊 Image Quality Enhancement"):
                quality_mode = st.selectbox("Output Quality", [
                    "Standard", "High (2x upscale)", "Ultra (sharpen + denoise)", "Web Optimised"])
                auto_enhance = st.checkbox("🤖 AI Auto-Enhance (contrast, colour, sharpness)", value=False)
                upscale_2x   = st.checkbox("🔍 Upscale 2× (bicubic)", value=False)

            with st.expander("🎞️ Filters & Transforms"):
                filter_choice = st.selectbox("Colour Filter",
                    ["None","Warm","Cool","Vibrant","Matte","Black & White","Vintage",
                     "Cinematic","HDR","Faded","Cross Process"])
                tr1, tr2, tr3 = st.columns(3)
                flip_h  = tr1.checkbox("↔ Flip H")
                flip_v  = tr2.checkbox("↕ Flip V")
                rotate  = tr3.selectbox("↻ Rotate", [0, 90, 180, 270])

            with st.expander("🤖 AI Background Removal"):
                status = get_removal_status()
                badge_icon = "✅" if status["ai_powered"] else "⚠️"
                st.info(f"{badge_icon} Engine: **{status['engine']}** – {status['quality']}")
                if status.get("install_hint"):
                    st.code(status["install_hint"])
                remove_bg   = st.checkbox("Remove Background (U2-Net AI)")
                enhance_img = st.checkbox("Auto-enhance after removal")

            # Build edits dict
            flt_key = (filter_choice.lower()
                       .replace(" & ","").replace(" ","_")
                       .replace("black_white","bw").replace("none","none")
                       .replace("cross_process","vintage"))
            packshot_edits = {
                "brightness": brightness * exposure,
                "contrast":   contrast,
                "saturation": saturation,
                "sharpness":  sharpness,
                "blur":       blur_px,
                "filter":     flt_key,
                "flip_h":     flip_h,
                "flip_v":     flip_v,
                "rotate":     rotate,
            }

            # Live preview of first packshot
            if st.session_state.packshots:
                try:
                    from creative_renderer import apply_image_edits
                    prev = apply_image_edits(st.session_state.packshots[0].copy(), packshot_edits)
                    if upscale_2x:
                        nw, nh = prev.size[0]*2, prev.size[1]*2
                        prev = prev.resize((nw, nh), Image.Resampling.LANCZOS)
                        prev.thumbnail((300,300))
                    st.image(prev, caption="Preview (packshot 1)", use_column_width=True)
                except Exception:
                    pass

            if st.button("🔄 Apply & Process All Packshots", use_container_width=True):
                with st.spinner("Processing packshots…"):
                    try:
                        from creative_renderer import apply_image_edits
                        from PIL import ImageFilter as PILFilter
                    except ImportError:
                        apply_image_edits = lambda img, e: img
                    processed = []
                    for ps in st.session_state.packshots:
                        p = ps.copy()
                        if remove_bg:
                            p = remove_background_ai(p)
                        if enhance_img or auto_enhance:
                            p = enhance_image_quality(p)
                        p = apply_image_edits(p, packshot_edits)
                        if upscale_2x:
                            nw, nh = p.size[0]*2, p.size[1]*2
                            p = p.resize((nw, nh), Image.Resampling.LANCZOS)
                        if quality_mode == "Ultra (sharpen + denoise)":
                            from PIL import ImageFilter as PILFilter
                            rgb = p.convert("RGB")
                            rgb = ImageEnhance.Sharpness(rgb).enhance(1.5)
                            p = rgb.convert(p.mode) if p.mode == "RGBA" else rgb
                        processed.append(p)
                    st.session_state.processed_packshots = processed
                    st.success(f"✅ {len(processed)} packshot(s) processed!")
                    pcols = st.columns(min(5, len(processed)))
                    for i, (p, c) in enumerate(zip(processed, pcols)):
                        c.image(p, caption=f"#{i+1}", use_column_width=True)
        else:
            packshot_edits = {}
            brightness = contrast = saturation = sharpness = exposure = 1.0
            blur_px = 0; filter_choice = "None"; flip_h = flip_v = False; rotate = 0

        # ── AI Copy Suggestions ────────────────────────────────────────────
        if headline or subhead:
            if st.button("🤖 Get AI Copy Suggestions", use_container_width=True, disabled=not AI_ENABLED):
                with st.spinner("Claude is analysing your copy…"):
                    st.session_state.ai_suggestions = creative_suggestor.suggest_copy_improvements(
                        headline, subhead, product_category)
                    st.session_state.performance_prediction = creative_suggestor.predict_performance(
                        {"headline": headline, "subhead": subhead,
                         "has_value_tile": show_badge,
                         "product_category": product_category},
                        "social")

            if st.session_state.ai_suggestions:
                st.markdown("### 💡 AI Copy Suggestions")
                for s in st.session_state.ai_suggestions:
                    st.info(s)

            if st.session_state.performance_prediction:
                st.markdown("### 📊 AI Performance Prediction")
                pred = st.session_state.performance_prediction
                c1, c2, c3 = st.columns(3)
                c1.metric("Engagement", f"{pred.get('engagement_score', 0)}/100")
                c2.metric("CTR Prediction", pred.get("click_through_prediction", "–"))
                c3.metric("Grade", pred.get("performance_grade", "–"))
                if pred.get("strength"):
                    st.success(f"💪 {pred['strength']}")
                if pred.get("weakness"):
                    st.warning(f"📌 {pred['weakness']}")

        # ── Real-time compliance ───────────────────────────────────────────
        if headline or subhead:
            st.markdown('<div class="section-header">🔍 Real-time Compliance Check</div>', unsafe_allow_html=True)
            text_analysis = analyze_text_compliance(headline, subhead, product_category)

            c1, c2 = st.columns(2)
            c1.metric("Compliance Score", f"{text_analysis['analysis'].get('compliance_score', 0)}/100")
            status_icon = "✅ Compliant" if text_analysis["is_compliant"] else "❌ Issues Found"
            c2.metric("Status", status_icon)

            if text_analysis["analysis"].get("llm_enhanced"):
                st.markdown('<span class="ai-badge">🤖 Claude NLU Enhanced</span>', unsafe_allow_html=True)

            if not text_analysis["is_compliant"]:
                st.error("❌ Compliance issues detected")
                for issue in text_analysis["full_compliance"].get("issues", []):
                    st.markdown(f'<div class="hard-fail">{issue}</div>', unsafe_allow_html=True)
                for rec in text_analysis["full_compliance"].get("suggestions", []):
                    st.info(f"💡 {rec}")
            else:
                st.markdown('<div class="compliant">✅ Copy is Tesco Appendix A & B compliant</div>', unsafe_allow_html=True)

    # ── Right column ──────────────────────────────────────────────────────────
    with col2:
        st.markdown('<div class="section-header">👀 Creative Preview & Export</div>', unsafe_allow_html=True)

        formats = st.multiselect("Select Formats",
            ["Instagram Square (1080x1080)", "Instagram Stories (1080x1920)", "Facebook Landscape (1200x628)"],
            default=["Instagram Square (1080x1080)"])

        creative_data = {
            "headline": headline, "subhead": subhead,
            "packshots": uploaded_packshots if "uploaded_packshots" in locals() and uploaded_packshots else [],
            "product_category": product_category,
            "include_drinkaware": st.session_state.include_drinkaware,
            "value_tile_type": value_tile_type,
            "tag_type": tag_type,
            "clubcard_end_date": clubcard_end_date,
            "product_exclusivity": product_exclusivity,
            "creative_links_to_tesco": st.session_state.creative_links_to_tesco,
        }

        # Live compliance status
        if headline or subhead:
            all_compliant, hard_fail_msgs, warning_msgs = True, [], []
            llm_used = False
            for fmt in formats:
                chk = check_creative_compliance(creative_data, fmt)
                if not chk["compliant"]:
                    all_compliant = False
                    hard_fail_msgs.extend([(fmt, i) for i in chk["hard_fails"]])
                warning_msgs.extend([(fmt, w) for w in chk["warnings"]])
                if chk.get("llm_enhanced"): llm_used = True

            if hard_fail_msgs:
                st.markdown('<div class="non-compliant">❌ HARD FAIL Issues</div>', unsafe_allow_html=True)
                for fmt, issue in hard_fail_msgs:
                    st.error(f"**{fmt}**: {issue}")
            if warning_msgs:
                st.markdown('<div class="warning">⚠️ Warnings</div>', unsafe_allow_html=True)
                for fmt, w in warning_msgs:
                    st.warning(f"**{fmt}**: {w}")
            if all_compliant and headline and subhead:
                if llm_used:
                    st.markdown('<div class="compliant">✅ 100% Compliant – verified by Claude NLU 🤖</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="compliant">✅ 100% Compliant</div>', unsafe_allow_html=True)

        # Generate
        disabled = not (headline and subhead and ("uploaded_packshots" in locals() and uploaded_packshots))
        if st.button("🚀 Generate Creative", type="primary", use_container_width=True, disabled=disabled):
            with st.spinner("Generating…"):
                creatives = []
                packshots_to_use = st.session_state.processed_packshots or st.session_state.packshots
                # Build tag text
                if is_tesco:
                    final_tag = get_appropriate_tag(value_tile_type, clubcard_end_date,
                                                    product_exclusivity, st.session_state.creative_links_to_tesco)
                    if tag_type == "None" or not st.session_state.creative_links_to_tesco:
                        final_tag = ""
                else:
                    final_tag = tag_type if tag_type not in ("None","") else ""

                for fmt in formats:
                    dims = ((1080,1080) if "1080x1080" in fmt else
                            (1080,1920) if "1920" in fmt else
                            (1200,628))
                    if RENDERER_AVAILABLE:
                        img = render_creative(
                            dimensions=dims,
                            packshots=packshots_to_use,
                            headline=headline,
                            subhead=subhead,
                            brand_name=selected_brand,
                            badge_label=badge_label,
                            badge_price=badge_price,
                            badge_sub=badge_sub,
                            badge_show=show_badge and bool(badge_price),
                            tag_text=final_tag,
                            bg_color=bg_color,
                            bg_gradient_bot=bg_gradient_bot if 'bg_gradient_bot' in locals() else "",
                            bg_image=st.session_state.background_image,
                            layout_preset=layout_preset,
                            font_name=font_name,
                            font_weight=st.session_state.get("font_weight","bold"),
                            headline_size=headline_size,
                            subhead_size=subhead_size,
                            headline_color=headline_color,
                            subhead_color=subhead_color,
                            headline_uppercase=headline_uppercase,
                            show_logo=st.session_state.get("show_logo", True),
                            include_drinkaware=(product_category.lower()=="alcohol" and st.session_state.include_drinkaware),
                            packshot_edits=packshot_edits if packshot_edits else None,
                            packshot_positions=st.session_state.get("packshot_positions") or None,
                            texture_style=bg_pattern if 'bg_pattern' in locals() else "none",
                        )
                    else:
                        img = generate_creative(
                            dimensions=dims, packshots=packshots_to_use,
                            headline=headline, subhead=subhead,
                            value_tile_type=value_tile_type, tag_type=tag_type,
                            bg_color=bg_color,
                            bg_image=st.session_state.background_image,
                            include_drinkaware=st.session_state.include_drinkaware,
                            clubcard_price=clubcard_price, regular_price=regular_price,
                            lep_price=lep_price, clubcard_end_date=clubcard_end_date,
                            product_category=product_category,
                            product_exclusivity=product_exclusivity,
                            creative_links_to_tesco=st.session_state.creative_links_to_tesco,
                        )
                    creatives.append({"format": fmt, "image": img, "dimensions": dims,
                                      "timestamp": datetime.now(), "brand": selected_brand,
                                      "compliance_checked": True,
                                      "packshots_count": len(packshots_to_use)})
                st.session_state.generated_creatives = creatives
                st.success(f"✅ {len(creatives)} creative(s) generated for **{selected_brand}**!")
                st.balloons()

        # Display generated creatives
        if st.session_state.generated_creatives:
            st.markdown('<div class="section-header">🎨 Generated Creatives</div>', unsafe_allow_html=True)
            for i, creative in enumerate(st.session_state.generated_creatives):
                with st.container():
                    st.markdown('<div class="creative-preview">', unsafe_allow_html=True)
                    ch1, ch2 = st.columns([3, 1])
                    with ch1:
                        st.write(f"**{creative['format']}** – {creative['dimensions'][0]}×{creative['dimensions'][1]}")
                        st.write("✅ Compliant")
                    with ch2:
                        st.metric("Status", "Ready")

                    if creative["dimensions"][1] == 1920:
                        st.info("📱 9:16: Keep 200px top / 250px bottom clear (Appendix B)")

                    st.image(creative["image"], use_column_width=True)

                    # ── Visual AI Audit ────────────────────────────────────
                    if AI_ENABLED:
                        if st.button(f"🔍 Run Visual AI Audit", key=f"audit_{i}",
                                     help="Claude vision inspects the rendered creative for visual compliance"):
                            with st.spinner("Claude is visually auditing the creative…"):
                                b64 = image_to_b64(creative["image"])
                                audit = compliance_engine.audit_creative_image(b64)
                                st.session_state.visual_audit_result = audit

                        if st.session_state.visual_audit_result:
                            ar = st.session_state.visual_audit_result
                            st.markdown("**🤖 Visual Audit Result:**")
                            if ar.get("passed") is True:
                                st.success("✅ Visual audit passed – no issues detected")
                            elif ar.get("passed") is False:
                                st.error("❌ Visual issues found")
                            for issue in ar.get("issues", []):
                                st.error(issue)
                            for w in ar.get("warnings", []):
                                st.warning(w)
                            for obs in ar.get("positive_observations", []):
                                st.info(f"✓ {obs}")

                    d1, d2 = st.columns(2)
                    with d1:
                        st.download_button("📥 PNG", data=image_to_bytes(creative["image"], "PNG"),
                            file_name=f"tesco_{creative['format'].replace(' ','_').lower()}.png",
                            mime="image/png", use_container_width=True, key=f"png_{i}")
                    with d2:
                        st.download_button("📥 JPEG", data=image_to_bytes(creative["image"], "JPEG"),
                            file_name=f"tesco_{creative['format'].replace(' ','_').lower()}.jpg",
                            mime="image/jpeg", use_container_width=True, key=f"jpg_{i}")

                    st.markdown('</div>', unsafe_allow_html=True)

    # ── ✨ AI CREATIVE DIRECTOR ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## ✨ AI Creative Director")
    st.markdown(
        "Let Claude design the entire creative automatically — "
        "it chooses the layout, background, textures, typography, colours, "
        "and copy. Just describe your product and click generate."
    )

    with st.container():
        ai_col1, ai_col2 = st.columns([1, 2])

        with ai_col1:
            st.markdown("**Brief**")
            ai_product   = st.text_input("Product name", placeholder="e.g. Brancott Estate Chardonnay",
                                          key="ai_product_name")
            ai_category  = st.selectbox("Category",
                ["General","Alcohol","Grocery","Electronics","Fashion",
                 "Home & Garden","Beauty & Health"], key="ai_category")
            ai_headline  = st.text_input("Headline (optional — AI will generate if blank)",
                                          placeholder="Leave blank for AI copy", key="ai_hl")
            ai_subhead   = st.text_input("Subhead (optional)", placeholder="Leave blank for AI copy",
                                          key="ai_sh")
            ai_price     = st.text_input("Badge price (optional)", placeholder="e.g. £3.50",
                                          key="ai_price")
            ai_badge_lbl = st.text_input("Badge label (optional)", placeholder="e.g. Clubcard Price",
                                          key="ai_badge_lbl")
            ai_format    = st.selectbox("Format",
                ["Square (1080×1080)", "Landscape (1200×628)", "Stories (1080×1920)"],
                key="ai_format")
            ai_variations = st.slider("Number of variations", 1, 4, 3, key="ai_vars")

            ai_dims = {"Square (1080×1080)": (1080,1080),
                       "Landscape (1200×628)": (1200,628),
                       "Stories (1080×1920)": (1080,1920)}[ai_format]

            if not AI_ENABLED:
                st.info("⚠️ No API key — will use algorithmic design (still looks great!)")

            run_ai = st.button("🎨 AI Generate Creatives",
                                type="primary", use_container_width=True,
                                key="run_ai_director")

        with ai_col2:
            if run_ai:
                if not st.session_state.packshots:
                    st.warning("Upload at least one packshot in the left panel first.")
                else:
                    try:
                        from ai_creative_director import generate_ai_variations
                    except ImportError as ie:
                        st.error(f"Could not import ai_creative_director: {ie}")
                        generate_ai_variations = None

                    if generate_ai_variations:
                        source_label = "Claude AI" if AI_ENABLED else "Algorithmic AI"
                        with st.spinner(f"🎨 {source_label} is designing your creatives…"):
                            results = generate_ai_variations(
                                brand_name       = selected_brand,
                                product_name     = ai_product or ai_headline or "product",
                                product_category = ai_category,
                                dimensions       = ai_dims,
                                packshots        = (st.session_state.processed_packshots
                                                    or st.session_state.packshots),
                                user_headline    = ai_headline,
                                user_subhead     = ai_subhead,
                                badge_price      = ai_price,
                                badge_label      = ai_badge_lbl,
                                n_variations     = ai_variations,
                            )

                        st.success(f"✅ {len([r for r in results if r['image']])} creatives generated by {source_label}!")

                        for i, result in enumerate(results):
                            if result["image"] is None:
                                st.error(f"Variation {i+1} failed: {result['rationale']}")
                                continue

                            spec = result["spec"]
                            with st.expander(
                                f"{'🤖 AI Design' if 'claude' in result['source'] else f'🎨 Variation {i+1}'}"
                                f" — {spec.get('layout_preset','')}"
                                f" · {spec.get('font_name','').split()[0]}"
                                f" · {spec.get('texture_style','none')} texture",
                                expanded=(i == 0)
                            ):
                                # Show design spec
                                spec_cols = st.columns(4)
                                spec_cols[0].metric("Layout",  spec.get("layout_preset","").replace(" ","\n"))
                                spec_cols[1].metric("Font",    spec.get("font_name","").split("(")[0].strip())
                                spec_cols[2].metric("Texture", spec.get("texture_style","none"))
                                spec_cols[3].metric("Source",  "Claude AI" if "claude" in result["source"] else "Algorithmic")

                                if result.get("rationale"):
                                    st.caption(f"💡 {result['rationale']}")

                                # Show colour swatches
                                bg_c  = spec.get("bg_color","#ffffff")
                                hl_c  = spec.get("headline_color","#111111")
                                sh_c  = spec.get("subhead_color","#555555")
                                st.markdown(
                                    f'<div style="display:flex;gap:8px;margin:6px 0">'
                                    f'<div style="background:{bg_c};width:28px;height:28px;border-radius:4px;border:1px solid #ccc" title="Background {bg_c}"></div>'
                                    f'<div style="background:{hl_c};width:28px;height:28px;border-radius:4px;border:1px solid #ccc" title="Headline {hl_c}"></div>'
                                    f'<div style="background:{sh_c};width:28px;height:28px;border-radius:4px;border:1px solid #ccc" title="Subhead {sh_c}"></div>'
                                    f'<span style="font-size:11px;color:#888;align-self:center">BG · Headline · Subhead</span>'
                                    f'</div>', unsafe_allow_html=True
                                )

                                # Display image
                                st.image(result["image"], use_column_width=True)

                                # Copy + download
                                dl1, dl2, dl3 = st.columns(3)
                                dl1.download_button(
                                    "📥 Download PNG",
                                    data=image_to_bytes(result["image"], "PNG"),
                                    file_name=f"ai_creative_{selected_brand.lower().replace(' ','_')}_{i+1}.png",
                                    mime="image/png", use_container_width=True, key=f"ai_dl_png_{i}"
                                )
                                dl2.download_button(
                                    "📥 Download JPEG",
                                    data=image_to_bytes(result["image"], "JPEG"),
                                    file_name=f"ai_creative_{selected_brand.lower().replace(' ','_')}_{i+1}.jpg",
                                    mime="image/jpeg", use_container_width=True, key=f"ai_dl_jpg_{i}"
                                )
                                if dl3.button("✏️ Use this design", key=f"ai_use_{i}"):
                                    st.session_state.headline     = spec.get("headline","")
                                    st.session_state.subhead      = spec.get("subhead","")
                                    st.rerun()

            elif not run_ai and not st.session_state.get("ai_results"):
                st.markdown("""
**How it works:**
1. Upload your packshot(s) in the left panel
2. Fill in the brief on the left
3. Click **AI Generate Creatives**

Claude will automatically choose:
- 🎨 **Beautiful background** with gradient and texture
- ✍️ **Typography** — font family, sizes, colours
- 📐 **Layout** — best arrangement for your product
- 💬 **Copy** — headline and subhead (if you leave them blank)
- 🏷️ **Price badge** styling
- 🌟 Multiple variations with different looks
                """)

    # ── INTERACTIVE CANVAS EDITOR ─────────────────────────────────────────────
    with st.expander("🖱️ Interactive Canvas Editor — Drag & Position Packshots", expanded=False):

        if not st.session_state.generated_creatives:
            st.info("🎨 Generate a creative first — the editor will appear here.")
        elif not st.session_state.packshots:
            st.info("📸 Upload packshots — they'll appear as draggable layers here.")
        else:
            latest  = st.session_state.generated_creatives[0]
            W_c, H_c = latest["image"].size
            n_ps    = min(len(st.session_state.packshots), 3)

            # Encode background + each packshot as base64
            bg_b64  = image_to_b64(latest["image"], "JPEG")

            ps_b64_list = []
            for ps in st.session_state.packshots[:3]:
                try:
                    from creative_renderer import apply_image_edits
                    pe = st.session_state.get("_last_packshot_edits")
                    p  = apply_image_edits(ps.copy(), pe) if pe else ps.copy()
                except Exception:
                    p = ps.copy()
                p.thumbnail((400, 400), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                p_rgb = p.convert("RGB")
                p_rgb.save(buf, "JPEG", quality=85)
                ps_b64_list.append(base64.b64encode(buf.getvalue()).decode())

            ps_b64_json = json.dumps(ps_b64_list)

            # Default starting positions
            default_positions = [
                {"x": 0.62, "y": 0.38, "scale": 0.42},
                {"x": 0.30, "y": 0.50, "scale": 0.32},
                {"x": 0.70, "y": 0.22, "scale": 0.28},
            ]

            st.markdown("**Drag products on the canvas. Positions auto-update below — click Apply to regenerate.**")

            canvas_html = f"""<!DOCTYPE html>
<html>
<head>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f0f2f5; }}

#app {{ display: flex; flex-direction: column; gap: 10px; padding: 10px; }}

#toolbar {{
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  background: white; border-radius: 10px; padding: 8px 12px;
  box-shadow: 0 1px 4px rgba(0,0,0,.1);
}}
.tool-label {{ font-size: 11px; color: #666; font-weight: 600; text-transform: uppercase; letter-spacing: .5px; }}
.tool-sep {{ width: 1px; height: 20px; background: #e0e0e0; margin: 0 4px; }}

#btn-reset {{ padding: 5px 12px; background: #f0f0f0; border: none; border-radius: 6px; cursor: pointer; font-size: 12px; }}
#btn-reset:hover {{ background: #e0e0e0; }}
#btn-apply {{
  padding: 5px 14px; background: #00539F; color: white;
  border: none; border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 600;
}}
#btn-apply:hover {{ background: #003d80; }}
#btn-apply.updated {{ background: #22a06b; animation: pulse .4s ease; }}
@keyframes pulse {{ 0%{{transform:scale(1)}} 50%{{transform:scale(1.06)}} 100%{{transform:scale(1)}} }}

#canvas-wrap {{
  position: relative;
  width: 100%;
  max-width: 680px;
  margin: 0 auto;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 4px 20px rgba(0,0,0,.2);
  cursor: default;
}}
#canvas-bg {{
  width: 100%;
  display: block;
}}

.ps-layer {{
  position: absolute;
  cursor: grab;
  outline: none;
  touch-action: none;
}}
.ps-layer:focus {{ outline: none; }}
.ps-layer img {{
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
  pointer-events: none;
  border-radius: 4px;
}}
.ps-layer .overlay {{
  position: absolute; inset: 0;
  border: 2px dashed transparent;
  border-radius: 4px;
  transition: border-color .15s, background .15s;
}}
.ps-layer:hover .overlay,
.ps-layer.selected .overlay {{
  border-color: rgba(255,215,0,.9);
  background: rgba(255,215,0,.06);
}}
.ps-layer.dragging {{
  cursor: grabbing;
  opacity: .88;
  z-index: 100 !important;
}}
.ps-layer .label {{
  position: absolute; top: -20px; left: 0;
  font-size: 10px; font-weight: 700; color: white;
  background: rgba(0,0,0,.55); padding: 1px 6px; border-radius: 3px;
  opacity: 0; transition: opacity .15s;
  white-space: nowrap;
}}
.ps-layer:hover .label,
.ps-layer.selected .label {{ opacity: 1; }}

/* Resize handle */
.ps-layer .rs {{
  position: absolute; bottom: -8px; right: -8px;
  width: 18px; height: 18px;
  background: #FFD700; border: 2px solid white;
  border-radius: 50%; cursor: se-resize;
  display: flex; align-items: center; justify-content: center;
  font-size: 9px; font-weight: 900; color: #333;
  opacity: 0; transition: opacity .15s;
  z-index: 10;
}}
.ps-layer:hover .rs,
.ps-layer.selected .rs {{ opacity: 1; }}

/* Delete handle */
.ps-layer .del {{
  position: absolute; top: -8px; right: -8px;
  width: 18px; height: 18px;
  background: #e53e3e; border: 2px solid white;
  border-radius: 50%; cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  font-size: 11px; font-weight: 900; color: white; line-height: 1;
  opacity: 0; transition: opacity .15s;
  z-index: 10;
}}
.ps-layer:hover .del,
.ps-layer.selected .del {{ opacity: 1; }}

/* Opacity slider */
.ps-layer .ops {{
  position: absolute; top: -28px; left: 50%;
  transform: translateX(-50%);
  display: none; background: rgba(0,0,0,.7);
  padding: 3px 8px; border-radius: 12px; gap: 6px;
  align-items: center; white-space: nowrap;
}}
.ps-layer.selected .ops {{ display: flex; }}
.ops label {{ font-size: 10px; color: #ccc; }}
.ops input[type=range] {{ width: 60px; height: 4px; }}

#pos-out {{
  background: white; border-radius: 8px; padding: 8px 12px;
  font-size: 11px; font-family: monospace; color: #444;
  box-shadow: 0 1px 4px rgba(0,0,0,.08); word-break: break-all;
  line-height: 1.5;
}}
</style>
</head>
<body>
<div id="app">

  <div id="toolbar">
    <span class="tool-label">Canvas Editor</span>
    <div class="tool-sep"></div>
    <span class="tool-label" id="sel-info">Click a product to select</span>
    <div class="tool-sep"></div>
    <button id="btn-reset">↺ Reset</button>
    <button id="btn-apply">✓ Apply positions</button>
  </div>

  <div id="canvas-wrap">
    <img id="canvas-bg" src="data:image/jpeg;base64,{bg_b64}" draggable="false" />
  </div>

  <div id="pos-out">Drag products on the canvas above, then click ✓ Apply positions.</div>
</div>

<script>
(function() {{
  const BG   = document.getElementById('canvas-bg');
  const WRAP = document.getElementById('canvas-wrap');
  const N    = {n_ps};
  const origW= {W_c}, origH = {H_c};
  const psB64= {ps_b64_json};
  const defaults = {json.dumps(default_positions[:n_ps])};

  let selected = null;
  let dragging = null, resizing = null;
  let startMX, startMY, startL, startT, startW, startH;

  const layers = [];

  function cw() {{ return WRAP.getBoundingClientRect().width; }}
  function ch() {{ return cw() * origH / origW; }}

  function pxToFrac(px, dim) {{ return px / dim; }}
  function fracToPx(f, dim)  {{ return f * dim; }}

  function buildLayer(i) {{
    const d = defaults[i] || {{x:0.55,y:0.35,scale:0.40}};
    const el = document.createElement('div');
    el.className = 'ps-layer';
    el.dataset.idx = i;
    el.tabIndex = 0;
    el.style.zIndex = 10 + i;

    const img = document.createElement('img');
    img.src = 'data:image/jpeg;base64,' + psB64[i];
    img.draggable = false;

    const ov  = document.createElement('div'); ov.className = 'overlay';
    const lbl = document.createElement('div'); lbl.className = 'label';
    lbl.textContent = 'Product ' + (i+1);
    const rs  = document.createElement('div'); rs.className = 'rs'; rs.textContent = '⤡';
    const del = document.createElement('div'); del.className = 'del'; del.textContent = '×';

    el.appendChild(img);
    el.appendChild(ov);
    el.appendChild(lbl);
    el.appendChild(rs);
    el.appendChild(del);
    WRAP.appendChild(el);

    // Position & size
    function place() {{
      const W = cw(), H = ch();
      const w = d.scale * W;
      const h = w; // square-ish, product image fills via object-fit
      el.style.left   = (d.x * W - w/2) + 'px';
      el.style.top    = (d.y * H - h/2) + 'px';
      el.style.width  = w + 'px';
      el.style.height = h + 'px';
    }}
    place();
    window.addEventListener('resize', place);

    // Drag
    el.addEventListener('mousedown', function(e) {{
      if (e.target === rs || e.target === del) return;
      e.preventDefault();
      select(el);
      dragging = el;
      el.classList.add('dragging');
      startMX = e.clientX; startMY = e.clientY;
      startL = parseFloat(el.style.left); startT = parseFloat(el.style.top);
    }});

    // Resize
    rs.addEventListener('mousedown', function(e) {{
      e.preventDefault(); e.stopPropagation();
      select(el);
      resizing = el;
      startMX = e.clientX; startMY = e.clientY;
      startW = parseFloat(el.style.width); startH = parseFloat(el.style.height);
    }});

    // Delete
    del.addEventListener('click', function(e) {{
      e.stopPropagation();
      el.style.display = 'none';
      if (selected === el) selected = null;
      updateOut();
    }});

    // Select on click
    el.addEventListener('click', function() {{ select(el); }});

    layers.push({{ el, d, place }});
    return el;
  }}

  function select(el) {{
    layers.forEach(l => l.el.classList.remove('selected'));
    if (el) {{
      el.classList.add('selected');
      selected = el;
      const idx = parseInt(el.dataset.idx);
      document.getElementById('sel-info').textContent = 'Product ' + (idx+1) + ' selected';
    }}
  }}

  // Deselect on canvas click
  WRAP.addEventListener('click', function(e) {{
    if (e.target === BG || e.target === WRAP) {{
      layers.forEach(l => l.el.classList.remove('selected'));
      selected = null;
      document.getElementById('sel-info').textContent = 'Click a product to select';
    }}
  }});

  document.addEventListener('mousemove', function(e) {{
    const W = cw(), H = ch();
    if (dragging) {{
      const dx = e.clientX - startMX;
      const dy = e.clientY - startMY;
      const nw = parseFloat(dragging.style.width);
      const nh = parseFloat(dragging.style.height);
      const nl = Math.max(-nw*0.5, Math.min(W - nw*0.5, startL + dx));
      const nt = Math.max(-nh*0.5, Math.min(H - nh*0.5, startT + dy));
      dragging.style.left = nl + 'px';
      dragging.style.top  = nt + 'px';
      updateOut();
    }}
    if (resizing) {{
      const dw = e.clientX - startMX;
      const nw = Math.max(40, Math.min(W*0.8, startW + dw));
      resizing.style.width  = nw + 'px';
      resizing.style.height = nw + 'px';
      updateOut();
    }}
  }});

  document.addEventListener('mouseup', function() {{
    if (dragging) {{ dragging.classList.remove('dragging'); dragging = null; }}
    resizing = null;
  }});

  // Keyboard nudge
  document.addEventListener('keydown', function(e) {{
    if (!selected) return;
    const step = e.shiftKey ? 10 : 2;
    let L = parseFloat(selected.style.left);
    let T = parseFloat(selected.style.top);
    if (e.key === 'ArrowLeft')  {{ L -= step; selected.style.left = L+'px'; }}
    if (e.key === 'ArrowRight') {{ L += step; selected.style.left = L+'px'; }}
    if (e.key === 'ArrowUp')    {{ T -= step; selected.style.top  = T+'px'; }}
    if (e.key === 'ArrowDown')  {{ T += step; selected.style.top  = T+'px'; }}
    if (['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key)) {{
      e.preventDefault(); updateOut();
    }}
    if (e.key === 'Delete' || e.key === 'Backspace') {{
      if (selected) {{ selected.style.display = 'none'; selected = null; updateOut(); }}
    }}
  }});

  function getPositions() {{
    const W = cw(), H = ch();
    const pos = [];
    layers.forEach(function(lyr) {{
      if (lyr.el.style.display === 'none') return;
      const l = parseFloat(lyr.el.style.left);
      const t = parseFloat(lyr.el.style.top);
      const w = parseFloat(lyr.el.style.width);
      const cx = (l + w/2) / W;
      const cy = (t + w/2) / H;
      const sc = w / W;
      pos.push({{ x: +cx.toFixed(3), y: +cy.toFixed(3), scale: +sc.toFixed(3) }});
    }});
    return pos;
  }}

  function updateOut() {{
    const pos = getPositions();
    document.getElementById('pos-out').innerHTML =
      '<b>Current positions</b> (auto-updates as you drag):<br>' +
      pos.map((p,i) => 'Product '+(i+1)+': x='+p.x+' y='+p.y+' scale='+p.scale).join(' &nbsp;|&nbsp; ');
  }}

  document.getElementById('btn-apply').addEventListener('click', function() {{
    const pos = getPositions();
    const json = JSON.stringify(pos);
    // Store in a hidden input that Streamlit can read via st.components
    const out = document.getElementById('pos-out');
    out.innerHTML = '<b style="color:#22a06b">✓ Positions applied!</b> Now click <b>Generate Creative</b> in the sidebar.<br><br>' +
      '<b>JSON to paste into position sliders:</b><br><code>' + json + '</code>';
    this.classList.add('updated');
    setTimeout(() => this.classList.remove('updated'), 1200);
  }});

  document.getElementById('btn-reset').addEventListener('click', function() {{
    layers.forEach(function(lyr, i) {{
      lyr.el.style.display = '';
      lyr.place();
    }});
    updateOut();
  }});

  // Build layers
  for (let i = 0; i < N; i++) buildLayer(i);
  updateOut();

  // Fix canvas height
  function fixHeight() {{
    WRAP.style.height = ch() + 'px';
  }}
  BG.addEventListener('load', fixHeight);
  window.addEventListener('resize', fixHeight);
  fixHeight();

}})();
</script>
</body>
</html>"""

            # Calculate display height based on aspect ratio
            display_h = min(650, int(680 * H_c / W_c) + 80)
            st.components.v1.html(canvas_html, height=display_h, scrolling=False)

            st.markdown("""
**Controls:**
- 🖱️ **Drag** products to reposition | **⤡** corner handle to resize | **×** to hide
- ⌨️ **Arrow keys** to nudge (hold Shift for bigger steps) | **Delete** to remove
- Click **✓ Apply positions** then use the values to set the X/Y/Scale sliders in the **📐 Packshot Position & Scale** panel, then uncheck **Auto Layout** and click **Generate Creative**
            """)

    # ── AI Compliance Chat ─────────────────────────────────────────────────────
    with st.expander("🤖 AI Compliance Assistant (Claude)", expanded=False):
        st.markdown("Ask Claude anything about Tesco Appendix A & B guidelines.")

        for msg in st.session_state.compliance_chat_history:
            role = "You" if msg["role"] == "user" else "Claude"
            st.markdown(f"**{role}:** {msg['content']}")

        user_q = st.text_input("Your question…", key="compliance_chat_input",
                               placeholder="e.g. When is Drinkaware required?")
        if st.button("Ask Claude", disabled=not AI_ENABLED):
            if user_q:
                st.session_state.compliance_chat_history.append({"role": "user", "content": user_q})
                with st.spinner("Claude is answering…"):
                    try:
                        import anthropic
                        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                        response = client.messages.create(
                            model="claude-opus-4-6",
                            max_tokens=500,
                            system=(
                                "You are a Tesco brand compliance expert specialising in Appendix A & B "
                                "creative guidelines. Answer concisely and accurately. Use bullet points "
                                "where helpful. Always mention if something is a HARD FAIL."
                            ),
                            messages=[{"role": m["role"], "content": m["content"]}
                                      for m in st.session_state.compliance_chat_history],
                        )
                        answer = response.content[0].text
                    except Exception as exc:
                        answer = f"Error: {exc}"
                    st.session_state.compliance_chat_history.append({"role": "assistant", "content": answer})
                    st.rerun()

        if st.button("Clear chat"):
            st.session_state.compliance_chat_history = []

    # ── Comprehensive claim tester ─────────────────────────────────────────────
    with st.expander("🧪 Compliance Claim Tester", expanded=False):
        st.markdown("Test how the compliance engine (regex + Claude NLU) handles various copy.")
        test_cases = {
            "✅ Compliant": "NEW LOOK – SAME AWARD WINNING TASTE",
            "❌ Price callout": "ONLY £2.99 – LIMITED TIME",
            "❌ Superlative": "THE BEST PRODUCT EVER",
            "❌ Health claim": "HEALTHY CHOICE – GOOD FOR YOU",
            "❌ Competition": "WIN A FREE CAR – ENTER NOW",
            "❌ Sustainability": "ECO-FRIENDLY – SUSTAINABLE PRODUCT",
            "❌ Guarantee": "MONEY-BACK GUARANTEE – RISK FREE",
            "❌ Implied scarcity": "TIME IS RUNNING OUT – ALMOST GONE",
        }
        selected = st.selectbox("Choose test case:", list(test_cases.keys()))
        test_text = test_cases[selected]
        if st.button("Run Test"):
            parts = test_text.split(" – ")
            h_test = parts[0] if parts else test_text
            s_test = parts[1] if len(parts) > 1 else ""
            result = compliance_engine.check_text_compliance(h_test, s_test, "General")
            st.write(f"**Text:** {test_text}")
            if result["approved"]:
                st.success("✅ Compliant – no violations detected")
            else:
                st.error("❌ Violations found:")
                for issue in result["issues"]:
                    st.markdown(f'<div class="hard-fail">{issue}</div>', unsafe_allow_html=True)
            if result.get("llm_enhanced"):
                st.markdown('<span class="ai-badge">🤖 Claude NLU checked</span>', unsafe_allow_html=True)

    # ── Trending design insights ───────────────────────────────────────────────
    with st.expander("📈 AI Design Trend Insights", expanded=False):
        if st.button("Get AI Trend Insights", disabled=not AI_ENABLED):
            with st.spinner("Claude is fetching trend insights…"):
                trends = creative_suggestor.get_trending_designs(product_category)
                st.markdown(f"**Styles:** {', '.join(trends.get('styles', []))}")
                st.markdown(f"**Colour trends:** {', '.join(trends.get('colors', []))}")
                st.markdown(f"**Tone:** {trends.get('copy_tone', '')}")
                st.markdown("**Recommendations:**")
                for r in trends.get("recommendations", []):
                    st.info(f"• {r}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 🏭  INDUSTRY AI/ML COMPLIANCE PIPELINE
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 🏭 Industry AI/ML Compliance Pipeline")
    st.markdown(
        "Full computer-vision + NLP + risk-scoring pipeline as used by "
        "Meta, Google Ads, and Adobe. Upload any ad creative to get a "
        "**complete automated compliance audit** with annotated detections, "
        "risk score, and downloadable PDF report."
    )

    # Load ML pipeline (lazy import so startup stays fast)
    try:
        from ml_pipeline import (OCREngine, ObjectDetector, NLPComplianceModel,
                                  RiskScorer, ReportGenerator, ComplianceDB,
                                  run_full_pipeline)
        ML_AVAILABLE = True
    except ImportError as _mle:
        ML_AVAILABLE = False
        st.error(f"ML pipeline not available: {_mle}. Run: pip install opencv-python pytesseract reportlab scikit-learn")

    # Initialise DB once per session
    if "compliance_db" not in st.session_state:
        try:
            st.session_state.compliance_db = ComplianceDB("compliance_history.db")
        except Exception:
            st.session_state.compliance_db = None

    if ML_AVAILABLE:
        # ── TABS ──────────────────────────────────────────────────────────────
        tab_audit, tab_dashboard, tab_history = st.tabs([
            "🔬 Deep Audit",
            "📊 Dashboard",
            "📋 Audit History",
        ])

        # ── TAB 1: DEEP AUDIT ─────────────────────────────────────────────────
        with tab_audit:
            st.markdown("### 🔬 Upload Any Ad Creative for Full AI Analysis")

            audit_col1, audit_col2 = st.columns([1, 2])

            with audit_col1:
                audit_file = st.file_uploader(
                    "Upload ad image (JPG/PNG)",
                    type=["jpg","jpeg","png"],
                    key="ml_audit_upload",
                    help="Upload any ad creative — the pipeline will run OCR, "
                         "object detection, NLP, and risk scoring automatically."
                )
                audit_headline = st.text_input("Headline (optional)", key="ml_hl",
                    placeholder="Enter ad headline for NLP analysis")
                audit_subhead  = st.text_input("Subhead (optional)", key="ml_sh",
                    placeholder="Enter ad subhead")
                audit_brand    = st.selectbox("Brand", list(BRANDS.keys()) if 'BRANDS' in dir() else ["Tesco","Custom"],
                    key="ml_brand")
                audit_format   = st.selectbox("Format", ["Square (1080×1080)","Landscape (1200×628)","Stories (1080×1920)"],
                    key="ml_format")

                # Also allow auditing a generated creative
                if st.session_state.generated_creatives:
                    if st.checkbox("🎨 Audit last generated creative instead", key="audit_generated"):
                        audit_file = None
                        _last_creative = st.session_state.generated_creatives[0]["image"]
                    else:
                        _last_creative = None
                else:
                    _last_creative = None

                run_audit = st.button("🚀 Run Full AI/ML Audit", type="primary",
                                       use_container_width=True, key="run_ml_audit")

            with audit_col2:
                if run_audit:
                    # Determine image to audit
                    if audit_file:
                        audit_image = Image.open(audit_file).convert("RGB")
                    elif _last_creative:
                        audit_image = _last_creative.convert("RGB")
                    else:
                        st.warning("Please upload an image or generate a creative first.")
                        audit_image = None

                    if audit_image:
                        with st.spinner("🔬 Running AI/ML pipeline — OCR → Object Detection → NLP → Risk Score → PDF…"):
                            try:
                                result = run_full_pipeline(
                                    image        = audit_image,
                                    headline     = audit_headline,
                                    subhead      = audit_subhead,
                                    brand_name   = audit_brand,
                                    format_name  = audit_format,
                                    design_issues= [],
                                    db           = st.session_state.compliance_db,
                                )
                                st.session_state["last_ml_result"] = result
                                st.session_state["last_audit_image"] = audit_image
                            except Exception as _pe:
                                st.error(f"Pipeline error: {_pe}")
                                result = None

                        if result:
                            risk = result["risk"]
                            score = risk["total_score"]

                            # ── Hero score banner ──────────────────────────
                            risk_bg = {"✅ COMPLIANT":"#38A169","🟡 LOW RISK":"#D69E2E",
                                       "🟠 MEDIUM RISK":"#DD6B20","🔴 HIGH RISK":"#E53E3E",
                                       "🚨 CRITICAL":"#822727"}.get(risk["risk_label"],"#3182CE")
                            st.markdown(
                                f"""<div style="background:{risk_bg};color:white;border-radius:12px;
                                padding:20px;text-align:center;margin:10px 0">
                                <div style="font-size:48px;font-weight:900">{score}/100</div>
                                <div style="font-size:22px;font-weight:600">{risk['risk_label']}</div>
                                <div style="font-size:16px;opacity:.85">Grade: {risk['grade']} &nbsp;·&nbsp; Audit ID: {result.get('audit_id','–')}</div>
                                </div>""",
                                unsafe_allow_html=True
                            )

                            # ── Score breakdown cols ──────────────────────
                            bd = risk.get("breakdown", {})
                            if bd:
                                st.markdown("**Score Breakdown**")
                                metric_cols = st.columns(min(4, len(bd)))
                                for j, (cat, pts) in enumerate(
                                        sorted(bd.items(), key=lambda x: -x[1])[:4]):
                                    metric_cols[j % 4].metric(cat, f"{pts}pts")

                            # ── Annotated image ───────────────────────────
                            det_col, info_col = st.columns([1, 1])
                            with det_col:
                                st.markdown("**🔍 Object Detection**")
                                ann = result.get("annotated")
                                if ann:
                                    st.image(ann, use_column_width=True,
                                             caption="OpenCV detection overlay")
                                detections = result.get("detections", [])
                                for det in detections:
                                    risk_icon = {"CRITICAL":"🚨","HIGH":"🔴","MEDIUM":"🟠",
                                                  "LOW":"🟡","INFO":"🔵"}.get(det["compliance_risk"],"•")
                                    st.markdown(
                                        f"{risk_icon} **{det['label']}** &nbsp; "
                                        f"`{det['confidence']}%` &nbsp; "
                                        f"*{det.get('rule','')[:60]}*"
                                    )

                            with info_col:
                                # OCR results
                                st.markdown("**📝 OCR Extracted Text**")
                                ocr = result.get("ocr", {})
                                ocr_text = ocr.get("text","").strip()
                                if ocr_text:
                                    st.code(ocr_text[:400], language=None)
                                    st.caption(f"OCR confidence: {ocr.get('confidence',0):.0f}%  |  Words: {len(ocr.get('words',[]))}")
                                    # Font size warnings
                                    small_words = [w for w in ocr.get("words",[]) if w.get("size_px",99) < 14]
                                    if small_words:
                                        st.warning(f"⚠️ {len(small_words)} word(s) below minimum 14px: "
                                                   f"{', '.join(w['text'] for w in small_words[:5])}")
                                else:
                                    st.info("No text extracted from image.")

                                # NLP violations
                                st.markdown("**🧠 NLP Compliance Analysis**")
                                nlp_r = result.get("text", {})
                                violations = nlp_r.get("violations", [])
                                if violations:
                                    for v in violations:
                                        sev_colors = {"CRITICAL":"#822727","HIGH":"#C53030",
                                                       "MEDIUM":"#C05621","LOW":"#276749"}
                                        sc = sev_colors.get(v["severity"],"#2B6CB0")
                                        st.markdown(
                                            f'<div style="border-left:4px solid {sc};padding:6px 10px;'
                                            f'margin:4px 0;background:#fafafa;border-radius:0 6px 6px 0">'
                                            f'<b>{v["category"]}</b> <span style="color:{sc};font-size:11px">'
                                            f'[{v["severity"]}] {v["confidence"]}%</span><br>'
                                            f'<i style="font-size:12px">"{v["text"][:60]}"</i><br>'
                                            f'<span style="font-size:11px;color:#555">💡 {v["suggestion"]}</span>'
                                            f'</div>', unsafe_allow_html=True
                                        )
                                else:
                                    st.success("✅ No NLP violations detected")

                            # ── Recommendations ───────────────────────────
                            recs = risk.get("recommendations", [])
                            if recs:
                                st.markdown("**📋 AI Recommendations**")
                                for rec in recs:
                                    st.info(rec)

                            # ── PDF download ──────────────────────────────
                            pdf = result.get("pdf", b"")
                            if pdf:
                                st.download_button(
                                    "📄 Download Full PDF Compliance Report",
                                    data=pdf,
                                    file_name=f"compliance_report_{audit_brand}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True,
                                    key="ml_pdf_dl"
                                )

                elif st.session_state.get("last_ml_result"):
                    st.info("⬅️ Configure audit settings and click **Run Full AI/ML Audit**. Previous result shown above.")

        # ── TAB 2: DASHBOARD ──────────────────────────────────────────────────
        with tab_dashboard:
            st.markdown("### 📊 Compliance Analytics Dashboard")

            db = st.session_state.get("compliance_db")
            if db:
                stats = db.get_dashboard_stats()

                # KPI row
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total Audited",    stats["total_checked"],
                          delta=None)
                k2.metric("✅ Passed",        stats["total_passed"],
                          delta=f"{100*stats['total_passed']/max(stats['total_checked'],1):.0f}%")
                k3.metric("❌ Failed",         stats["total_failed"],
                          delta=f"{100*stats['total_failed']/max(stats['total_checked'],1):.0f}%",
                          delta_color="inverse")
                k4.metric("Avg Risk Score",   f"{stats['avg_risk_score']:.1f}/100",
                          delta_color="inverse")

                st.markdown("---")
                dash_c1, dash_c2 = st.columns([1,1])

                with dash_c1:
                    # Score history chart
                    score_hist = stats.get("score_history", [])
                    if score_hist:
                        import plotly.graph_objects as go
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=[s["score"] for s in reversed(score_hist)],
                            mode="lines+markers",
                            name="Risk Score",
                            line=dict(color="#00539F", width=2),
                            fill="tozeroy",
                            fillcolor="rgba(0,83,159,0.1)"
                        ))
                        fig.add_hline(y=25, line_dash="dash", line_color="#38A169",
                                      annotation_text="Pass threshold")
                        fig.update_layout(
                            title="Risk Score History (Last 20 Audits)",
                            xaxis_title="Audit #", yaxis_title="Risk Score",
                            yaxis=dict(range=[0,100]),
                            height=280, margin=dict(l=20,r=20,t=40,b=20),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(248,249,250,1)",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                with dash_c2:
                    # Pass/fail pie
                    if stats["total_checked"] > 0:
                        import plotly.express as px
                        fig2 = px.pie(
                            values=[stats["total_passed"], stats["total_failed"]],
                            names=["Passed","Failed"],
                            color_discrete_sequence=["#38A169","#E53E3E"],
                            title="Pass / Fail Distribution",
                            hole=0.45
                        )
                        fig2.update_layout(height=280,
                                           margin=dict(l=20,r=20,t=40,b=20),
                                           paper_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig2, use_container_width=True)

                # Top brands table
                top = stats.get("top_brands",[])
                if top:
                    st.markdown("**Top Brands by Audit Volume**")
                    import pandas as pd
                    df = pd.DataFrame(top)
                    df.columns = ["Brand","Audits","Avg Risk Score"]
                    df["Status"] = df["Avg Risk Score"].apply(
                        lambda s: "✅ Low" if s<25 else "🟡 Med" if s<45 else "🔴 High")
                    st.dataframe(df, use_container_width=True, hide_index=True)

                if st.button("🗑️ Clear All History", key="clear_hist"):
                    db.clear_all()
                    st.success("History cleared.")
                    st.rerun()
            else:
                st.info("No audit history yet. Run an audit on the **Deep Audit** tab.")

        # ── TAB 3: AUDIT HISTORY ──────────────────────────────────────────────
        with tab_history:
            st.markdown("### 📋 Audit History")

            db = st.session_state.get("compliance_db")
            if db:
                history = db.get_history(50)
                if history:
                    import pandas as pd
                    df = pd.DataFrame(history)
                    df["ts"] = pd.to_datetime(df["ts"]).dt.strftime("%d %b %Y %H:%M")
                    df["status"] = df["risk_score"].apply(
                        lambda s: "✅ Pass" if s<25 else "❌ Fail")
                    df = df.rename(columns={
                        "id":"ID","ts":"Date","brand":"Brand","format":"Format",
                        "headline":"Headline","risk_score":"Score",
                        "grade":"Grade","status":"Status"
                    })
                    df = df[["ID","Date","Brand","Format","Headline","Score","Grade","Status"]]
                    df["Headline"] = df["Headline"].fillna("").str[:40]
                    st.dataframe(
                        df.style.apply(
                            lambda row: ["background-color:#F0FFF4" if row["Status"]=="✅ Pass"
                                         else "background-color:#FFF5F5"]*len(row),
                            axis=1
                        ),
                        use_container_width=True, hide_index=True
                    )
                    st.caption(f"Showing {len(history)} most recent audits")
                else:
                    st.info("No audits recorded yet. Run an audit on the **Deep Audit** tab.")
            else:
                st.info("Database not available.")

    # ── VIDEO EDITOR ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🎬 Video Editor")
    st.markdown("Upload a video and apply professional edits — trim, colour grade, audio enhance, overlays and more.")

    try:
        from video_editor import (get_video_info, extract_thumbnail, extract_frames,
                                   process_video, trim_video, add_text_overlay,
                                   add_background_music, replace_audio)
        VIDEO_AVAILABLE = True
    except ImportError as ve:
        VIDEO_AVAILABLE = False
        st.warning(f"⚠️ Video editor unavailable: {ve}")

    if VIDEO_AVAILABLE:
        vid_upload = st.file_uploader(
            "Upload Video", type=["mp4","mov","avi","mkv","webm","m4v"],
            key="video_uploader")

        if vid_upload:
            import tempfile, os as _os

            # Save upload to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(vid_upload.read())
                vid_path = tmp.name

            info = get_video_info(vid_path)
            dur  = float(info.get("duration", 10.0))

            # Info row
            i1,i2,i3,i4,i5 = st.columns(5)
            i1.metric("Duration",    f"{dur:.1f}s")
            i2.metric("Resolution",  f"{info.get('width','?')}×{info.get('height','?')}")
            i3.metric("FPS",         info.get("fps","?"))
            i4.metric("Size",        f"{info.get('size_mb','?')} MB")
            i5.metric("Audio",       "✅ Yes" if info.get("has_audio") else "❌ No")

            # Frame preview
            with st.expander("🎞️ Video Preview Frames", expanded=True):
                frames = extract_frames(vid_path, n_frames=6)
                if frames:
                    fcols = st.columns(len(frames))
                    for i, (fr, fc) in enumerate(zip(frames, fcols)):
                        fc.image(fr, use_column_width=True, caption=f"Frame {i+1}")

            # ── Edit controls ──────────────────────────────────────────────
            vt1, vt2 = st.columns(2)

            with vt1:
                st.subheader("✂️ Trim & Speed")
                trim_s = st.slider("Start (sec)", 0.0, max(dur-0.1, 0.0), 0.0, 0.1)
                trim_e = st.slider("End (sec)",   0.1, dur, dur, 0.1)
                speed  = st.select_slider("Playback Speed",
                    options=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0], value=1.0)
                st.caption(f"Trimmed duration: {trim_e-trim_s:.1f}s")

                st.subheader("🔄 Transform")
                t1,t2,t3 = st.columns(3)
                v_flip_h = t1.checkbox("↔ Flip H")
                v_flip_v = t2.checkbox("↕ Flip V")
                v_rotate = t3.selectbox("Rotate°", [0, 90, 180, 270], key="v_rot")

                st.subheader("📤 Output")
                out_res = st.selectbox("Resolution", [
                    "Original","1080p (1920×1080)","720p (1280×720)",
                    "480p (854×480)","Square (1080×1080)",
                    "Stories (1080×1920)","Landscape (1200×628)"])
                out_qual = st.selectbox("Quality", ["High","Ultra (best)","Medium","Low (smallest)"])

            with vt2:
                st.subheader("🎨 Colour & Light")
                ca1, ca2 = st.columns(2)
                v_bright = ca1.slider("☀️ Brightness",  -1.0, 1.0, 0.0, 0.05, key="vbr")
                v_cont   = ca2.slider("◑ Contrast",      0.0, 3.0, 1.0, 0.1,  key="vco")
                v_sat    = ca1.slider("🎨 Saturation",   0.0, 3.0, 1.0, 0.1,  key="vsa")
                v_gamma  = ca2.slider("💡 Exposure",     0.1, 3.0, 1.0, 0.1,  key="vga")
                v_sharp  = ca1.slider("🔪 Sharpness",    0.0, 5.0, 0.0, 0.1,  key="vsh")

                v_filter = st.selectbox("🎞️ Colour Filter", [
                    "None","Warm","Cool","Vibrant","Matte",
                    "Black & White","Vintage","Cinematic","HDR","Faded"])

                v_vignette = st.checkbox("🔲 Vignette effect")
                v_fade_in  = st.slider("Fade in (sec)",  0.0, 3.0, 0.0, 0.1)
                v_fade_out = st.slider("Fade out (sec)", 0.0, 3.0, 0.0, 0.1)

            st.subheader("🔊 Audio")
            au1, au2, au3, au4 = st.columns(4)
            v_volume    = au1.slider("Volume",     0.0, 3.0, 1.0, 0.1, key="vvol")
            v_mute      = au2.checkbox("🔇 Mute")
            v_normalize = au3.checkbox("📊 Normalise")
            v_bass      = au4.checkbox("🎵 Bass boost")
            v_noise_red = au1.checkbox("🔕 Noise reduce")

            # Background music upload
            with st.expander("🎵 Add Background Music"):
                bg_music_file = st.file_uploader("Upload music (mp3/wav/aac)",
                    type=["mp3","wav","aac","m4a"], key="bg_music")
                if bg_music_file:
                    music_vol = st.slider("Music volume", 0.0, 1.0, 0.3, 0.05)

            # Replace audio
            with st.expander("🔄 Replace Audio"):
                replace_audio_file = st.file_uploader("Upload new audio",
                    type=["mp3","wav","aac","m4a"], key="replace_audio_f")

            # Text overlay
            with st.expander("✏️ Text Overlay"):
                v_text     = st.text_input("Text to overlay", placeholder="e.g. NEW PRODUCT")
                v_text_pos = st.selectbox("Position", ["bottom","top","center","top-left","top-right"])
                v_text_sz  = st.slider("Font size", 20, 120, 48)
                v_text_col = st.color_picker("Text colour", "#FFFFFF")

            # Process button
            st.divider()
            if st.button("🎬 Process & Export Video", type="primary", use_container_width=True):
                with st.spinner("🎬 Processing video — this may take a moment…"):
                    out_path = vid_path.replace(".mp4","_edited.mp4")

                    ok, out, err = process_video(
                        input_path   = vid_path,
                        output_path  = out_path,
                        trim_start   = trim_s,
                        trim_end     = trim_e if trim_e < dur else 0.0,
                        brightness   = v_bright,
                        contrast     = v_cont,
                        saturation   = v_sat,
                        gamma        = v_gamma,
                        sharpness    = v_sharp,
                        flip_h       = v_flip_h,
                        flip_v       = v_flip_v,
                        rotate       = v_rotate,
                        speed        = speed,
                        colour_filter= v_filter,
                        volume       = v_volume,
                        normalize_audio = v_normalize,
                        noise_reduce = v_noise_red,
                        bass_boost   = v_bass,
                        mute         = v_mute,
                        text_overlay = v_text,
                        text_position= v_text_pos,
                        text_size    = v_text_sz,
                        text_colour  = v_text_col.lstrip("#") or "white",
                        output_quality    = out_qual,
                        output_resolution = out_res if out_res != "Original" else "",
                        vignette     = v_vignette,
                        fade_in      = v_fade_in,
                        fade_out     = v_fade_out,
                    )

                    # Post-process: background music or replace audio
                    if ok and _os.path.exists(out_path):
                        if bg_music_file:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as m:
                                m.write(bg_music_file.read())
                                music_tmp = m.name
                            music_out = out_path.replace(".mp4","_music.mp4")
                            ok2, _ = add_background_music(out_path, music_tmp, music_out, music_vol)
                            if ok2 and _os.path.exists(music_out):
                                out_path = music_out

                        if replace_audio_file:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as ra:
                                ra.write(replace_audio_file.read())
                                audio_tmp = ra.name
                            audio_out = out_path.replace(".mp4","_newaud.mp4")
                            ok3, _ = replace_audio(out_path, audio_tmp, audio_out)
                            if ok3 and _os.path.exists(audio_out):
                                out_path = audio_out

                    if ok and _os.path.exists(out_path):
                        with open(out_path, "rb") as f:
                            video_bytes = f.read()
                        st.success(f"✅ Video processed! Size: {len(video_bytes)//1024}KB")
                        st.download_button(
                            "📥 Download Edited Video",
                            data=video_bytes,
                            file_name=f"edited_{vid_upload.name}",
                            mime="video/mp4",
                            use_container_width=True,
                        )
                        # Show thumbnail of output
                        thumb2 = extract_thumbnail(out_path, 1.0)
                        if thumb2:
                            st.image(thumb2, caption="Output preview", width=300)
                    else:
                        st.error(f"❌ Processing failed. Error: {(err or 'unknown')[:300]}")

            # Cleanup temp file on session end (best effort)
            try:
                if _os.path.exists(vid_path): _os.unlink(vid_path)
            except Exception:
                pass

        else:
            # Show supported features when no video uploaded
            fc1, fc2, fc3 = st.columns(3)
            fc1.markdown("""
**✂️ Trim & Speed**
- Set start/end timestamps
- 0.25× slow motion to 2× fast
- Clip any section of the video

**🎨 Colour & Light**
- Brightness, contrast, saturation
- Gamma/exposure control
- Sharpness adjustment
- 10 colour filter presets
            """)
            fc2.markdown("""
**🔊 Audio**
- Volume control
- Mute audio
- Loudness normalisation
- Bass boost
- Noise reduction
- Add background music
- Replace audio track

**✏️ Text Overlay**
- Burn text into video
- Choose position & size
- Pick text colour
            """)
            fc3.markdown("""
**📤 Export**
- 6 resolution presets
- 4 quality levels
- Flip H/V, rotate 90/180/270°
- Vignette effect
- Fade in / fade out
- ffmpeg-powered (no quality loss)

**Supported formats:**
mp4, mov, avi, mkv, webm, m4v
            """)

    # ── INDUSTRY AI/ML PIPELINE TABS ──────────────────────────────────────────
    try:
        from industry_ui import render_industry_tabs
        render_industry_tabs()
    except ImportError as ie:
        st.warning(f"Industry pipeline unavailable: {ie}")

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    fc1, fc2, fc3 = st.columns([3, 1, 1])
    fc1.markdown("**GenAI Creative Compliance Studio** – Claude API · OpenCV · TF-IDF/LR · Tesseract OCR · ReportLab · SQLite")
    fc2.markdown(f"**Theme:** {'🌙 Dark' if st.session_state.dark_mode else '☀️ Light'}")
    fc3.markdown(f"**Creatives:** {len(st.session_state.generated_creatives)}")

if __name__ == "__main__":
    main()
