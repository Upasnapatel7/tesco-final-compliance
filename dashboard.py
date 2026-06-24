"""
dashboard.py  —  GenAI Creative Compliance Studio
==================================================
streamlit run dashboard.py

One command. All features:
  ✅ 65+ brands across 16 industries
  ✅ Upload up to 5 packshot images
  ✅ Background: brand / solid / gradient / pattern / upload
  ✅ Font family, weight, size, colour
  ✅ Layout presets
  ✅ AI Background removal
  ✅ Image editing (brightness, contrast, saturation, etc.)
  ✅ AI creative generation with textures
  ✅ NLP + OCR + CV compliance check
  ✅ Risk score, PDF report
  ✅ Video editor (trim, colour, audio, text)
  ✅ Audit history + analytics
"""

import streamlit as st
import io, os, json, base64, tempfile, math
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

try:
    from creative_studio import render_studio_tab
    STUDIO_OK = True
except ImportError:
    STUDIO_OK = False
    
# Auth
try:
    from auth import render_auth_page, render_user_badge
    AUTH_OK = True
except ImportError:
    AUTH_OK = False

# AI Creative Assistant
try:
    from ai_creative_assistant import (
        RealTimeViolationChecker, HeadlineSuggestionEngine,
        SubheadSuggestionEngine, CopyImprovementEngine, LayoutColourAdvisor
    )
    ASSISTANT_OK = True
    _checker = RealTimeViolationChecker()
except ImportError:
    ASSISTANT_OK = False
    _checker = None

# Policy Analyser
try:
    from policy_analyser import (
        render_policy_uploader, render_policy_analysis, load_policies
    )
    POLICY_OK = True
except ImportError:
    POLICY_OK = False

# Advanced AI
try:
    from advanced_ai_tab import render_advanced_ai_tab
    ADVANCED_AI_OK = True
except ImportError:
    ADVANCED_AI_OK = False

# Creative Studio
try:
    from creative_studio import render_studio_tab
    STUDIO_OK = True
except ImportError:
    STUDIO_OK = False

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GenAI Creative Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Auth gate ────────────────────────────────────────────────────────────────
# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# Show auth page if not logged in
if not st.session_state["authenticated"]:
    if AUTH_OK:
        render_auth_page()
    else:
        st.warning("⚠️ Auth module not found. Bypassing login.")
        st.session_state["authenticated"] = True
    st.stop()

# ✅ Your actual dashboard starts BELOW this
st.title("Dashboard")

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Sidebar dark theme */
  section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #21262d;
  }
  section[data-testid="stSidebar"] .stMarkdown p,
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] .stRadio label span,
  section[data-testid="stSidebar"] .stCheckbox label span {
    color: #c9d1d9 !important;
  }
  /* Section headers */
  .sh {
    background: linear-gradient(90deg, #1f6feb 0%, #58a6ff 100%);
    color: white !important; font-weight: 700; font-size: 11px;
    text-transform: uppercase; letter-spacing: 1.2px;
    padding: 6px 12px; border-radius: 6px; margin: 16px 0 8px;
  }
  /* Cards */
  .card {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 10px; padding: 20px; margin: 8px 0;
  }
  /* Metric pill */
  .mpill {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-weight: 700; font-size: 13px; margin: 2px;
  }
  .risk-A { background:#238636; color:white; }
  .risk-B { background:#1f6feb; color:white; }
  .risk-C { background:#9e6a03; color:white; }
  .risk-D { background:#da3633; color:white; }
  .risk-F { background:#6e1818; color:white; }
  .risk-crit { background:#ff4444; color:white; }
  .risk-high { background:#ff8c00; color:white; }
  .risk-med  { background:#ffd700; color:#111; }
  .risk-low  { background:#3fb950; color:white; }
  /* Tab styling */
  button[data-baseweb="tab"] {
    font-size: 14px !important; font-weight: 600 !important;
  }
  /* Colour swatch row */
  .swatch-row { display: flex; gap: 4px; margin: 4px 0; flex-wrap: wrap; }
  .swatch {
    width: 24px; height: 24px; border-radius: 4px;
    border: 1px solid #30363d; cursor: pointer;
  }
</style>
""", unsafe_allow_html=True)

def _sh(txt):
    st.markdown(f'<div class="sh">{txt}</div>', unsafe_allow_html=True)

def _card(content_fn):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    content_fn()
    st.markdown('</div>', unsafe_allow_html=True)

# ─── Load all modules (cached) ────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Loading AI/ML models…")
def _load():
    mods = {}
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    except Exception:
        pass
    try:
        from brand_config import (BRANDS, BRAND_CATEGORIES, FONT_MAP,
                                   BG_PATTERNS, BG_GRADIENTS, COLOR_PALETTES,
                                   LAYOUT_PRESETS)
        mods.update(BRANDS=BRANDS, BRAND_CATEGORIES=BRAND_CATEGORIES,
                    FONT_MAP=FONT_MAP, BG_PATTERNS=BG_PATTERNS,
                    BG_GRADIENTS=BG_GRADIENTS, COLOR_PALETTES=COLOR_PALETTES,
                    LAYOUT_PRESETS=LAYOUT_PRESETS)
    except Exception as e:
        mods.update(BRANDS={}, BRAND_CATEGORIES={}, FONT_MAP={},
                    BG_PATTERNS={}, BG_GRADIENTS={}, COLOR_PALETTES={},
                    LAYOUT_PRESETS={})
    try:
        from creative_renderer import render_creative, apply_image_edits, _smart_bg_remove
        mods["render_creative"]   = render_creative
        mods["apply_edits"]       = apply_image_edits
        mods["smart_bg_remove"]   = _smart_bg_remove
        mods["RENDERER"] = True
    except Exception as e:
        mods["RENDERER"] = False; mods["renderer_err"] = str(e)
    try:
        from ml_pipeline import run_full_pipeline, _db, _nlp
        mods["run_pipeline"] = run_full_pipeline
        mods["db"]  = _db
        mods["nlp"] = _nlp
        mods["ML"]  = True
    except Exception as e:
        mods["ML"] = False; mods["ml_err"] = str(e)
    try:
        from video_editor import (get_video_info, extract_frames,
                                   extract_thumbnail, process_video,
                                   add_background_music, replace_audio)
        mods["get_info"]    = get_video_info
        mods["get_frames"]  = extract_frames
        mods["get_thumb"]   = extract_thumbnail
        mods["proc_video"]  = process_video
        mods["add_music"]   = add_background_music
        mods["repl_audio"]  = replace_audio
        mods["VIDEO"] = True
    except Exception as e:
        mods["VIDEO"] = False; mods["video_err"] = str(e)
    try:
        from ai_creative_director import generate_ai_variations, AI_AVAILABLE
        mods["gen_variations"] = generate_ai_variations
        mods["AI_API"] = AI_AVAILABLE
        mods["AI_DIRECTOR"] = True
    except Exception as e:
        mods["AI_DIRECTOR"] = False
    return mods

M = _load()

BRANDS           = M.get("BRANDS", {})
BRAND_CATEGORIES = M.get("BRAND_CATEGORIES", {})
FONT_MAP         = M.get("FONT_MAP", {})
BG_PATTERNS      = M.get("BG_PATTERNS", {
    "None":"none","Subtle Dots":"subtle_dots","Diagonal Lines":"diagonal_lines",
    "Corner Circles":"corner_circles","Geometric":"geometric_slab",
    "Soft Noise":"soft_noise","Wave Lines":"wave_lines","Halftone":"halftone",
    "Crosshatch":"crosshatch","Grid":"grid"})
BG_GRADIENTS     = M.get("BG_GRADIENTS", {})
COLOR_PALETTES   = M.get("COLOR_PALETTES", {})
LAYOUTS          = list(M.get("LAYOUT_PRESETS", {}).keys()) or \
                   ["Product Hero","Centered Minimal","Bold Left","Full Bleed","Split Panel"]
FONTS            = list(FONT_MAP.keys()) or ["Poppins (Modern)"]
CATEGORIES       = list(BRAND_CATEGORIES.keys()) or ["Custom"]

# ─── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "packshots": [], "processed": [], "generated": [],
    "bg_image": None, "audit_result": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

def _to_bytes(img, fmt="PNG"):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, fmt, quality=92)
    return buf.getvalue()

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    if AUTH_OK:
        render_user_badge()
    st.markdown("## 🎨 Creative Studio")
    st.markdown("*65+ brands · AI/ML pipeline*")
    st.divider()

    # ── Brand ──────────────────────────────────────────────────────────────────
    _sh("🏢 Brand & Industry")
    cat = st.selectbox("Industry", CATEGORIES, label_visibility="collapsed")
    brands_in_cat = BRAND_CATEGORIES.get(cat, ["Custom Brand"])
    brand = st.selectbox("Brand", brands_in_cat, label_visibility="collapsed")
    bcfg  = BRANDS.get(brand, {
        "primary":"#333","bg_default":"#F5F5F5","text_dark":"#111",
        "text_light":"#FFF","font_family":"Poppins (Modern)","logo_text":"BRAND",
        "logo_shape":"pill","category":cat})

    # Swatch
    pc = bcfg.get("primary","#333")
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;padding:4px 0">'
        f'<div style="width:16px;height:16px;background:{pc};border-radius:3px;border:1px solid #555"></div>'
        f'<span style="font-size:11px;color:#8b949e">{brand} · {bcfg.get("category","")}</span></div>',
        unsafe_allow_html=True)

    if brand == "Custom Brand":
        c1, c2 = st.columns(2)
        cp = c1.color_picker("Primary", "#333333")
        ca = c2.color_picker("Accent",  "#666666")
        cl = st.text_input("Logo text", "BRAND", max_chars=12)
        bcfg = {**bcfg, "primary":cp, "accent":ca, "logo_text":cl,
                "text_light":"#FFF","text_dark":"#111","bg_default":"#F5F5F5"}
        if BRANDS: BRANDS["Custom Brand"].update(bcfg)

    prod_cat = st.selectbox("Product category", [
        "General","Alcohol","Grocery","Electronics","Fashion",
        "Home & Garden","Beauty & Health","Finance","Healthcare",
        "Pharmacy","Automotive","Travel","Food & Beverage"],
        label_visibility="collapsed")
    include_da = (prod_cat == "Alcohol" and
                  st.checkbox("Include Drinkaware", value=True))
    show_logo  = st.checkbox("Show brand logo", value=True)

    # ── Copy ───────────────────────────────────────────────────────────────────
    _sh("📝 Copy + AI Assist")

    # Use suggested values if AI selected one
    _default_hl = st.session_state.pop("suggested_headline", "")
    _default_sh = st.session_state.pop("suggested_subhead",  "")

    headline = st.text_input("Headline",
        value=_default_hl,
        placeholder="e.g. DISCOVER NEW RANGE",
        label_visibility="collapsed")
    subhead  = st.text_input("Subhead",
        value=_default_sh,
        placeholder="e.g. Available in stores",
        label_visibility="collapsed")
    tag_text = st.text_input("Tag line", f"Available at {brand}",
                              label_visibility="collapsed")

    # ── Real-time compliance + AI suggestions ─────────────────────────────────
    if ASSISTANT_OK and headline:
        hl_check = _checker.check(headline)
        if hl_check["is_compliant"]:
            st.markdown(
                '<div style="background:#238636;color:white;padding:5px 10px;'
                'border-radius:5px;font-size:12px;margin:2px 0">✅ Headline looks compliant</div>',
                unsafe_allow_html=True)
        else:
            viols = hl_check["violations"] or hl_check["quick_violations"]
            if viols:
                v = viols[0]
                sev = v.get("severity","HIGH")
                tip = v.get("suggestion", v.get("category","Remove violation language"))
                col = {"CRITICAL":"#6e1818","HIGH":"#da3633","MEDIUM":"#9e6a03","LOW":"#1f6feb"}.get(sev,"#da3633")
                st.markdown(
                    f'<div style="background:{col};color:white;padding:5px 10px;'
                    f'border-radius:5px;font-size:12px;margin:2px 0">'
                    f'⚠️ {sev}: {v.get("category","Violation")} — {tip[:60]}</div>',
                    unsafe_allow_html=True)

    if ASSISTANT_OK and subhead:
        sh_check = _checker.check(subhead)
        if not sh_check["is_compliant"]:
            viols = sh_check["violations"] or sh_check["quick_violations"]
            if viols:
                v = viols[0]
                st.markdown(
                    f'<div style="background:#9e6a03;color:white;padding:4px 10px;'
                    f'border-radius:5px;font-size:11px;margin:2px 0">'
                    f'⚠️ Subhead: {v.get("category","Violation")}</div>',
                    unsafe_allow_html=True)

    # AI suggestions inline
        # AI suggestions inline
    if ASSISTANT_OK and prod_cat:
        with st.expander("🤖 AI Headline Suggestions"):
            _product_for_sugg = headline or "new product"
            if st.button("✨ Get AI Headlines", key="inline_hl_sugg", use_container_width=True):
                with st.spinner("Generating…"):
                    _eng = HeadlineSuggestionEngine()
                    _res = _eng.suggest(brand, prod_cat, _product_for_sugg, headline, n=5)
                st.session_state["_inline_hl"] = _res
            if "_inline_hl" in st.session_state:
                for i, _s in enumerate(st.session_state["_inline_hl"]["suggestions"]):
                    # Define icon based on compliance
                    _icon = "🟢" if _s.get("compliant", False) else "💡"
                    if st.button(
                        f"{_icon} {_s['headline']}",
                        key=f"ihl_{i}_{_s['headline'][:20]}",
                        use_container_width=True
                    ):
                        st.session_state["suggested_headline"] = _s["headline"]
                        st.rerun()

        with st.expander("🤖 AI Subhead Suggestions"):
            if st.button("✨ Get AI Subheads", key="inline_sh_sugg", use_container_width=True):
                with st.spinner("Generating…"):
                    _eng2 = SubheadSuggestionEngine()
                    _res2 = _eng2.suggest(brand, prod_cat, headline or "", headline, n=4)
                st.session_state["_inline_sh"] = _res2
            if "_inline_sh" in st.session_state:
                for _s in st.session_state["_inline_sh"]["suggestions"]:
                    _icon = "🟢" if _s["compliant"] else "🔴"
                    if st.button(
                        f"{_icon} {_s['subhead']}",
                        key=f"ish_{_s['subhead'][:15]}",
                        use_container_width=True
                    ):
                        st.session_state["suggested_subhead"] = _s["subhead"]
                        st.rerun()

        with st.expander("📊 Analyse My Copy"):
            if (headline or subhead) and st.button("Analyse", key="inline_analyse", use_container_width=True):
                _analyser = CopyImprovementEngine()
                _analysis = _analyser.analyse(headline, subhead, brand, prod_cat)
                _gcol = {"A":"#238636","B":"#1f6feb","C":"#9e6a03","D":"#da3633","F":"#6e1818"}.get(_analysis["grade"],"#888")
                st.markdown(
                    f'<div style="background:{_gcol};color:white;padding:6px 12px;'
                    f'border-radius:6px;font-size:13px">'
                    f'Grade <b>{_analysis["grade"]}</b> · Score {_analysis["copy_score"]}/100'
                    f'</div>', unsafe_allow_html=True)
                for _issue in _analysis["issues"][:3]:
                    st.caption(f"⚠️ {_issue['issue']}: {_issue['detail'][:60]}")
                for _strength in _analysis["strengths"][:2]:
                    st.caption(f"✅ {_strength}")

    # ── Badge ──────────────────────────────────────────────────────────────────
    _sh("💰 Price Badge")
    show_badge = st.checkbox("Show price badge")
    blabel = bprice = bsub = ""
    badge_colour = bcfg.get("primary","#333")
    badge_size   = 0.200
    if show_badge:
        c1, c2 = st.columns(2)
        blabel = c1.text_input("Label", "Special Offer")
        bprice = c2.text_input("Price",  "£9.99")
        bsub   = st.text_input("Was price", "")
        bc1, bc2 = st.columns(2)
        badge_colour = bc1.color_picker("Badge colour", bcfg.get("primary","#333"))
        badge_size   = bc2.slider("Badge size", 0.10, 0.35, 0.20, 0.01)

    _sh("🏷️ Logo Size & Colour")
    logo_col_override = st.color_picker("Logo colour", bcfg.get("primary","#333"))
    logo_size_factor  = st.slider("Logo size", 0.08, 0.30, 0.15, 0.01)

    # ── Layout ─────────────────────────────────────────────────────────────────
    _sh("📐 Layout & Format")
    layout = st.selectbox("Layout", LAYOUTS, label_visibility="collapsed")
    fmt_map = {
        "Square (1080×1080)":(1080,1080),"Landscape (1200×628)":(1200,628),
        "Stories (1080×1920)":(1080,1920),"Facebook (1200×630)":(1200,630),
        "Twitter (1200×675)":(1200,675),"LinkedIn (1200×627)":(1200,627)}
    fmt_name = st.selectbox("Format", list(fmt_map.keys()), label_visibility="collapsed")
    dims = fmt_map[fmt_name]

    # ── Background ─────────────────────────────────────────────────────────────
    _sh("🎨 Background")
    bg_type = st.selectbox("Type", [
        "Brand colour","Solid colour","Gradient preset",
        "Custom gradient","Upload image"], label_visibility="collapsed")

    bg_color = bcfg.get("bg_default","#F0F4F8")
    bg_bot   = ""

    if bg_type == "Solid colour":
        bg_color = st.color_picker("Colour", bcfg.get("bg_default","#F0F4F8"))
        st.session_state.bg_image = None
    elif bg_type == "Gradient preset":
        gnames = list(BG_GRADIENTS.keys()) if BG_GRADIENTS else []
        if gnames:
            gn = st.selectbox("Preset", gnames, label_visibility="collapsed")
            g  = BG_GRADIENTS.get(gn)
            if g:
                bg_color, bg_bot = g[0], g[1]
                st.markdown(f'<div style="height:20px;border-radius:4px;background:linear-gradient(to right,{bg_color},{bg_bot});margin:4px 0"></div>', unsafe_allow_html=True)
        st.session_state.bg_image = None
    elif bg_type == "Custom gradient":
        c1, c2 = st.columns(2)
        bg_color = c1.color_picker("Top",    bcfg.get("bg_default","#BFE0F5"))
        bg_bot   = c2.color_picker("Bottom", "#FFFFFF")
        st.markdown(f'<div style="height:20px;border-radius:4px;background:linear-gradient(to bottom,{bg_color},{bg_bot});margin:4px 0"></div>', unsafe_allow_html=True)
        st.session_state.bg_image = None
    elif bg_type == "Upload image":
        bgf = st.file_uploader("BG image", type=["png","jpg","jpeg"])
        if bgf: st.session_state.bg_image = Image.open(bgf)
        bg_color = bcfg.get("bg_default","#F0F4F8")
    else:
        bg_color = bcfg.get("bg_default","#F0F4F8")
        st.session_state.bg_image = None

    # Pattern overlay
    pat_name = st.selectbox("Pattern overlay", list(BG_PATTERNS.keys()),
                             label_visibility="collapsed")
    bg_pattern = BG_PATTERNS.get(pat_name, "none")

    # ── Typography ─────────────────────────────────────────────────────────────
    _sh("✍️ Typography")
    brand_font  = bcfg.get("font_family","Poppins (Modern)")
    default_fi  = next((i for i,f in enumerate(FONTS) if brand_font.split()[0] in f), 0)
    font_name   = st.selectbox("Font family", FONTS, index=default_fi,
                                label_visibility="collapsed")
    font_weight = st.radio("Weight", ["Bold","Regular","Medium"], horizontal=True)
    c1, c2 = st.columns(2)
    hl_auto = c1.checkbox("Auto size", value=True, key="hl_auto_chk")
    sh_auto = c2.checkbox("Auto size", value=True, key="sh_auto_chk")
    hl_size_raw = c1.slider("Headline px", 10, 150, 48, key="hl_size_sl")
    sh_size_raw = c2.slider("Subhead px",  10, 100, 24, key="sh_size_sl")
    # Pass 0 = auto, or exact px value
    hl_size = 0 if hl_auto else hl_size_raw
    sh_size = 0 if sh_auto else sh_size_raw
    if not hl_auto: c1.caption(f"📏 {hl_size_raw}px")
    else: c1.caption("📏 auto")
    if not sh_auto: c2.caption(f"📏 {sh_size_raw}px")
    else: c2.caption("📏 auto")
    c3, c4 = st.columns(2)
    hl_col  = c3.color_picker("Headline colour", bcfg.get("text_dark","#111"))
    sh_col  = c4.color_picker("Subhead colour",  "#555555")
    hl_upper = st.checkbox("UPPERCASE headline", value=True)

    # Colour palettes
    if COLOR_PALETTES:
        with st.expander("🎨 Colour Palettes"):
            pn = st.selectbox("Palette", ["—"]+list(COLOR_PALETTES.keys()))
            if pn != "—":
                pal = COLOR_PALETTES[pn]
                sw  = "".join([f'<span style="display:inline-block;width:22px;height:22px;background:{c};border-radius:3px;margin:2px;border:1px solid #444" title="{c}"></span>' for c in pal])
                st.markdown(sw, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="background:linear-gradient(135deg,#0d1117,#161b22);
     border:1px solid #21262d;border-radius:12px;
     padding:20px 28px;margin-bottom:20px">
  <div style="display:flex;align-items:center;gap:16px">
    <div style="font-size:36px">🎨</div>
    <div>
      <h1 style="margin:0;color:#e6edf3;font-size:22px;font-weight:700">
        GenAI Creative Compliance Studio</h1>
      <p style="margin:0;color:#8b949e;font-size:13px">
        65+ brands · AI/ML compliance · Video editor · Creative generator · One command
      </p>
    </div>
    <div style="margin-left:auto;display:flex;gap:8px">
      <span style="background:#238636;color:white;padding:4px 10px;border-radius:6px;font-size:11px;font-weight:700">✅ ML Models</span>
      <span style="background:#1f6feb;color:white;padding:4px 10px;border-radius:6px;font-size:11px;font-weight:700">🔍 OCR Active</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🖼️ Creative Builder",
    "🎬 Video Editor",
    "🔍 AI Compliance",
    "✨ AI Director",
    "📊 Analytics",
    "🧠 Advanced AI",
])

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — CREATIVE BUILDER
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    left, right = st.columns([1, 2])

    with left:
        # ── Upload up to 5 packshots ────────────────────────────────────────
        st.subheader("📸 Packshots (up to 5)")
        uploads = st.file_uploader(
            "Upload product images",
            type=["png","jpg","jpeg","webp"],
            accept_multiple_files=True,
            key="packshot_upload")

        if uploads:
            if len(uploads) > 5:
                st.warning("Max 5 images — using first 5.")
                uploads = uploads[:5]
            st.session_state.packshots = [Image.open(f) for f in uploads]
            n = len(st.session_state.packshots)
            cols = st.columns(min(n, 5))
            for i, (ps, c) in enumerate(zip(st.session_state.packshots, cols)):
                c.image(ps, caption=f"#{i+1}", use_column_width=True)
            st.caption(f"✅ {n} image{'s' if n>1 else ''} ready")

        # ── Image editing ────────────────────────────────────────────────────
        if st.session_state.packshots:
            with st.expander("🖌️ Image Adjustments", expanded=True):
                a1, a2 = st.columns(2)
                brightness = a1.slider("☀️ Brightness",  0.3, 2.0, 1.0, 0.05, key="ib")
                contrast   = a2.slider("◑ Contrast",     0.3, 2.0, 1.0, 0.05, key="ic")
                saturation = a1.slider("🎨 Saturation",  0.0, 2.5, 1.0, 0.05, key="is")
                sharpness  = a2.slider("🔪 Sharpness",   0.0, 2.5, 1.0, 0.05, key="ish")
                blur_v     = a1.slider("🌫️ Blur",        0,   12,  0,   key="ibl")
                exposure   = a2.slider("💡 Exposure",    0.3, 2.0, 1.0, 0.05, key="ie")

            with st.expander("🎞️ Filters & Transforms"):
                img_filter = st.selectbox("Filter", [
                    "None","Warm","Cool","Vibrant","Matte",
                    "Black & White","Vintage","Cinematic"], key="iflt")
                t1, t2, t3 = st.columns(3)
                flip_h  = t1.checkbox("↔ Flip H", key="ifh")
                flip_v  = t2.checkbox("↕ Flip V", key="ifv")
                rotate  = t3.selectbox("↻ Rotate", [0,90,180,270], key="irot")

            with st.expander("🤖 AI Background Removal"):
                remove_bg = st.checkbox("Remove background (AI)")
                if remove_bg:
                    st.info("Uses smart corner-sampling background removal.")

            edits = {
                "brightness": brightness * exposure,
                "contrast":   contrast,
                "saturation": saturation,
                "sharpness":  sharpness,
                "blur":       blur_v,
                "filter":     img_filter.lower().replace(" & ","").replace(" ","_").replace("black_white","bw").replace("none","none"),
                "flip_h":     flip_h,
                "flip_v":     flip_v,
                "rotate":     rotate,
            }

            # Live preview
            if M.get("RENDERER") and M.get("apply_edits"):
                try:
                    prev = M["apply_edits"](st.session_state.packshots[0].copy(), edits)
                    st.image(prev, caption="Preview (image 1)", use_column_width=True)
                except Exception:
                    pass

            if st.button("🔄 Apply to all packshots", use_container_width=True):
                processed = []
                for ps in st.session_state.packshots:
                    p = ps.copy()
                    if remove_bg and M.get("smart_bg_remove"):
                        try: p = M["smart_bg_remove"](p)
                        except Exception: pass
                    if M.get("apply_edits"):
                        try: p = M["apply_edits"](p, edits)
                        except Exception: pass
                    processed.append(p)
                st.session_state.processed = processed
                st.success(f"✅ {len(processed)} packshot(s) processed!")
                cols2 = st.columns(min(len(processed), 5))
                for i, (p, c) in enumerate(zip(processed, cols2)):
                    c.image(p, caption=f"#{i+1}", use_column_width=True)
        else:
            edits = {}

        # ── Packshot positions ────────────────────────────────────────────────
        if st.session_state.packshots:
            with st.expander("📐 Packshot Positions"):
                use_auto = st.checkbox("Auto layout", value=True)
                positions = []
                if not use_auto:
                    for i in range(min(5, len(st.session_state.packshots))):
                        st.markdown(f"**Image {i+1}**")
                        pc1, pc2, pc3 = st.columns(3)
                        px_ = pc1.slider("X", 0.0, 1.0, [0.65,0.35,0.70,0.50,0.80][i], 0.01, key=f"px{i}")
                        py_ = pc2.slider("Y", 0.0, 1.0, [0.40,0.60,0.25,0.50,0.35][i], 0.01, key=f"py{i}")
                        ps_ = pc3.slider("Scale", 0.1, 0.9, 0.40, 0.01, key=f"psc{i}")
                        positions.append({"x":px_,"y":py_,"scale":ps_})
                st.session_state["positions"] = positions if not use_auto else []

    with right:
        st.subheader("🎨 Generate Creative")

        # Format display
        st.caption(f"Format: **{fmt_name}** — {dims[0]}×{dims[1]}px")

        if st.button("✨ Generate Creative", type="primary", use_container_width=True):
            if not (headline or subhead):
                st.warning("Add a headline or subhead first.")
            elif not M.get("RENDERER"):
                st.error(f"Renderer not available: {M.get('renderer_err','')}")
            else:
                packshots_to_use = (st.session_state.processed or
                                    st.session_state.packshots or [])
                if not packshots_to_use:
                    st.warning("Upload at least one packshot image.")
                else:
                    with st.spinner("Generating…"):
                        try:
                            img = M["render_creative"](
                                dimensions=dims,
                                packshots=packshots_to_use,
                                headline=headline or "New Arrival",
                                subhead=subhead  or "Available now",
                                brand_name=brand,
                                badge_label=blabel,
                                badge_price=bprice,
                                badge_sub=bsub,
                                badge_show=show_badge and bool(bprice),
                                tag_text=tag_text,
                                bg_color=bg_color,
                                bg_gradient_bot=bg_bot,
                                bg_image=st.session_state.bg_image,
                                layout_preset=layout,
                                font_name=font_name,
                                font_weight=font_weight.lower(),
                                headline_size=hl_size,
                                subhead_size=sh_size,
                                headline_color=hl_col,
                                subhead_color=sh_col,
                                headline_uppercase=hl_upper,
                                show_logo=show_logo,
                                include_drinkaware=include_da,
                                packshot_edits=edits if edits else None,
                                packshot_positions=st.session_state.get("positions") or None,
                                texture_style=bg_pattern,
                                logo_size_factor=logo_size_factor,
                                logo_col_override=logo_col_override,
                                badge_size_factor=badge_size,
                                badge_col_override=badge_colour,
                            )
                            st.session_state.generated.append(img)
                            st.image(img, use_column_width=True)

                            # Download buttons
                            d1, d2 = st.columns(2)
                            d1.download_button("📥 PNG",
                                data=_to_bytes(img,"PNG"),
                                file_name=f"creative_{brand.lower().replace(' ','_')}.png",
                                mime="image/png", use_container_width=True, key="dl_png")
                            d2.download_button("📥 JPEG",
                                data=_to_bytes(img,"JPEG"),
                                file_name=f"creative_{brand.lower().replace(' ','_')}.jpg",
                                mime="image/jpeg", use_container_width=True, key="dl_jpg")

                        except Exception as e:
                            st.error(f"Error: {e}")
                            import traceback; st.code(traceback.format_exc())

        # Show previous generations
        if st.session_state.generated:
            st.subheader(f"Previous Creatives ({len(st.session_state.generated)})")
            cols_prev = st.columns(min(3, len(st.session_state.generated)))
            for i, (gi, gc) in enumerate(zip(reversed(st.session_state.generated[-6:]), cols_prev*2)):
                gc.image(gi, use_column_width=True)
                gc.download_button("⬇️", data=_to_bytes(gi,"JPEG"),
                    file_name=f"creative_{i}.jpg", mime="image/jpeg",
                    use_container_width=True, key=f"prev_dl_{i}")

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — VIDEO EDITOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🎬 Video Editor")

    if not M.get("VIDEO"):
        st.error(f"Video editor unavailable: {M.get('video_err','ffmpeg not found')}")
        st.info("Install ffmpeg: https://ffmpeg.org/download.html — then restart the app.")
    else:
        vid_up = st.file_uploader("Upload video",
            type=["mp4","mov","avi","mkv","webm","m4v"], key="vid_upload")

        if vid_up:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(vid_up.read())
                vpath = tmp.name

            info = M["get_info"](vpath)
            dur  = float(info.get("duration", 10.0))

            # Info row
            ic = st.columns(5)
            ic[0].metric("Duration",   f"{dur:.1f}s")
            ic[1].metric("Resolution", f"{info.get('width','?')}×{info.get('height','?')}")
            ic[2].metric("FPS",        info.get("fps","?"))
            ic[3].metric("Size",       f"{info.get('size_mb','?')} MB")
            ic[4].metric("Audio",      "✅" if info.get("has_audio") else "❌")

            # Video player — watch before trimming
            st.subheader("▶️ Preview Video")
            with open(vpath, "rb") as _vf:
                _vbytes = _vf.read()
            st.video(_vbytes)
            st.caption("Watch the video above to find exact trim points before editing.")

            # Frame strip
            st.subheader("🎞️ Frame Strip")
            frames = M["get_frames"](vpath, n_frames=8)
            if frames:
                fc = st.columns(len(frames))
                for i,(fr,fcc) in enumerate(zip(frames,fc)):
                    ts = round(dur * i / max(len(frames)-1,1), 1)
                    fcc.image(fr, use_column_width=True, caption=f"{ts}s")
            st.caption("Frame strip shows key moments. Use timestamps above to set trim points.")

            st.divider()
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("**✂️ Trim & Speed**")
                t_s = st.slider("Start (sec)", 0.0, max(dur-0.1,0.0), 0.0, 0.1, key="vts")
                t_e = st.slider("End (sec)", 0.1, dur, dur, 0.1, key="vte")
                spd = st.select_slider("Speed",
                    [0.25,0.5,0.75,1.0,1.25,1.5,2.0], value=1.0, key="vspd")
                st.markdown("**🔄 Transform**")
                vfh = st.checkbox("↔ Flip H", key="vfh")
                vfv = st.checkbox("↕ Flip V", key="vfv")
                vrot = st.selectbox("Rotate°", [0,90,180,270], key="vrot")
                st.markdown("**📤 Output**")
                out_res = st.selectbox("Resolution", [
                    "Original","1080p (1920×1080)","720p (1280×720)",
                    "480p (854×480)","Square (1080×1080)",
                    "Stories (1080×1920)","Landscape (1200×628)"], key="vres")
                out_q = st.selectbox("Quality",
                    ["High","Ultra (best)","Medium","Low (smallest)"], key="vq")

            with c2:
                st.markdown("**🎨 Colour & Light**")
                vbr = st.slider("☀️ Brightness", -1.0, 1.0, 0.0, 0.05, key="vbr")
                vco = st.slider("◑ Contrast",    0.0,  3.0, 1.0, 0.1,  key="vco")
                vsa = st.slider("🎨 Saturation", 0.0,  3.0, 1.0, 0.1,  key="vsa")
                vga = st.slider("💡 Exposure",   0.1,  3.0, 1.0, 0.1,  key="vga")
                vsh = st.slider("🔪 Sharpness",  0.0,  5.0, 0.0, 0.1,  key="vsh")
                vflt = st.selectbox("🎞️ Filter", [
                    "None","Warm","Cool","Vibrant","Matte",
                    "Black & White","Vintage","Cinematic","HDR","Faded"], key="vflt")
                vvig = st.checkbox("🔲 Vignette", key="vvig")
                vfi  = st.slider("Fade in (s)",  0.0, 3.0, 0.0, 0.1, key="vfi")
                vfo  = st.slider("Fade out (s)", 0.0, 3.0, 0.0, 0.1, key="vfo")

            with c3:
                st.markdown("**🔊 Audio**")
                vvol  = st.slider("Volume", 0.0, 3.0, 1.0, 0.1, key="vvol")
                vmute = st.checkbox("🔇 Mute", key="vmute")
                vnorm = st.checkbox("📊 Normalise", key="vnorm")
                vbass = st.checkbox("🎵 Bass boost", key="vbass")
                vnr   = st.checkbox("🔕 Noise reduce", key="vnr")

                st.markdown("**✏️ Text Overlay**")
                vtxt  = st.text_input("Text", placeholder="e.g. NEW PRODUCT", key="vtxt")
                vtpos = st.selectbox("Position",
                    ["bottom","top","center","top-left","top-right"], key="vtpos")
                vtsz  = st.slider("Font size", 20, 120, 48, key="vtsz")
                vtcol = st.color_picker("Text colour", "#FFFFFF", key="vtcol")

                st.markdown("**🎵 Background Music**")
                bgm = st.file_uploader("Music (mp3/wav)", type=["mp3","wav","aac"], key="vbgm")
                bgm_vol = 0.3
                if bgm: bgm_vol = st.slider("Music volume", 0.0, 1.0, 0.3, 0.05, key="vbgmv")

            if st.button("🎬 Process & Export Video", type="primary", use_container_width=True):
                with st.spinner("Processing video… this may take a minute."):
                    out_path = vpath.replace(".mp4","_out.mp4")
                    ok, out, err = M["proc_video"](
                        input_path=vpath, output_path=out_path,
                        trim_start=t_s, trim_end=t_e if t_e < dur else 0.0,
                        brightness=vbr, contrast=vco, saturation=vsa,
                        gamma=vga, sharpness=vsh,
                        flip_h=vfh, flip_v=vfv, rotate=vrot,
                        speed=spd, colour_filter=vflt,
                        volume=vvol, normalize_audio=vnorm,
                        noise_reduce=vnr, bass_boost=vbass, mute=vmute,
                        text_overlay=vtxt, text_position=vtpos,
                        text_size=vtsz, text_colour=vtcol.lstrip("#"),
                        output_quality=out_q,
                        output_resolution=out_res if out_res != "Original" else "",
                        vignette=vvig, fade_in=vfi, fade_out=vfo,
                    )
                    if ok and os.path.exists(out_path):
                        # Add background music if uploaded
                        if bgm:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as mf:
                                mf.write(bgm.read()); mpath = mf.name
                            mout = out_path.replace(".mp4","_m.mp4")
                            ok2, _ = M["add_bg_music"](out_path, mpath, mout, bgm_vol)
                            if ok2 and os.path.exists(mout): out_path = mout
                        with open(out_path,"rb") as f: vbytes = f.read()
                        st.success(f"✅ Done! {len(vbytes)//1024}KB")
                        st.download_button("📥 Download Edited Video",
                            data=vbytes, file_name=f"edited_{vid_up.name}",
                            mime="video/mp4", use_container_width=True)
                        thumb = M["get_thumb"](out_path, 1.0)
                        if thumb: st.image(thumb, caption="Output preview", width=320)
                    else:
                        st.error(f"Failed: {(err or 'unknown error')[:300]}")
        else:
            # Feature overview when no video uploaded
            fc1, fc2, fc3 = st.columns(3)
            fc1.markdown("""
**✂️ Trim & Speed**
- Set start/end timestamps
- 0.25× slow-mo to 2× fast
- Any clip section

**🔄 Transform**
- Flip horizontal/vertical
- Rotate 90/180/270°
- Resize to any format
""")
            fc2.markdown("""
**🎨 Colour & Light**
- Brightness, contrast, saturation
- Gamma/exposure
- Sharpness
- 10 colour filter presets
- Vignette effect
- Fade in/out
""")
            fc3.markdown("""
**🔊 Audio**
- Volume control
- Mute track
- Loudness normalise
- Bass boost
- Noise reduction
- Add background music

**✏️ Text**
- Burn text overlay
- Position & font size
- Custom colour
""")

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — AI COMPLIANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔍 AI Compliance Audit")
    st.caption("Upload any ad creative — OCR + NLP + Computer Vision + Risk Score + PDF report")

    if not M.get("ML"):
        st.error(f"ML pipeline unavailable: {M.get('ml_err','')}")
    else:
        al, ar = st.columns([1, 2])
        with al:
            audit_file = st.file_uploader("Upload creative image",
                type=["png","jpg","jpeg","webp"], key="audit_img")
            audit_brand = st.selectbox("Brand", list(BRANDS.keys()) or ["Custom"],
                                        key="ab")
            audit_fmt   = st.selectbox("Format",
                ["Square (1080×1080)","Landscape (1200×628)","Stories (1080×1920)"],
                key="afmt")
            audit_hl = st.text_input("Headline text", key="ahl")
            audit_sh = st.text_input("Subhead text",  key="ash")
            run_audit = st.button("🚀 Run Full AI Audit", type="primary",
                                   use_container_width=True,
                                   disabled=audit_file is None)
        with ar:
            if audit_file:
                aimg = Image.open(audit_file)
                st.image(aimg, caption=f"{aimg.size[0]}×{aimg.size[1]}px",
                         use_column_width=True)

        if run_audit and audit_file:
            aimg = Image.open(audit_file)
            with st.spinner("🤖 Running AI/ML pipeline…"):
                result = M["run_pipeline"](aimg, audit_hl, audit_sh,
                                           audit_brand, audit_fmt, save_to_db=True)
            st.session_state.audit_result = result
            st.success(f"✅ Audit complete — ID #{result.get('audit_id','?')}")
            st.divider()

            risk = result["risk"]
            # Score gauge
            import plotly.graph_objects as go
            score = risk["total_score"]
            grade = risk["grade"]
            gcols = {"A":"#238636","B":"#1f6feb","C":"#9e6a03","D":"#da3633","F":"#6e1818"}
            gcol  = gcols.get(grade,"#888")
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=score,
                title={"text":f"Risk Score — Grade {grade}","font":{"size":14}},
                gauge={"axis":{"range":[0,100]},
                       "bar":{"color":gcol,"thickness":0.25},
                       "steps":[{"range":[0,30],"color":"#0d2b0d"},
                                {"range":[30,60],"color":"#2d1b00"},
                                {"range":[60,100],"color":"#2d0000"}]},
                number={"font":{"size":36,"color":gcol},"suffix":"/100"}))
            fig.update_layout(height=220, margin=dict(t=40,b=0,l=20,r=20),
                              paper_bgcolor="rgba(0,0,0,0)")

            sc1, sc2 = st.columns([1,2])
            with sc1:
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f'<div class="mpill risk-{grade}">{grade} — {risk["risk_label"]}</div>',
                            unsafe_allow_html=True)
            with sc2:
                bd = risk.get("breakdown",{})
                import plotly.express as px
                df_bd = __import__("pandas").DataFrame(
                    [{"Factor":k,"Points":v} for k,v in bd.items()])
                if not df_bd.empty:
                    fig2 = px.bar(df_bd, x="Points", y="Factor", orientation="h",
                                  color="Points", color_continuous_scale=["#238636","#9e6a03","#da3633"],
                                  range_color=[0,35])
                    fig2.update_layout(height=240, margin=dict(t=10,b=10,l=10,r=10),
                                       showlegend=False, coloraxis_showscale=False,
                                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                       font_color="#c9d1d9")
                    st.plotly_chart(fig2, use_container_width=True)

            # Detailed results in tabs
            rt1, rt2, rt3, rt4, rt5 = st.tabs([
                "🔤 NLP", "👁️ Vision", "📝 OCR", "📏 Fonts & Contrast", "🤖 Image ML"])

            with rt1:
                viols = result["text"].get("violations",[])
                if viols:
                    for v in viols:
                        sev = v["severity"]
                        cls = {"CRITICAL":"crit","HIGH":"high","MEDIUM":"med","LOW":"low"}.get(sev,"low")
                        st.markdown(
                            f'<div class="mpill risk-{cls}">{sev}</div> '
                            f'<strong>{v["category"]}</strong> ({v["confidence"]}%)<br>'
                            f'<em>"{v["text"][:80]}"</em><br>'
                            f'💡 {v["suggestion"]}', unsafe_allow_html=True)
                        st.divider()
                else:
                    st.success("✅ No NLP text violations detected.")

            with rt2:
                dets = result["detections"]
                if dets:
                    ann = result["annotated"]
                    st.image(ann, caption="Annotated — bounding boxes",
                             use_column_width=True)
                    for d in dets:
                        risk_cls = {"CRITICAL":"crit","HIGH":"high","MEDIUM":"med","INFO":"low"}.get(d["compliance_risk"],"low")
                        st.markdown(
                            f'<div class="mpill risk-{risk_cls}">{d["compliance_risk"]}</div> '
                            f'{d["icon"]} **{d["label"]}** — {d["confidence"]}%<br>'
                            f'<span style="color:#8b949e;font-size:12px">{d["rule"]}</span>',
                            unsafe_allow_html=True)
                else:
                    st.success("✅ No compliance objects detected.")

            with rt3:
                ocr = result["ocr"]
                st.metric("OCR Confidence", f"{ocr.get('confidence',0):.0f}%")
                st.text_area("Extracted text", ocr.get("text","(none)"),
                             height=100, disabled=True)
                if ocr.get("words"):
                    import pandas as pd
                    df_w = pd.DataFrame([{
                        "Word":w["text"],"Conf":w["conf"],"Size(px)":w["size_px"]
                    } for w in ocr["words"] if w["conf"]>30])
                    st.dataframe(df_w.head(20), use_container_width=True, hide_index=True)

            with rt4:
                fonts = result.get("fonts",{})
                contrast = result.get("contrast",{})
                if fonts.get("measurements"):
                    import pandas as pd
                    df_f = pd.DataFrame(fonts["measurements"])
                    df_f["Status"] = df_f["pass"].map({True:"✅ PASS",False:"❌ FAIL"})
                    st.dataframe(df_f[["element","measured_px","min_required","Status"]],
                                 use_container_width=True, hide_index=True)
                if contrast:
                    ratio = contrast.get("ratio",0)
                    passed = contrast.get("pass",True)
                    st.metric("WCAG AA Contrast", f"{ratio}:1",
                              "✅ Pass (≥4.5:1)" if passed else "❌ Fail (<4.5:1)")

            with rt5:
                ir = result.get("image_risk",{})
                st.metric("Image Risk Category",
                          ir.get("image_risk_category","unknown").replace("_"," ").title())
                scores = ir.get("image_risk_scores",{})
                if scores:
                    import pandas as pd, plotly.express as px
                    df_ir = pd.DataFrame([{
                        "Category":k.replace("_risk","").replace("_"," ").title(),
                        "Score":round(float(v),1)
                    } for k,v in scores.items()])
                    fig_ir = px.bar(df_ir, x="Score", y="Category", orientation="h",
                                   color="Score", color_continuous_scale=["#238636","#da3633"])
                    fig_ir.update_layout(height=180, margin=dict(t=10,b=10,l=10,r=10),
                                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                         coloraxis_showscale=False, font_color="#c9d1d9")
                    st.plotly_chart(fig_ir, use_container_width=True)

            # PDF download
            st.divider()
            pdf = result.get("pdf",b"")
            if pdf:
                st.download_button("📄 Download PDF Report", data=pdf,
                    file_name=f"compliance_{audit_brand}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — AI DIRECTOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("✨ AI Creative Director")
    st.caption("AI designs the entire creative — layout, colours, fonts, copy, textures. Auto-generates multiple variations.")

    if not M.get("AI_DIRECTOR"):
        st.warning("AI Director module unavailable.")
    else:
        if not M.get("AI_API"):
            st.info("ℹ️ No API key set — using built-in algorithmic design. Set `ANTHROPIC_API_KEY` for Claude AI.")

        adl, adr = st.columns([1, 2])
        with adl:
            ai_prod  = st.text_input("Product name", placeholder="e.g. Brancott Chardonnay")
            ai_cat   = st.selectbox("Category", [
                "General","Alcohol","Grocery","Electronics","Fashion",
                "Home & Garden","Beauty & Health","Finance","Healthcare"])
            ai_hl    = st.text_input("Headline (blank = AI writes)", placeholder="Leave blank")
            ai_sh    = st.text_input("Subhead (blank = AI writes)",  placeholder="Leave blank")
            ai_price = st.text_input("Badge price (optional)", placeholder="e.g. £9.99")
            ai_fmt   = st.selectbox("Format", [
                "Square (1080×1080)","Landscape (1200×628)","Stories (1080×1920)"])
            ai_vars  = st.slider("Variations", 1, 4, 2)
            ai_dims  = {"Square (1080×1080)":(1080,1080),
                        "Landscape (1200×628)":(1200,628),
                        "Stories (1080×1920)":(1080,1920)}[ai_fmt]
            run_ai = st.button("🎨 AI Generate", type="primary",
                                use_container_width=True,
                                disabled=not st.session_state.packshots)
            if not st.session_state.packshots:
                st.caption("Upload packshots in the Creative Builder tab first.")

        with adr:
            if run_ai and st.session_state.packshots:
                src = M.get("AI_API","Algorithmic AI")
                with st.spinner(f"🎨 AI designing {ai_vars} variation(s)…"):
                    try:
                        results = M["gen_variations"](
                            brand_name=brand,
                            product_name=ai_prod or headline or "product",
                            product_category=ai_cat,
                            dimensions=ai_dims,
                            packshots=(st.session_state.processed or
                                       st.session_state.packshots),
                            user_headline=ai_hl,
                            user_subhead=ai_sh,
                            badge_price=ai_price,
                            badge_label="",
                            n_variations=ai_vars,
                        )
                        good = [r for r in results if r.get("image")]
                        st.success(f"✅ {len(good)} creative(s) generated!")
                        for i, r in enumerate(good):
                            spec = r["spec"]
                            with st.expander(
                                f"Variation {i+1} — {spec.get('layout_preset','')} · "
                                f"{spec.get('font_name','').split('(')[0].strip()} · "
                                f"{spec.get('texture_style','none')} texture",
                                expanded=(i==0)):
                                mc = st.columns(4)
                                mc[0].metric("Layout",  spec.get("layout_preset","").split()[0])
                                mc[1].metric("Font",    spec.get("font_name","").split("(")[0].strip())
                                mc[2].metric("Texture", spec.get("texture_style","none"))
                                mc[3].metric("Source",  "Claude AI" if "claude" in r.get("source","") else "Algorithmic")
                                if r.get("rationale"):
                                    st.caption(f"💡 {r['rationale']}")
                                # Colour swatches
                                bg_c = spec.get("bg_color","#fff")
                                hl_c = spec.get("headline_color","#111")
                                sh_c = spec.get("subhead_color","#555")
                                st.markdown(
                                    f'<div class="swatch-row">'
                                    f'<div class="swatch" style="background:{bg_c}" title="BG {bg_c}"></div>'
                                    f'<div class="swatch" style="background:{hl_c}" title="Headline {hl_c}"></div>'
                                    f'<div class="swatch" style="background:{sh_c}" title="Subhead {sh_c}"></div>'
                                    f'<span style="font-size:11px;color:#8b949e;align-self:center;margin-left:4px">BG · Headline · Subhead</span>'
                                    f'</div>', unsafe_allow_html=True)
                                st.image(r["image"], use_column_width=True)
                                dc1, dc2 = st.columns(2)
                                dc1.download_button("📥 PNG", data=_to_bytes(r["image"],"PNG"),
                                    file_name=f"ai_v{i+1}.png", mime="image/png",
                                    use_container_width=True, key=f"ai_dl_png_{i}")
                                dc2.download_button("📥 JPEG", data=_to_bytes(r["image"],"JPEG"),
                                    file_name=f"ai_v{i+1}.jpg", mime="image/jpeg",
                                    use_container_width=True, key=f"ai_dl_jpg_{i}")
                    except Exception as e:
                        st.error(f"Error: {e}")
                        import traceback; st.code(traceback.format_exc())
            else:
                st.markdown("""
### How it works

1. Upload packshot(s) in **Creative Builder** tab
2. Fill the brief on the left
3. Click **AI Generate**

The AI automatically selects:
- 🎨 Background colour + gradient
- 🌟 Texture / pattern overlay
- ✍️ Font family + sizes + colours
- 📐 Best layout for your product
- 💬 Headline + subhead copy (if left blank)
- 🔁 Multiple distinct variations

With an `ANTHROPIC_API_KEY`, Claude designs each variation from scratch.
Without a key, the built-in brand-profile system produces on-brand designs automatically.
""")

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("📊 Compliance Analytics")

    if not M.get("ML"):
        st.warning("ML pipeline not available.")
    else:
        db   = M["db"]
        stats = db.stats()

        # KPIs
        k1, k2, k3, k4, k5 = st.columns(5)
        total  = stats.get("total", 0)
        passed = stats.get("passed", 0)
        failed = stats.get("failed", 0)
        avg    = stats.get("avg_score", 0)
        pr     = round(passed / max(total,1) * 100)
        k1.metric("Total Audited",  total)
        k2.metric("Compliant",      passed, f"{pr}%")
        k3.metric("Non-Compliant",  failed)
        k4.metric("Avg Risk Score", avg)
        k5.metric("Last Audit",    (stats.get("last_ts","Never") or "Never")[:10])

        st.divider()
        history = db.history(limit=200)
        if not history:
            st.info("No audits yet. Run a compliance check in the AI Compliance tab.")
        else:
            import pandas as pd, plotly.express as px, plotly.graph_objects as go

            df = pd.DataFrame(history)
            df["ts"]   = pd.to_datetime(df["ts"])
            df["date"] = df["ts"].dt.date

            # Filters
            f1, f2 = st.columns(2)
            brand_f = f1.selectbox("Filter brand", ["All"]+sorted(df["brand"].dropna().unique().tolist()))
            grade_f = f2.multiselect("Filter grades", ["A","B","C","D","F"], default=["A","B","C","D","F"])
            if brand_f != "All": df = df[df["brand"]==brand_f]
            if grade_f: df = df[df["grade"].isin(grade_f)]

            if df.empty:
                st.warning("No data matches filters.")
            else:
                ch1, ch2 = st.columns(2)
                with ch1:
                    # Score history
                    fig_h = px.scatter(df, x="ts", y="risk_score", color="brand",
                                       title="Risk Score History",
                                       labels={"risk_score":"Score","ts":"Date"})
                    fig_h.add_hline(y=30, line_dash="dash", line_color="#238636",
                                    annotation_text="Pass threshold")
                    fig_h.update_layout(height=280, margin=dict(t=40,b=20,l=20,r=20),
                                        paper_bgcolor="rgba(0,0,0,0)",
                                        plot_bgcolor="rgba(0,0,0,0)",
                                        font_color="#c9d1d9")
                    st.plotly_chart(fig_h, use_container_width=True)

                with ch2:
                    # Grade donut
                    gc = df["grade"].value_counts()
                    fig_g = go.Figure(go.Pie(
                        labels=gc.index, values=gc.values, hole=0.55,
                        marker_colors=["#238636","#1f6feb","#9e6a03","#da3633","#6e1818"]))
                    fig_g.update_layout(title="Grade Distribution", height=280,
                                        margin=dict(t=40,b=0,l=0,r=0),
                                        paper_bgcolor="rgba(0,0,0,0)",
                                        font_color="#c9d1d9")
                    st.plotly_chart(fig_g, use_container_width=True)

                ch3, ch4 = st.columns(2)
                with ch3:
                    ba = df.groupby("brand")["risk_score"].mean().reset_index()
                    ba.columns = ["Brand","Avg Score"]
                    fig_b = px.bar(ba, x="Brand", y="Avg Score", color="Avg Score",
                                   color_continuous_scale=["#238636","#9e6a03","#da3633"],
                                   range_color=[0,100], title="Avg Score by Brand")
                    fig_b.update_layout(height=260, margin=dict(t=40,b=40,l=20,r=20),
                                        paper_bgcolor="rgba(0,0,0,0)",
                                        plot_bgcolor="rgba(0,0,0,0)",
                                        font_color="#c9d1d9", coloraxis_showscale=False)
                    st.plotly_chart(fig_b, use_container_width=True)

                with ch4:
                    fig_hist = px.histogram(df, x="risk_score", nbins=20,
                                           title="Score Distribution",
                                           color_discrete_sequence=["#1f6feb"])
                    fig_hist.add_vline(x=30, line_dash="dash", line_color="#238636")
                    fig_hist.update_layout(height=260, margin=dict(t=40,b=20,l=20,r=20),
                                           paper_bgcolor="rgba(0,0,0,0)",
                                           plot_bgcolor="rgba(0,0,0,0)",
                                           font_color="#c9d1d9")
                    st.plotly_chart(fig_hist, use_container_width=True)

                # History table
                st.subheader("Audit History")
                grade_bg = {"A":"#238636","B":"#1f6feb","C":"#9e6a03","D":"#da3633","F":"#6e1818"}
                for row in history[:50]:
                    bg = grade_bg.get(row["grade"],"#333")
                    c1,c2,c3,c4,c5 = st.columns([1,2,3,2,2])
                    c1.markdown(f"`#{row['id']}`")
                    c2.markdown(f"**{row['brand'] or '—'}**")
                    c3.markdown(f"*{(row['headline'] or '—')[:45]}*")
                    c4.markdown(f"`{(row['ts'] or '')[:16]}`")
                    c5.markdown(
                        f'<span style="background:{bg};color:white;padding:2px 8px;border-radius:10px;font-weight:700;font-size:12px">'
                        f'{row["grade"]}</span> &nbsp; {row["risk_score"]}/100',
                        unsafe_allow_html=True)
                st.divider()
                if st.button("🗑️ Clear History", type="secondary"):
                    db.clear(); st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 6 — ADVANCED AI
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    adv1, adv2 = st.tabs(["🧠 XAI & Recommendations", "📋 Policy Analyser"])

    with adv1:
        if ADVANCED_AI_OK:
            render_advanced_ai_tab(
                audit_result=st.session_state.get("audit_result"),
                image=st.session_state.generated[-1] if st.session_state.generated else None,
                brand=brand,
                headline=headline,
                subhead=subhead,
                brand_primary=bcfg.get("primary","#00539F"),
                brand_accent=bcfg.get("accent","#E31837"),
            )
        else:
            st.warning("Place advanced_ai_tab.py, xai_explainer.py, vector_db.py in this folder.")

    with adv2:
        if POLICY_OK:
            policies = render_policy_uploader()
            st.divider()
            render_policy_analysis(headline, subhead, brand, policies)
        else:
            st.warning("Place policy_analyser.py in this folder.")

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#8b949e;font-size:12px;padding:20px 0;border-top:1px solid #21262d;margin-top:30px">
  GenAI Creative Compliance Studio &nbsp;·&nbsp;
  OCR + NLP + Computer Vision + Random Forest &nbsp;·&nbsp;
  65+ brands · 16 industries · Video Editor
</div>
""", unsafe_allow_html=True)