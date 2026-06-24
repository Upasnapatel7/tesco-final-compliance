"""
ai_creative_assistant.py
========================
Real-time AI creative assistant that suggests headlines, subheads,
tag lines, colours, layouts and copy improvements AS THE USER TYPES.

Works in two modes:
  1. WITH Claude API key → Claude generates high-quality suggestions
  2. WITHOUT key → NLP model + brand database generates suggestions

Shown in the sidebar of dashboard.py alongside the creative builder,
so users get suggestions BEFORE generating, saving time and rework.
"""

import os
import json
import random
import re
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  HEADLINE SUGGESTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

# High-performing headline structures by category
# Based on real advertising research and copywriting principles
_HEADLINE_TEMPLATES = {
    "Grocery": [
        "Discover {product} — Fresh Every Day",
        "New {product} — Now in Stores",
        "Find Your Favourite {product}",
        "Fresh {product}, Great Value",
        "Introducing {product} This Season",
        "Taste the Difference with {product}",
        "{product} — Quality You Can Trust",
        "The New {product} Has Arrived",
    ],
    "Fashion": [
        "The New {product} Collection",
        "Introducing {product} This Season",
        "{product} — Style Redefined",
        "Discover the New {product} Look",
        "{product} — New Season, New You",
        "The {product} Edit Is Here",
        "Explore the {product} Collection",
    ],
    "Electronics": [
        "Introducing the New {product}",
        "{product} — Engineered for You",
        "The Smarter {product} Is Here",
        "Meet the New {product}",
        "{product} — Performance Reimagined",
        "Experience the New {product}",
    ],
    "Beauty & Health": [
        "Introducing {product}",
        "Discover {product} — For You",
        "The New {product} Range",
        "{product} — Feel Your Best",
        "Try the New {product} Today",
        "{product} — Made for You",
    ],
    "Finance": [
        "Introducing {product}",
        "{product} — Built Around You",
        "Discover {product} Today",
        "The {product} That Works for You",
        "{product} — Your Way",
    ],
    "Automotive": [
        "Introducing the New {product}",
        "{product} — Drive the Difference",
        "The New {product} Is Here",
        "Meet the {product}",
        "{product} — Engineered to Perform",
    ],
    "General": [
        "Discover {product}",
        "Introducing {product}",
        "The New {product}",
        "Find {product} In Stores Now",
        "{product} — Now Available",
        "Experience {product}",
        "{product} — Something New",
    ],
    "Alcohol": [
        "Introducing {product}",
        "Discover {product}",
        "The New {product}",
        "{product} — Crafted with Care",
        "Explore {product}",
    ],
}

_SUBHEAD_TEMPLATES = {
    "Grocery":        ["Available in selected stores", "In store and online now",
                       "Find it at your local store", "Available across all stores",
                       "Fresh to your table", "In store now"],
    "Fashion":        ["Available in stores and online", "Shop the collection now",
                       "In store and online", "Available now at selected stores",
                       "Explore the full range"],
    "Electronics":    ["Available now", "In store and online",
                       "Find it at selected stores", "Available across all stores",
                       "Shop now"],
    "Beauty & Health":["In store and online now", "Available at selected retailers",
                       "Find it in stores now", "Available now"],
    "Finance":        ["Terms and conditions apply", "Available to new customers",
                       "Subject to eligibility", "Find out more today"],
    "Automotive":     ["Visit your nearest dealer", "Available at selected dealerships",
                       "Book a test drive today", "Find out more"],
    "General":        ["Available now", "In stores now", "Find out more",
                       "Available in selected stores", "Shop now",
                       "In store and online"],
    "Alcohol":        ["Please drink responsibly", "Available in selected stores",
                       "In store now", "Explore the range"],
}

# Words/phrases that perform well in proven advertising copy
_POWER_WORDS = {
    "discovery":    ["discover", "explore", "find", "introducing", "new", "meet"],
    "availability": ["now", "in stores", "available", "find it", "shop"],
    "quality":      ["crafted", "quality", "fresh", "premium", "trusted", "great"],
    "seasonal":     ["this season", "now", "today", "this autumn", "this spring"],
}

# Words that commonly cause compliance violations — used to warn user
_VIOLATION_TRIGGERS = {
    "price_violation":      ["save", "off", "%", "discount", "sale", "clearance",
                             "bargain", "cheap", "was", "now only", "£", "$",
                             "limited time", "act now", "hurry", "offer ends"],
    "health_claim":         ["healthy", "healthier", "boost", "immunity", "proven",
                             "clinical", "doctor", "cure", "wellness", "detox",
                             "weight loss", "anti-aging", "benefit", "treats"],
    "misleading_claim":     ["best", "number one", "#1", "greatest", "unbeatable",
                             "superior", "revolutionary", "breakthrough", "voted",
                             "awarded", "world's", "ultimate", "guaranteed"],
    "competition":          ["win", "prize", "competition", "enter", "free",
                             "draw", "raffle", "jackpot", "contest"],
    "age_restricted":       ["get drunk", "shots", "binge", "party hard",
                             "celebrate with alcohol", "drink up"],
    "sustainability_claim": ["eco", "sustainable", "green", "carbon neutral",
                             "recyclable", "biodegradable", "zero waste"],
}


class RealTimeViolationChecker:
    """
    Checks copy for violations as user types.
    Uses the real NLP model for accurate detection.
    """

    def __init__(self):
        try:
            from ml_pipeline import _nlp
            self._nlp = _nlp
            self._nlp_ok = True
        except ImportError:
            self._nlp = None
            self._nlp_ok = False

    def check(self, text: str) -> dict:
        """
        Check text for violations instantly.
        Returns violation info + specific fix suggestion.
        """
        if not text or len(text.strip()) < 3:
            return {"status": "ok", "violations": [], "quick_violations": []}

        violations = []

        # Fast rule-based check first (instant, no model needed)
        text_lower = text.lower()
        quick_violations = []
        for category, triggers in _VIOLATION_TRIGGERS.items():
            for trigger in triggers:
                if trigger in text_lower:
                    quick_violations.append({
                        "trigger":  trigger,
                        "category": category.replace("_", " ").title(),
                        "severity": "HIGH" if category in
                                    ("health_claim","competition","age_restricted")
                                    else "MEDIUM",
                    })
                    break

        # Real NLP model check
        if self._nlp_ok and len(text) > 5:
            try:
                result = self._nlp.analyse(text)
                violations = result.get("violations", [])
            except Exception:
                pass

        status = "ok"
        if violations:
            sev = violations[0]["severity"]
            status = "critical" if sev == "CRITICAL" else \
                     "high" if sev == "HIGH" else \
                     "medium" if sev == "MEDIUM" else "low"
        elif quick_violations:
            status = "warning"

        return {
            "status":           status,
            "violations":       violations,
            "quick_violations": quick_violations,
            "is_compliant":     len(violations) == 0 and len(quick_violations) == 0,
        }


class HeadlineSuggestionEngine:
    """
    Generates headline suggestions based on brand, category, and product.
    Uses Claude API when available, algorithmic system as fallback.
    """

    def __init__(self):
        self._api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self._checker = RealTimeViolationChecker()

    def suggest(self, brand: str, category: str, product: str,
                current_headline: str = "", n: int = 5) -> dict:
        """
        Generate n headline suggestions.
        Returns suggestions with compliance pre-check for each.
        """
        if self._api_key and len(self._api_key) > 10:
            suggestions = self._claude_suggest(brand, category, product,
                                               current_headline, n)
        else:
            suggestions = self._algorithmic_suggest(brand, category, product, n)

        # Pre-check each suggestion for compliance
        checked = []
        for s in suggestions:
            check = self._checker.check(s)
            checked.append({
                "headline":    s,
                "compliant":   check["is_compliant"],
                "status":      check["status"],
                "violation":   check["violations"][0]["category"]
                               if check["violations"] else None,
            })

        # Sort: compliant first, then by quality
        checked.sort(key=lambda x: (not x["compliant"], x["headline"]))

        return {
            "suggestions":  checked,
            "source":       "Claude AI" if (self._api_key and len(self._api_key)>10)
                            else "Algorithmic",
            "product":      product,
            "brand":        brand,
            "category":     category,
        }

    def _claude_suggest(self, brand, category, product,
                        current_headline, n) -> list:
        """Generate suggestions using Claude API."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self._api_key)

            prompt = f"""You are an expert advertising copywriter for {brand} ({category}).
Generate {n} short, punchy, compliant advertising headlines for: {product or 'a new product'}.

Rules (strictly follow all):
- Maximum 8 words per headline
- No health claims (no: proven, boost, immunity, healthy, clinical)
- No superlatives (no: best, number one, greatest, unbeatable)
- No price claims (no: save, discount, sale, offer, £, $)
- No competition language (no: win, prize, free, enter)
- No urgent language (no: hurry, limited time, act now)
- Must sound natural and aspirational
- Must fit {brand} brand voice
- Start with action words like: Discover, Introducing, Find, Explore, Meet, New
{f'Current headline to improve: "{current_headline}"' if current_headline else ''}

Return ONLY a JSON array of {n} headline strings, nothing else.
Example: ["Discover the New Range", "Introducing Something Special"]"""

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text.strip()
            text = re.sub(r"```json|```", "", text).strip()
            suggestions = json.loads(text)
            return [str(s) for s in suggestions[:n]]
        except Exception as e:
            return self._algorithmic_suggest(brand, category, product, n)

    def _algorithmic_suggest(self, brand, category, product, n) -> list:
        """Generate suggestions algorithmically from templates."""
        random.seed(hash(f"{brand}{category}{product}") % 2**31)
        cat_key = category if category in _HEADLINE_TEMPLATES else "General"
        templates = _HEADLINE_TEMPLATES[cat_key]

        product_name = product.strip() if product.strip() else "New Product"
        # Title case the product name
        product_display = " ".join(w.capitalize() for w in product_name.split())

        suggestions = []
        used = set()
        shuffled = templates.copy()
        random.shuffle(shuffled)

        for tpl in shuffled:
            s = tpl.replace("{product}", product_display)
            if s not in used:
                suggestions.append(s)
                used.add(s)
            if len(suggestions) >= n:
                break

        # If not enough, generate simple ones
        while len(suggestions) < n:
            s = f"Discover {product_display}"
            if s not in used:
                suggestions.append(s)
                used.add(s)
            else:
                suggestions.append(f"The New {product_display}")
                break

        return suggestions[:n]


class SubheadSuggestionEngine:
    """Generates subheadline suggestions."""

    def __init__(self):
        self._api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self._checker = RealTimeViolationChecker()

    def suggest(self, brand: str, category: str, product: str,
                headline: str = "", n: int = 4) -> dict:
        if self._api_key and len(self._api_key) > 10:
            suggestions = self._claude_suggest(brand, category, product,
                                               headline, n)
        else:
            suggestions = self._algorithmic_suggest(category, n)

        checked = []
        for s in suggestions:
            check = self._checker.check(s)
            checked.append({
                "subhead":   s,
                "compliant": check["is_compliant"],
                "status":    check["status"],
            })
        checked.sort(key=lambda x: not x["compliant"])
        return {"suggestions": checked, "source":
                "Claude AI" if (self._api_key and len(self._api_key)>10)
                else "Algorithmic"}

    def _claude_suggest(self, brand, category, product, headline, n) -> list:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self._api_key)
            prompt = f"""Write {n} short subheadlines (max 6 words each) for a {brand} ad.
Headline: "{headline}"
Product: {product}
Rules: no price claims, no health claims, no superlatives, available in stores type messaging.
Return ONLY a JSON array of {n} strings."""
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}])
            text = re.sub(r"```json|```", "",
                          response.content[0].text.strip()).strip()
            return [str(s) for s in json.loads(text)[:n]]
        except Exception:
            return self._algorithmic_suggest(category, n)

    def _algorithmic_suggest(self, category, n) -> list:
        cat_key = category if category in _SUBHEAD_TEMPLATES else "General"
        options = _SUBHEAD_TEMPLATES[cat_key]
        random.shuffle(options := options.copy())
        return options[:n]


class CopyImprovementEngine:
    """
    Analyses existing copy and suggests improvements.
    Checks compliance, readability, and marketing effectiveness.
    """

    def __init__(self):
        self._checker = RealTimeViolationChecker()
        self._api_key = os.getenv("ANTHROPIC_API_KEY", "")

    def analyse(self, headline: str, subhead: str,
                brand: str, category: str) -> dict:
        """
        Full copy analysis: compliance + readability + suggestions.
        """
        hl_check = self._checker.check(headline)
        sh_check  = self._checker.check(subhead)

        # Readability metrics (real calculations)
        hl_words   = len(headline.split())
        sh_words   = len(subhead.split())
        hl_chars   = len(headline)
        uppercase  = sum(1 for c in headline if c.isupper())
        uc_ratio   = uppercase / max(len(headline.replace(" ","")), 1)
        has_excl   = "!" in headline or "!" in subhead
        has_question = "?" in headline

        issues    = []
        strengths = []

        # Length check
        if hl_words > 8:
            issues.append({
                "issue":      "Headline too long",
                "detail":     f"{hl_words} words — aim for 6 or fewer",
                "severity":   "MEDIUM",
                "fix":        self._shorten_headline(headline),
            })
        elif hl_words <= 5:
            strengths.append("Headline length is punchy and concise")

        # Compliance
        if hl_check["violations"]:
            v = hl_check["violations"][0]
            issues.append({
                "issue":    f"Compliance violation: {v['category']}",
                "detail":   v.get("suggestion", ""),
                "severity": v["severity"],
                "fix":      None,
            })
        else:
            strengths.append("Headline passes NLP compliance check")

        if sh_check["violations"]:
            v = sh_check["violations"][0]
            issues.append({
                "issue":    f"Subhead violation: {v['category']}",
                "detail":   v.get("suggestion", ""),
                "severity": v["severity"],
                "fix":      None,
            })
        else:
            strengths.append("Subhead passes compliance check")

        # Exclamation marks
        if has_excl:
            issues.append({
                "issue":    "Exclamation mark present",
                "detail":   "Exclamation marks can trigger urgency flags. Use sparingly.",
                "severity": "LOW",
                "fix":      headline.replace("!", "").strip(),
            })

        # Uppercase ratio
        if uc_ratio > 0.7 and not headline.isupper():
            issues.append({
                "issue":    "Inconsistent capitalisation",
                "detail":   "Use all caps or title case consistently",
                "severity": "LOW",
                "fix":      headline.title(),
            })

        # Action word check
        action_words = ["discover","introducing","find","explore",
                        "meet","new","get","try","shop","see"]
        has_action = any(headline.lower().startswith(w) for w in action_words)
        if has_action:
            strengths.append("Starts with a strong action word")
        else:
            issues.append({
                "issue":    "No action word at start",
                "detail":   "Start with: Discover, Introducing, Find, Explore, New, Meet",
                "severity": "LOW",
                "fix":      f"Discover {headline}",
            })

        # Brand mention in subhead
        if brand.lower() in subhead.lower():
            strengths.append("Brand name present in subhead")

        # Overall score
        issue_penalty = sum(
            {"CRITICAL":25,"HIGH":15,"MEDIUM":8,"LOW":3}.get(i["severity"],0)
            for i in issues)
        copy_score = max(0, 100 - issue_penalty)
        grade = "A" if copy_score>=85 else "B" if copy_score>=70 else \
                "C" if copy_score>=55 else "D" if copy_score>=40 else "F"

        # Claude improvement if available
        improved = None
        if self._api_key and len(self._api_key) > 10 and issues:
            improved = self._claude_improve(headline, subhead, brand,
                                            category, issues)

        return {
            "headline":          headline,
            "subhead":           subhead,
            "copy_score":        copy_score,
            "grade":             grade,
            "issues":            issues,
            "strengths":         strengths,
            "hl_word_count":     hl_words,
            "sh_word_count":     sh_words,
            "is_compliant":      not hl_check["violations"] and not sh_check["violations"],
            "improved_version":  improved,
        }

    def _shorten_headline(self, headline: str) -> str:
        words = headline.split()
        # Keep first 6 meaningful words
        stop_words = {"a","an","the","and","or","but","in","on","at","to","for"}
        kept = []
        for w in words:
            if len(kept) >= 6: break
            kept.append(w)
        return " ".join(kept)

    def _claude_improve(self, headline, subhead, brand, category, issues) -> dict:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self._api_key)
            issue_list = "\n".join(f"- {i['issue']}: {i['detail']}"
                                   for i in issues[:3])
            prompt = f"""Improve this ad copy for {brand} ({category}).

Current headline: "{headline}"
Current subhead:  "{subhead}"

Issues to fix:
{issue_list}

Rules: max 8 words per headline, no violations, compliant, punchy.
Return ONLY JSON: {{"headline": "...", "subhead": "..."}}"""

            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                messages=[{"role":"user","content":prompt}])
            text = re.sub(r"```json|```","",resp.content[0].text.strip()).strip()
            return json.loads(text)
        except Exception:
            return None


class LayoutColourAdvisor:
    """
    Recommends layout, background colour, font, and pattern
    based on brand, category, and product type.
    Uses brand database + category knowledge.
    """

    _CATEGORY_ADVICE = {
        "Grocery": {
            "layouts":    ["Product Hero", "Centered Minimal"],
            "bg_style":   "Light, clean backgrounds. White or very light grey.",
            "fonts":      ["Poppins (Modern)", "Liberation Sans (Clean)"],
            "colours":    "Use brand primary. Avoid busy patterns.",
            "pattern":    "None or Subtle Dots",
            "tip":        "Put the product front and centre. Keep text minimal.",
        },
        "Fashion": {
            "layouts":    ["Full Bleed", "Centered Minimal", "Split Panel"],
            "bg_style":   "Dark or solid brand colours for luxury feel.",
            "fonts":      ["Lora (Elegant Serif)", "TeX Gyre Heros (Swiss)"],
            "colours":    "High contrast. Black/white with one accent colour.",
            "pattern":    "None or Minimal",
            "tip":        "Let the product breathe. Minimal copy. Strong typography.",
        },
        "Electronics": {
            "layouts":    ["Product Hero", "Bold Left"],
            "bg_style":   "Dark backgrounds make tech products pop.",
            "fonts":      ["TeX Gyre Heros (Swiss)", "Liberation Sans (Clean)"],
            "colours":    "Dark bg with white or accent text.",
            "pattern":    "Geometric or Subtle Dots",
            "tip":        "Show the product in action. Use technical language sparingly.",
        },
        "Beauty & Health": {
            "layouts":    ["Centered Minimal", "Product Hero"],
            "bg_style":   "Soft pastels or white. Clean and aspirational.",
            "fonts":      ["Lora (Elegant Serif)", "Poppins (Modern)"],
            "colours":    "Soft pinks, whites, creams. Never harsh colours.",
            "pattern":    "None or Soft Noise",
            "tip":        "Aspirational not clinical. Focus on feeling, not ingredients.",
        },
        "Finance": {
            "layouts":    ["Split Panel", "Bold Left"],
            "bg_style":   "Deep blue or navy. Trustworthy and professional.",
            "fonts":      ["Liberation Sans (Clean)", "Poppins (Modern)"],
            "colours":    "Navy, white, accent gold or green.",
            "pattern":    "Geometric or None",
            "tip":        "Lead with trust and simplicity. Avoid fine print in headline.",
        },
        "Alcohol": {
            "layouts":    ["Product Hero", "Full Bleed"],
            "bg_style":   "Dark, premium feel. Black or deep colour.",
            "fonts":      ["Lora (Elegant Serif)", "TeX Gyre Heros (Swiss)"],
            "colours":    "Dark with gold or white accents.",
            "pattern":    "None or Minimal",
            "tip":        "Must include Drinkaware. No language encouraging excess consumption.",
        },
        "General": {
            "layouts":    ["Product Hero", "Centered Minimal"],
            "bg_style":   "Use brand background colour.",
            "fonts":      ["Poppins (Modern)"],
            "colours":    "Follow brand colour palette.",
            "pattern":    "None",
            "tip":        "Keep it simple. Product + headline + brand logo.",
        },
    }

    def advise(self, brand: str, category: str, product: str) -> dict:
        cat_key = category if category in self._CATEGORY_ADVICE else "General"
        advice  = self._CATEGORY_ADVICE[cat_key]

        try:
            from brand_config import BRANDS
            bcfg = BRANDS.get(brand, {})
            brand_primary = bcfg.get("primary", "#333333")
            brand_bg      = bcfg.get("bg_default", "#F5F5F5")
        except Exception:
            brand_primary = "#333333"
            brand_bg      = "#F5F5F5"

        return {
            "recommended_layouts":  advice["layouts"],
            "background_advice":    advice["bg_style"],
            "recommended_fonts":    advice["fonts"],
            "colour_advice":        advice["colours"],
            "pattern_advice":       advice["pattern"],
            "top_tip":              advice["tip"],
            "brand_primary_colour": brand_primary,
            "brand_bg_colour":      brand_bg,
            "format_advice":        self._format_advice(category),
            "do_list":              self._dos(category),
            "dont_list":            self._donts(category),
        }

    @staticmethod
    def _format_advice(category: str) -> str:
        formats = {
            "Grocery":        "Square (1080×1080) works best for grocery ads on Instagram.",
            "Fashion":        "Stories (1080×1920) or Square for fashion. Avoid landscape.",
            "Electronics":    "Landscape (1200×628) for tech on Facebook. Square for Instagram.",
            "Beauty & Health":"Square or Stories. Vertical formats suit beauty content well.",
            "Finance":        "Landscape (1200×628) for LinkedIn. Square for social.",
            "Alcohol":        "Square or landscape. Stories if budget allows.",
        }
        return formats.get(category, "Square (1080×1080) is the safest starting format.")

    @staticmethod
    def _dos(category: str) -> list:
        dos = {
            "Grocery":   ["Show product clearly", "Use white or light background",
                          "Keep headline under 6 words", "Include brand logo"],
            "Fashion":   ["Use high contrast", "Let product breathe",
                          "Use elegant serif font", "Keep copy minimal"],
            "Alcohol":   ["Include Drinkaware logo", "Show product prominently",
                          "Use premium dark colours", "Keep message simple"],
            "Finance":   ["Use trustworthy navy/blue", "Keep copy factual",
                          "Include T&C notice", "Use clean sans-serif font"],
        }
        return dos.get(category, ["Keep it simple", "Show the product clearly",
                                   "Use brand colours", "Include brand logo"])

    @staticmethod
    def _donts(category: str) -> list:
        donts = {
            "Grocery":   ["Don't use health claims", "Don't add price in headline",
                          "Don't use busy backgrounds", "Don't use small font"],
            "Fashion":   ["Don't overcrowd with text", "Don't use clash colours",
                          "Don't use comic or playful fonts", "Don't add price"],
            "Alcohol":   ["Don't encourage excess drinking", "Don't target under-25s",
                          "Don't link alcohol to success", "Don't omit Drinkaware"],
            "Finance":   ["Don't make guaranteed return claims",
                          "Don't hide T&Cs in small print", "Don't over-promise"],
        }
        return donts.get(category, ["Don't use violation language",
                                     "Don't overcrowd the creative",
                                     "Don't use tiny fonts",
                                     "Don't omit brand logo"])


# ─────────────────────────────────────────────────────────────────────────────
#  STREAMLIT UI COMPONENT
#  Call render_ai_assistant() in the sidebar of dashboard.py
# ─────────────────────────────────────────────────────────────────────────────

def render_ai_assistant_sidebar(brand: str, category: str, product: str,
                                 headline: str, subhead: str):
    """
    Renders the AI assistant panel in the sidebar.
    Shows real-time feedback as user types.

    Call this inside `with st.sidebar:` in dashboard.py
    """
    import streamlit as st

    st.divider()
    st.markdown("## 🤖 AI Creative Assistant")
    st.caption("Real-time suggestions as you build")

    checker = RealTimeViolationChecker()

    # ── Real-time compliance light ───────────────────────────────────────────
    if headline or subhead:
        hl_check = checker.check(headline)
        sh_check  = checker.check(subhead)

        all_ok = hl_check["is_compliant"] and sh_check["is_compliant"]

        if all_ok:
            st.markdown(
                '<div style="background:#238636;color:white;padding:8px 12px;'
                'border-radius:6px;font-weight:700;font-size:13px;margin:4px 0">'
                '✅ Copy looks compliant</div>', unsafe_allow_html=True)
        else:
            # Show the most serious violation
            viols = (hl_check["violations"] or sh_check["violations"] or
                     hl_check["quick_violations"] or sh_check["quick_violations"])
            if viols:
                v = viols[0]
                sev = v.get("severity", "MEDIUM")
                col = {"CRITICAL":"#6e1818","HIGH":"#da3633",
                       "MEDIUM":"#9e6a03","LOW":"#1f6feb"}.get(sev,"#da3633")
                cat = v.get("category", v.get("category","Violation"))
                tip = v.get("suggestion","Remove violation language from copy.")
                st.markdown(
                    f'<div style="background:{col};color:white;padding:8px 12px;'
                    f'border-radius:6px;font-size:12px;margin:4px 0">'
                    f'⚠️ <b>{sev}</b>: {cat}<br>'
                    f'<span style="font-size:11px">{tip[:80]}</span>'
                    f'</div>', unsafe_allow_html=True)

    # ── Layout & colour advice ────────────────────────────────────────────────
    advisor = LayoutColourAdvisor()
    advice  = advisor.advise(brand, category, product)

    with st.expander("📐 Layout & Style Advice", expanded=True):
        st.markdown(f"**Best layouts:** {' · '.join(advice['recommended_layouts'])}")
        st.markdown(f"**Background:** {advice['background_advice']}")
        st.markdown(f"**Font:** {' · '.join(advice['recommended_fonts'])}")
        st.markdown(f"**Format:** {advice['format_advice']}")
        st.markdown(f"**💡 Top tip:** {advice['top_tip']}")

        dc1, dc2 = st.columns(2)
        with dc1:
            st.markdown("**✅ Do:**")
            for d in advice["do_list"]:
                st.markdown(f"· {d}")
        with dc2:
            st.markdown("**❌ Don't:**")
            for d in advice["dont_list"]:
                st.markdown(f"· {d}")

    # ── Headline suggestions ─────────────────────────────────────────────────
    with st.expander("✍️ Headline Suggestions", expanded=bool(product)):
        if not product:
            st.caption("Enter a product name in the Copy section to get suggestions")
        else:
            if st.button("🎯 Get Headline Ideas", use_container_width=True,
                         key="get_hl_suggestions"):
                with st.spinner("Generating…"):
                    engine = HeadlineSuggestionEngine()
                    result = engine.suggest(brand, category, product,
                                           headline, n=5)
                st.session_state["hl_suggestions"] = result

            if "hl_suggestions" in st.session_state:
                res = st.session_state["hl_suggestions"]
                st.caption(f"Source: {res['source']}")
                for s in res["suggestions"]:
                    col_ind = "🟢" if s["compliant"] else "🔴"
                    if st.button(f"{col_ind} {s['headline']}",
                                 key=f"hl_pick_{s['headline'][:20]}",
                                 use_container_width=True):
                        st.session_state["suggested_headline"] = s["headline"]
                        st.success(f"✅ Selected: {s['headline']}")
                        st.rerun()

    # ── Subhead suggestions ──────────────────────────────────────────────────
    with st.expander("💬 Subhead Suggestions"):
        if st.button("🎯 Get Subhead Ideas", use_container_width=True,
                     key="get_sh_suggestions"):
            with st.spinner("Generating…"):
                engine = SubheadSuggestionEngine()
                result = engine.suggest(brand, category, product, headline, n=4)
            st.session_state["sh_suggestions"] = result

        if "sh_suggestions" in st.session_state:
            res = st.session_state["sh_suggestions"]
            for s in res["suggestions"]:
                col_ind = "🟢" if s["compliant"] else "🔴"
                if st.button(f"{col_ind} {s['subhead']}",
                             key=f"sh_pick_{s['subhead'][:20]}",
                             use_container_width=True):
                    st.session_state["suggested_subhead"] = s["subhead"]
                    st.success(f"✅ Selected: {s['subhead']}")
                    st.rerun()

    # ── Copy analyser ────────────────────────────────────────────────────────
    with st.expander("🔬 Analyse My Copy"):
        if (headline or subhead) and st.button("📊 Analyse Copy",
                                                use_container_width=True,
                                                key="analyse_copy"):
            with st.spinner("Analysing…"):
                engine = CopyImprovementEngine()
                analysis = engine.analyse(headline, subhead, brand, category)
            st.session_state["copy_analysis"] = analysis

        if "copy_analysis" in st.session_state:
            a = st.session_state["copy_analysis"]
            grade_col = {"A":"#238636","B":"#1f6feb","C":"#9e6a03",
                         "D":"#da3633","F":"#6e1818"}.get(a["grade"],"#888")
            st.markdown(
                f'<div style="display:flex;gap:12px;align-items:center;margin:8px 0">'
                f'<div style="background:{grade_col};color:white;padding:8px 14px;'
                f'border-radius:8px;font-size:22px;font-weight:700">{a["grade"]}</div>'
                f'<div><b>Copy Score: {a["copy_score"]}/100</b><br>'
                f'<span style="font-size:11px;color:#8b949e">'
                f'Words: HL={a["hl_word_count"]} SH={a["sh_word_count"]}</span>'
                f'</div></div>', unsafe_allow_html=True)

            if a["strengths"]:
                for s in a["strengths"]:
                    st.markdown(f"✅ {s}")

            if a["issues"]:
                for issue in a["issues"]:
                    sev_col = {"CRITICAL":"#da3633","HIGH":"#ff8c00",
                               "MEDIUM":"#9e6a03","LOW":"#1f6feb"}.get(
                               issue["severity"],"#888")
                    st.markdown(
                        f'<div style="border-left:3px solid {sev_col};'
                        f'padding:6px 10px;margin:3px 0;background:#0d1117;'
                        f'border-radius:3px;font-size:12px">'
                        f'<b>{issue["issue"]}</b><br>'
                        f'{issue["detail"]}'
                        + (f'<br><i>Fix: {issue["fix"]}</i>' if issue.get("fix") else "")
                        + '</div>', unsafe_allow_html=True)

            if a.get("improved_version"):
                st.markdown("**🤖 Claude's improved version:**")
                iv = a["improved_version"]
                st.info(f"Headline: **{iv.get('headline','')}**\n\n"
                       f"Subhead: {iv.get('subhead','')}")
                if st.button("Use this version", key="use_improved"):
                    st.session_state["suggested_headline"] = iv.get("headline","")
                    st.session_state["suggested_subhead"]  = iv.get("subhead","")
                    st.rerun()