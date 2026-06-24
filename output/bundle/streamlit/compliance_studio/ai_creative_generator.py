"""
ai_creative_generator.py  –  UPGRADED
Uses the Anthropic Claude API for real LLM-powered copy generation,
performance prediction, and creative suggestions.
"""

import os
import json
import re
from datetime import datetime

# ── Anthropic client (lazy-init so import never crashes) ──────────────────────
try:
    import anthropic
    _client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
    ANTHROPIC_AVAILABLE = True
except Exception:
    _client = None
    ANTHROPIC_AVAILABLE = False


# ── Tesco brand constraint block injected into every system prompt ────────────
TESCO_SYSTEM_PROMPT = """You are a senior Tesco brand-compliance specialist and retail advertising copywriter.
You MUST follow Tesco's Appendix A & B creative guidelines at all times.

HARD FAIL – forbidden terms you can NEVER use:
• Price/value words: £, $, price, cost, only, just, cheap, affordable, discount, sale, clearance,
  bargain, deal, offer, special offer, limited time, save, reduced, free, freebie
• Competition/prize words: win, winner, prize, competition, contest, raffle, lottery, jackpot
• Sustainability green-claims: eco, ecological, sustainable, green, carbon, zero waste,
  recyclable, biodegradable, organic, natural (unless part of a product name)
• Health/medical claims: healthy, healthier, cure, treatment, clinical, proven, doctor,
  wellness, boosts, enhances, improves performance, nutritious, detox, cleanse
• Superlatives: best, perfect, ideal, ultimate, premium, luxury, superior, excellent, amazing,
  incredible, fantastic, number one, #1, finest, greatest, most, unbeatable, unmatched
• Guarantee: money-back, guarantee, warranty, refund, risk-free, guaranteed
• Scarcity: limited, limited edition, while stocks last, last chance, ending soon, hurry
• Claim indicators: *, †, ‡, footnote, see below, survey claims, clinically proven
• T&Cs: terms, conditions, t&c, t&cs, conditions apply

ALLOWED exclusivity tags (exact wording only):
  "Only at Tesco" | "Available at Tesco" | "Selected stores. While stocks last."

Alcohol creatives additionally require:
  - Drinkaware logo mentioned/included
  - No: enjoy more, drink up, celebrate with, party, cheers, get drunk, intoxicated, binge, shots

Always return valid JSON unless told otherwise. Never include markdown fences in JSON responses."""


def _call_claude(prompt: str, system: str = TESCO_SYSTEM_PROMPT, max_tokens: int = 800) -> str:
    """Core helper – calls Claude API, returns plain text response."""
    if not ANTHROPIC_AVAILABLE or not _client:
        return ""
    try:
        response = _client.messages.create(
            model="claude-opus-4-6",
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as exc:
        return f"__error__:{exc}"


def _safe_json(text: str, fallback):
    """Parse JSON, return fallback on failure."""
    try:
        cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
        return json.loads(cleaned)
    except Exception:
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
class AICreativeSuggestor:

    # ── Copy improvements ────────────────────────────────────────────────────
    def suggest_copy_improvements(
        self, headline: str, subhead: str, product_type: str
    ) -> list[str]:
        """
        Ask Claude to review the copy and return Tesco-compliant improvement
        suggestions.  Falls back to rule-based hints when the API is unavailable.
        """
        if not ANTHROPIC_AVAILABLE or not _client:
            return self._fallback_suggestions(headline, subhead, product_type)

        prompt = f"""
Analyse this Tesco advertising copy for brand-compliance and creative quality.

Product type : {product_type}
Headline     : {headline}
Subhead      : {subhead}

Return a JSON array of up to 6 short, actionable suggestion strings.
Each suggestion must start with an action verb (e.g. "Replace", "Shorten", "Add", "Remove", "Consider").
Flag every Tesco hard-fail violation first.
Example format:
["Replace 'best' – superlative is a hard fail", "Shorten headline to under 60 characters", "Add product benefit in subhead"]
"""
        raw = _call_claude(prompt)
        if raw.startswith("__error__"):
            return self._fallback_suggestions(headline, subhead, product_type)

        result = _safe_json(raw, None)
        if isinstance(result, list):
            return result
        # If Claude returned prose instead of JSON, wrap each line
        return [line.strip("•- ") for line in raw.splitlines() if line.strip()][:6]

    def _fallback_suggestions(self, headline, subhead, product_type) -> list[str]:
        tips = []
        text = f"{headline} {subhead}".lower()
        forbidden = ["best","free","sale","guarantee","win","healthy","premium","amazing","only","cheap"]
        for word in forbidden:
            if re.search(r'\b' + word + r'\b', text):
                tips.append(f"Remove '{word}' – Tesco hard-fail term (Appendix B)")
        if len(headline) > 60:
            tips.append("Shorten headline to under 60 characters for better readability")
        if len(headline) < 10:
            tips.append("Expand headline – current text is too short to communicate value")
        if product_type.lower() == "alcohol":
            tips.append("Ensure Drinkaware lock-up is included (Appendix B hard fail)")
        tips.append("Use active, benefit-focused language without making clinical claims")
        tips.append("Ensure strong colour contrast between text and background (WCAG AA)")
        return tips

    # ── Headline generation ──────────────────────────────────────────────────
    def generate_headline_variants(
        self, product_name: str, product_type: str, tone: str = "confident"
    ) -> list[dict]:
        """
        Generate 3 brand-compliant headline variants via Claude.
        Returns list of dicts: {headline, subhead, rationale, score}
        """
        if not ANTHROPIC_AVAILABLE or not _client:
            return self._fallback_headlines(product_name, product_type)

        prompt = f"""
Generate 3 distinct Tesco-compliant advertising headline + subhead pairs for:

Product : {product_name}
Category: {product_type}
Tone    : {tone}

Rules:
- Headline: max 55 characters, no hard-fail terms
- Subhead : max 80 characters, no hard-fail terms
- Avoid ALL forbidden Tesco terms listed in your guidelines
- Do NOT use: best, free, save, only, amazing, premium, guarantee, win, healthy, discount

Return a JSON array of exactly 3 objects:
[
  {{
    "headline": "...",
    "subhead": "...",
    "rationale": "why this works for {product_type}",
    "predicted_engagement": <integer 60-95>
  }}
]
"""
        raw = _call_claude(prompt)
        if raw.startswith("__error__"):
            return self._fallback_headlines(product_name, product_type)

        result = _safe_json(raw, None)
        if isinstance(result, list) and len(result) >= 1:
            return result[:3]
        return self._fallback_headlines(product_name, product_type)

    def _fallback_headlines(self, product_name, product_type) -> list[dict]:
        return [
            {
                "headline": f"Discover {product_name}",
                "subhead": f"Find it now in our {product_type} range",
                "rationale": "Neutral discovery framing – no hard-fail terms",
                "predicted_engagement": 72,
            },
            {
                "headline": f"Try {product_name} today",
                "subhead": "Available across selected formats at Tesco",
                "rationale": "Call-to-action without promotional claims",
                "predicted_engagement": 68,
            },
        ]

    # ── Performance prediction ───────────────────────────────────────────────
    def predict_performance(
        self, creative_elements: dict, target_platform: str
    ) -> dict:
        """
        Ask Claude to estimate performance metrics for the creative.
        Returns: engagement_score, click_through_prediction, conversion_likelihood, grade
        """
        if not ANTHROPIC_AVAILABLE or not _client:
            return self._fallback_performance(creative_elements)

        headline        = creative_elements.get("headline", "")
        subhead         = creative_elements.get("subhead", "")
        has_value_tile  = creative_elements.get("has_value_tile", False)
        product_cat     = creative_elements.get("product_category", "General")

        prompt = f"""
Evaluate the predicted performance of this Tesco retail media creative.

Headline      : {headline}
Subhead       : {subhead}
Value tile    : {"Yes" if has_value_tile else "No"}
Platform      : {target_platform}
Product cat   : {product_cat}

Score the creative on retail advertising best practices (NOT on compliance – assume compliant).
Consider: clarity, relevance, emotional resonance, platform fit, call-to-action strength.

Return a single JSON object:
{{
  "engagement_score": <integer 50-98>,
  "click_through_prediction": "<float>%",
  "conversion_likelihood": "High"|"Medium"|"Low",
  "performance_grade": "A+"|"A"|"A-"|"B+"|"B"|"B-"|"C",
  "strength": "<one sentence: what works best>",
  "weakness": "<one sentence: biggest area to improve>"
}}
"""
        raw = _call_claude(prompt, max_tokens=400)
        if raw.startswith("__error__"):
            return self._fallback_performance(creative_elements)

        result = _safe_json(raw, None)
        if isinstance(result, dict) and "engagement_score" in result:
            return result
        return self._fallback_performance(creative_elements)

    def _fallback_performance(self, creative_elements) -> dict:
        headline = creative_elements.get("headline", "")
        score = 70
        if 20 <= len(headline) <= 55:
            score += 8
        if creative_elements.get("has_value_tile"):
            score += 7
        score = min(score, 95)
        grade = (
            "A+" if score >= 93 else "A" if score >= 88 else "A-" if score >= 83
            else "B+" if score >= 78 else "B" if score >= 73 else "B-" if score >= 68
            else "C"
        )
        return {
            "engagement_score": score,
            "click_through_prediction": f"{score / 10:.1f}%",
            "conversion_likelihood": "High" if score > 85 else "Medium" if score > 70 else "Low",
            "performance_grade": grade,
            "strength": "Clear product focus",
            "weakness": "Headline length could be optimised",
        }

    # ── Trending designs ─────────────────────────────────────────────────────
    def get_trending_designs(self, product_category: str) -> dict:
        """
        Ask Claude for current design/copy trend recommendations for the category.
        """
        if not ANTHROPIC_AVAILABLE or not _client:
            return self._fallback_trends(product_category)

        prompt = f"""
What are the current retail advertising creative trends for the {product_category} category
on social media (Instagram, Facebook)?

Return a JSON object:
{{
  "styles": ["<style 1>", "<style 2>", "<style 3>"],
  "colors": ["<color trend 1>", "<color trend 2>"],
  "copy_tone": "<recommended tone>",
  "recommendations": ["<recommendation 1>", "<recommendation 2>", "<recommendation 3>"]
}}
Keep each item under 10 words.
"""
        raw = _call_claude(prompt, max_tokens=400)
        if raw.startswith("__error__"):
            return self._fallback_trends(product_category)

        result = _safe_json(raw, None)
        if isinstance(result, dict):
            return result
        return self._fallback_trends(product_category)

    def _fallback_trends(self, product_category) -> dict:
        defaults = {
            "Alcohol": {
                "styles": ["Premium aesthetic", "Clean layouts", "Sophisticated typography"],
                "colors": ["Deep tones", "Metallic accents", "Rich backgrounds"],
                "copy_tone": "Sophisticated and confident",
                "recommendations": ["Focus on quality messaging", "Use premium lifestyle imagery", "Avoid health claims"],
            },
            "Electronics": {
                "styles": ["Modern tech", "Feature highlights", "Minimalist"],
                "colors": ["Blue gradients", "Dark themes", "Metallic accents"],
                "copy_tone": "Informative and direct",
                "recommendations": ["Highlight specifications clearly", "Use lifestyle context", "Focus on convenience"],
            },
        }
        return defaults.get(
            product_category,
            {
                "styles": ["Professional", "Clean", "Engaging"],
                "colors": ["Brand colours", "High contrast", "Accessible"],
                "copy_tone": "Clear and confident",
                "recommendations": ["Clear value propositions", "Strong visual hierarchy", "Consistent brand voice"],
            },
        )

    # ── Compliance-aware rewrite ─────────────────────────────────────────────
    def rewrite_compliant(
        self, headline: str, subhead: str, product_type: str
    ) -> dict:
        """
        Takes potentially non-compliant copy and returns a compliant rewrite.
        Returns: {headline, subhead, changes_made}
        """
        if not ANTHROPIC_AVAILABLE or not _client:
            return {"headline": headline, "subhead": subhead, "changes_made": ["API unavailable – manual review required"]}

        prompt = f"""
The following Tesco advertising copy may contain compliance violations.
Rewrite it to be 100% compliant with Tesco Appendix A & B guidelines.

Original headline : {headline}
Original subhead  : {subhead}
Product type      : {product_type}

Rules:
- Keep the intent and product reference
- Remove or replace ALL hard-fail terms
- Headline: max 55 characters
- Subhead : max 80 characters

Return a JSON object:
{{
  "headline": "<compliant headline>",
  "subhead": "<compliant subhead>",
  "changes_made": ["<change 1>", "<change 2>"]
}}
"""
        raw = _call_claude(prompt)
        if raw.startswith("__error__"):
            return {"headline": headline, "subhead": subhead, "changes_made": ["API error – manual review required"]}

        result = _safe_json(raw, None)
        if isinstance(result, dict) and "headline" in result:
            return result
        return {"headline": headline, "subhead": subhead, "changes_made": ["Parse error – review manually"]}

    # ── Generate creative variations ─────────────────────────────────────────
    def generate_variations(
        self, headline: str, subhead: str, packshot, value_tile_type: str
    ) -> list[dict]:
        """
        Generate 3 layout/style variation recommendations via Claude.
        """
        if not ANTHROPIC_AVAILABLE or not _client:
            return self._fallback_variations()

        prompt = f"""
Recommend 3 distinct creative layout variations for a Tesco social media banner.

Inputs:
- Headline      : {headline}
- Subhead       : {subhead}
- Value tile    : {value_tile_type}

For each variation suggest: template name, visual style, colour approach, layout emphasis.
Return a JSON array of 3 objects:
[
  {{
    "id": "var_1",
    "template": "<name>",
    "description": "<one sentence>",
    "confidence_score": <65-95>,
    "performance_prediction": <65-95>,
    "ai_suggestions": ["<tip 1>", "<tip 2>", "<tip 3>"]
  }}
]
"""
        raw = _call_claude(prompt)
        if raw.startswith("__error__"):
            return self._fallback_variations()

        result = _safe_json(raw, None)
        if isinstance(result, list):
            for i, v in enumerate(result):
                v.setdefault("id", f"var_{i+1}")
                v.setdefault("timestamp", datetime.now().isoformat())
            return result[:3]
        return self._fallback_variations()

    def _fallback_variations(self) -> list[dict]:
        return [
            {"id":"var_1","template":"Minimal","description":"Clean layout, product centred","confidence_score":82,"performance_prediction":80,"ai_suggestions":["Use generous whitespace","Limit to 2 font weights","High product-to-canvas ratio"],"timestamp":datetime.now().isoformat()},
            {"id":"var_2","template":"Bold","description":"Strong headline, full-bleed product","confidence_score":78,"performance_prediction":77,"ai_suggestions":["Ensure text contrast ratio ≥ 4.5:1","Scale headline to 30 % of canvas height","Keep subhead concise"],"timestamp":datetime.now().isoformat()},
            {"id":"var_3","template":"Split","description":"50/50 image and copy layout","confidence_score":74,"performance_prediction":73,"ai_suggestions":["Left-align copy block","Use brand blue (#00539F) background","Place value tile bottom-right"],"timestamp":datetime.now().isoformat()},
        ]
