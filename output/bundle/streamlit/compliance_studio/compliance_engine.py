"""
compliance_engine.py  –  UPGRADED
Combines deterministic regex hard-fail rules (fast, reliable) with a
Claude LLM layer for nuanced, context-aware compliance reasoning.
"""

import re
import os
import json
import time
from datetime import datetime
from PIL import Image, ImageDraw
from collections import defaultdict

try:
    import anthropic
    _client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
    ANTHROPIC_AVAILABLE = True
except Exception:
    _client = None
    ANTHROPIC_AVAILABLE = False


COMPLIANCE_SYSTEM_PROMPT = """You are a Tesco brand-compliance auditor.
Your job is to identify advertising copy violations against Tesco's Appendix A & B guidelines.

Hard-fail categories (must report as HARD FAIL):
1. Forbidden claims: win, prize, competition, free, guarantee, money-back, warranty
2. Price callouts: £, $, price, cost, only, just, cheap, discount, sale, clearance, bargain, save, reduced, offer
3. Sustainability green-claims: eco, sustainable, green, carbon, zero waste, recyclable, biodegradable, organic, natural
4. Health/medical claims: healthy, healthier, cure, treatment, clinical, proven, doctor, wellness, boosts, enhances
5. Superlatives: best, perfect, ideal, ultimate, premium, luxury, superior, excellent, amazing, incredible, fantastic, #1, finest
6. Scarcity/urgency: limited, limited edition, while stocks last, last chance, ending soon, hurry
7. T&Cs: terms, conditions, t&c, conditions apply
8. Claim indicators: *, †, see below, survey claims, clinically proven
9. Charity: charity, donation, proceeds, fundraising
10. Exclusivity without tag: exclusive, only at (only allowed as the approved tag "Only at Tesco")

For alcohol specifically:
- Drinkaware logo must be present
- Forbidden: enjoy more, drink up, celebrate with, party, cheers, get drunk, intoxicated, binge, shots, booze

Allowed tags (exact wording only):
"Only at Tesco" | "Available at Tesco" | "Selected stores. While stocks last." | "Clubcard/app required. Ends DD/MM"

Return structured JSON only. No markdown fences."""


def _call_claude(prompt: str, max_tokens: int = 600) -> str:
    if not ANTHROPIC_AVAILABLE or not _client:
        return ""
    try:
        response = _client.messages.create(
            model="claude-opus-4-6",
            max_tokens=max_tokens,
            system=COMPLIANCE_SYSTEM_PROMPT,
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


class AdvancedComplianceEngine:

    def __init__(self):
        self.hard_rules = self.load_tesco_guidelines()
        self.violation_history = []

    # ─────────────────────────────────────────────────────────────────────────
    def load_tesco_guidelines(self) -> dict:
        return {
            "forbidden_claims": [
                # T&Cs
                "terms","conditions","t&c","t&cs","terms and conditions","conditions apply",
                # Competitions
                "win","winner","winning","prize","competition","contest","raffle","lottery",
                "gamble","bet","chance to win","free entry","draw",
                # Sustainability
                "eco","ecological","sustainable","sustainability","green","environmental",
                "planet","carbon","zero waste","recyclable","biodegradable","compostable",
                "organic","natural","environment","earth","eco-friendly","environmentally",
                # Charity
                "charity","donation","proceeds","fundraising",
                # Price
                "£","$","price","cost","only","just","cheap","inexpensive",
                "affordable","budget","discount","sale","clearance","bargain","deal",
                "offer","special offer","limited time","act now","buy now","save",
                "reduced","markdown","price drop","now only","was £","was $",
                # Guarantees
                "money-back","guarantee","warranty","refund","risk-free","guaranteed",
                "money back","satisfaction guaranteed",
                # Claim indicators
                "†","‡","§","¶","※","footnote","see below","survey claims",
                "clinical","proven","studies show","research shows","tests prove",
                "clinically proven","scientifically proven","doctor recommended",
                # Health
                "healthy","healthier","health","cure","treatment","medical","clinical",
                "doctor","physician","wellness","good for you","better for you",
                "improve","improves","performance","energy","nutritious","vitamin",
                "mineral","supplement","immune","detox","cleanse","boosts","enhances",
                # Superlatives
                "best","perfect","ideal","ultimate","premium","luxury","superior",
                "excellent","amazing","incredible","fantastic","number one","#1",
                "top","finest","greatest","most","leading","unbeatable","unmatched",
                # Free
                "free","freebie","complimentary","gratis","no charge","free of charge",
                # Scarcity
                "limited","limited edition","while stocks last","last chance","final",
                "ending soon","almost gone","selling fast","hurry","limited supply",
            ],
            "allowed_tags": [
                "Only at Tesco",
                "Available at Tesco",
                "Selected stores. While stocks last.",
                "Clubcard/app required. Ends DD/MM",
            ],
            "design_rules": {
                "min_font_sizes": {
                    "headline": 20,
                    "subhead": 12,
                    "drinkaware": 20,
                },
                "safe_zones": {
                    "9:16": {"top": 200, "bottom": 250},
                },
                "value_tile_rules": {"no_overlap": True},
                "packshot_rules": {
                    "max_count": 3,
                    "min_gap_double_density": 24,
                    "min_gap_single_density": 12,
                },
            },
            "alcohol_requirements": {
                "drinkaware_required": True,
                "drinkaware_min_size": 20,
                "colors": ["black", "white"],
                "sufficient_contrast": True,
                "forbidden_alcohol_terms": [
                    "enjoy more","drink up","celebrate with","party","cheers",
                    "get the party started","perfect for parties","social gathering",
                    "great for celebrations","festive","toast","get drunk","intoxicated",
                    "binge","alcoholic","alcoholism","liquor","spirits","booze",
                    "shots","chug","hammered","wasted","plastered","smashed",
                ],
            },
            "conditional_rules": {
                "clubcard_requires_end_date": True,
                "lep_position_right": True,
                "pinterest_requires_tag": True,
                "creative_links_to_tesco_requires_tag": True,
                "creative_links_to_tesco_requires_value_tile": True,
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC: TEXT COMPLIANCE
    # ─────────────────────────────────────────────────────────────────────────
    def check_text_compliance(
        self, headline: str, subhead: str, product_category: str = "general"
    ) -> dict:
        """
        Two-layer check:
        1. Fast deterministic regex (catches obvious keyword violations)
        2. Claude NLU layer (catches context-aware / phrasing violations)
        """
        issues, suggestions = [], []
        text = f"{headline} {subhead}".lower()

        # ── Layer 1: Deterministic regex ──────────────────────────────────────
        for term in self.hard_rules["forbidden_claims"]:
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(
                    f"HARD FAIL: '{term}' detected – {self.get_rule_description(term)}"
                )
                alt = self.get_compliant_alternative(term)
                if alt:
                    suggestions.append(f"Replace '{term}' with: '{alt}'")

        if product_category.lower() == "alcohol":
            for term in self.hard_rules["alcohol_requirements"]["forbidden_alcohol_terms"]:
                if term in text:
                    issues.append(
                        f"HARD FAIL (Alcohol): '{term}' is forbidden in alcohol advertising"
                    )

        # Check for claim indicators (asterisks)
        if "*" in f"{headline} {subhead}":
            issues.append("HARD FAIL: Asterisk (*) claim indicator detected – not permitted")

        # ── Layer 2: Claude NLU (context-aware) ───────────────────────────────
        llm_issues = self._llm_text_check(headline, subhead, product_category)
        for item in llm_issues:
            # Deduplicate with existing regex findings
            if not any(item.lower()[:30] in existing.lower() for existing in issues):
                issues.append(item)

        approved = len(issues) == 0
        if approved:
            suggestions.append("Copy is Tesco-compliant. No violations detected.")

        result = {
            "approved": approved,
            "issues": issues,
            "suggestions": suggestions,
            "compliance_score": max(0, 100 - len(issues) * 15),
            "checked_at": datetime.now().isoformat(),
            "llm_enhanced": ANTHROPIC_AVAILABLE,
        }
        self.violation_history.append(result)
        return result

    def _llm_text_check(
        self, headline: str, subhead: str, product_category: str
    ) -> list[str]:
        """
        Claude looks at the MEANING of the copy, not just keywords.
        Catches things like implied health claims, implied scarcity, euphemistic pricing.
        """
        if not ANTHROPIC_AVAILABLE or not _client:
            return []

        prompt = f"""
Analyse this Tesco advertising copy for compliance violations.

Headline     : {headline}
Subhead      : {subhead}
Product type : {product_category}

Focus on:
- IMPLIED violations (e.g. "Feel the difference" = implied health claim)
- CONTEXTUAL violations (e.g. "Time is running out" = scarcity without the word "limited")
- SUBTLE superlatives (e.g. "like nothing else" = uniqueness claim)
- Correct use of allowed tags (must be exact approved wording)

If COMPLIANT return: []
If violations found, return a JSON array of short strings, each starting with "HARD FAIL:" or "WARNING:".
Example: ["HARD FAIL: 'Feel the difference' is an implied health benefit claim"]
"""
        raw = _call_claude(prompt)
        if not raw or raw.startswith("__error__"):
            return []

        result = _safe_json(raw, None)
        if isinstance(result, list):
            return [str(s) for s in result if s]
        return []

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC: FULL CREATIVE AUDIT (text + design + visual)
    # ─────────────────────────────────────────────────────────────────────────
    def full_creative_audit(self, creative_data: dict, format_name: str) -> dict:
        text_result = self.check_text_compliance(
            creative_data.get("headline", ""),
            creative_data.get("subhead", ""),
            creative_data.get("product_category", "general"),
        )
        design_result = self.validate_creative_design(creative_data, format_name)

        # Overall
        passed = text_result["approved"] and design_result["valid"]
        score = (text_result["compliance_score"] + (100 if design_result["valid"] else 60)) // 2

        # Claude overall summary
        summary = self._llm_overall_summary(creative_data, text_result, design_result)

        return {
            "overall_assessment": {
                "passed": passed,
                "score": score,
                "summary": summary,
                "format": format_name,
                "checked_at": datetime.now().isoformat(),
                "llm_enhanced": ANTHROPIC_AVAILABLE,
            },
            "text_compliance": text_result,
            "design_compliance": design_result,
        }

    def _llm_overall_summary(self, creative_data, text_result, design_result) -> str:
        if not ANTHROPIC_AVAILABLE or not _client:
            return "Compliance check complete."

        issues_summary = "; ".join(
            text_result.get("issues", [])[:3] + design_result.get("issues", [])[:2]
        ) or "None detected"

        prompt = f"""
Write a 2-sentence compliance summary for a Tesco creative director.
Violations found: {issues_summary}
Product category: {creative_data.get("product_category", "General")}
Format: text only, no bullet points, professional tone.
"""
        raw = _call_claude(prompt, max_tokens=150)
        if raw and not raw.startswith("__error__"):
            return raw
        return "Compliance check complete. Review all flagged items before publishing."

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC: DESIGN VALIDATION
    # ─────────────────────────────────────────────────────────────────────────
    def validate_creative_design(self, creative_data: dict, format_name: str) -> dict:
        issues, warnings, hard_fails = [], [], []
        rules = self.hard_rules["design_rules"]

        # Packshot count
        packshots = creative_data.get("packshots", [])
        if len(packshots) > rules["packshot_rules"]["max_count"]:
            hard_fails.append(
                f"HARD FAIL: {len(packshots)} packshots uploaded – maximum is {rules['packshot_rules']['max_count']} (Appendix A)"
            )

        # Alcohol – Drinkaware
        if (
            creative_data.get("product_category", "").lower() == "alcohol"
            and not creative_data.get("include_drinkaware", False)
        ):
            hard_fails.append("HARD FAIL: Drinkaware logo required for alcohol campaigns (Appendix B)")

        # Clubcard end date
        if (
            creative_data.get("value_tile_type") == "Clubcard Price"
            and not creative_data.get("clubcard_end_date", "").strip()
        ):
            hard_fails.append("HARD FAIL: Clubcard Price tile requires an end date in DD/MM format (Appendix A)")

        # Safe zones for 9:16 formats
        if "1920" in format_name or "9:16" in format_name or "Stories" in format_name:
            warnings.append(
                "9:16 format: content must stay within safe zone (200px from top, 250px from bottom) (Appendix B)"
            )

        # Tesco tag required if links to Tesco
        if (
            creative_data.get("creative_links_to_tesco")
            and creative_data.get("tag_type", "None") == "None"
        ):
            hard_fails.append(
                "HARD FAIL: Creative links to Tesco but no Tesco tag selected (Appendix A & B)"
            )

        # Value tile overlap warning
        if creative_data.get("value_tile_type") not in (None, "None", ""):
            warnings.append(
                "Ensure no creative element overlaps the value tile (Appendix B hard fail)"
            )

        issues.extend(hard_fails)
        valid = len(hard_fails) == 0

        return {
            "valid": valid,
            "issues": issues,
            "warnings": warnings,
            "hard_fails": hard_fails,
            "checked_at": datetime.now().isoformat(),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC: SAFE ZONES
    # ─────────────────────────────────────────────────────────────────────────
    def check_safe_zones(self, format_name: str, element_positions: dict) -> dict:
        issues = []
        safe_zones = self.hard_rules["design_rules"]["safe_zones"]

        if "9:16" in format_name or "Stories" in format_name or "1920" in format_name:
            sz = safe_zones["9:16"]
            for element, position in element_positions.items():
                y_pos = position.get("y", 0)
                if y_pos < sz["top"]:
                    issues.append(
                        f"HARD FAIL: '{element}' is above the 9:16 safe zone top boundary ({sz['top']}px)"
                    )
                if y_pos > (1920 - sz["bottom"]):
                    issues.append(
                        f"HARD FAIL: '{element}' is below the 9:16 safe zone bottom boundary"
                    )

        return {"passed": len(issues) == 0, "issues": issues}

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC: HEADLINE/SUBHEAD ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    def analyze_headline_subhead(
        self, headline: str, subhead: str, product_category: str
    ) -> dict:
        result = self.check_text_compliance(headline, subhead, product_category)

        headline_issues = [i for i in result["issues"] if "headline" in i.lower() or headline.lower()[:10] in i.lower()]
        subhead_issues  = [i for i in result["issues"] if i not in headline_issues]

        # LLM readability score
        readability = self._llm_readability_score(headline, subhead)

        return {
            "headline_issues": headline_issues,
            "subhead_issues": subhead_issues,
            "recommendations": result["suggestions"],
            "compliance_score": result["compliance_score"],
            "readability_score": readability,
            "llm_enhanced": ANTHROPIC_AVAILABLE,
        }

    def _llm_readability_score(self, headline: str, subhead: str) -> dict:
        if not ANTHROPIC_AVAILABLE or not _client:
            return {"score": 75, "note": "LLM unavailable"}

        prompt = f"""
Score the readability and clarity of this Tesco ad copy.

Headline : {headline}
Subhead  : {subhead}

Return a single JSON object:
{{
  "score": <integer 0-100>,
  "reading_level": "Simple"|"Moderate"|"Complex",
  "note": "<one sentence>"
}}
"""
        raw = _call_claude(prompt, max_tokens=150)
        if raw and not raw.startswith("__error__"):
            result = _safe_json(raw, None)
            if isinstance(result, dict):
                return result
        return {"score": 75, "note": "Readability check unavailable"}

    # ─────────────────────────────────────────────────────────────────────────
    # VISUAL / IMAGE COMPLIANCE (NEW – Claude Vision)
    # ─────────────────────────────────────────────────────────────────────────
    def audit_creative_image(self, image_b64: str, media_type: str = "image/png") -> dict:
        """
        Uses Claude vision to audit an uploaded creative image for visual
        compliance issues (value tile overlap, logo presence, layout, etc.)
        """
        if not ANTHROPIC_AVAILABLE or not _client:
            return {
                "passed": None,
                "issues": [],
                "warnings": ["Visual audit unavailable – Claude API not configured"],
                "llm_enhanced": False,
            }

        try:
            response = _client.messages.create(
                model="claude-opus-4-6",
                max_tokens=800,
                system=COMPLIANCE_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": """Audit this Tesco advertising creative image for compliance.

Check:
1. Is any text element overlapping the value tile (price badge)?
2. Is the Tesco logo present and appears correctly sized?
3. For alcohol products: is a Drinkaware logo visible?
4. Are there any visible forbidden terms or asterisk (*) claim indicators?
5. Does the layout look appropriate for a social media banner?
6. Is text readable against the background (sufficient contrast)?

Return a JSON object:
{
  "passed": true|false,
  "issues": ["HARD FAIL: ...", ...],
  "warnings": ["WARNING: ...", ...],
  "positive_observations": ["...", ...]
}""",
                            },
                        ],
                    }
                ],
            )
            raw = response.content[0].text.strip()
            result = _safe_json(raw, None)
            if isinstance(result, dict) and "passed" in result:
                result["llm_enhanced"] = True
                return result
        except Exception as exc:
            pass

        return {
            "passed": None,
            "issues": [],
            "warnings": [f"Visual audit error – {str(exc) if 'exc' in dir() else 'unknown'}"],
            "llm_enhanced": True,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────
    def get_rule_description(self, term: str) -> str:
        descriptions = {
            "win": "Competition/prize claims are forbidden (Appendix B)",
            "free": "Free offers are forbidden – Appendix B hard fail",
            "best": "Superlative claims forbidden – Appendix B hard fail",
            "premium": "Superlative/quality claims forbidden",
            "guarantee": "Money-back/guarantee claims forbidden",
            "healthy": "Health claims forbidden – Appendix B hard fail",
            "discount": "Price callout – Appendix B hard fail",
            "sale": "Price/offer callout – Appendix B hard fail",
            "save": "Price saving callout – Appendix B hard fail",
            "organic": "Green/sustainability claim – Appendix B hard fail",
            "limited": "Scarcity/urgency claim – Appendix B hard fail",
            "£": "Price callout – Appendix B hard fail",
            "*": "Claim indicator (*) forbidden – Appendix B hard fail",
        }
        return descriptions.get(term.lower(), "Forbidden term per Appendix B")

    def get_compliant_alternative(self, term: str) -> str:
        alternatives = {
            "best": "trusted",
            "premium": "quality",
            "amazing": "distinctive",
            "incredible": "notable",
            "fantastic": "enjoyable",
            "perfect": "ideal for",
            "free": "(remove or use 'complimentary' only if genuinely no conditions)",
            "save": "(remove – price messaging not permitted)",
            "discount": "(remove – price messaging not permitted)",
            "healthy": "(remove – health claims not permitted)",
            "guarantee": "(remove – guarantee language not permitted)",
            "limited": "(remove – scarcity language not permitted)",
            "organic": "(remove unless it is part of a certified product name)",
            "eco": "(remove – green claims not permitted without approval)",
        }
        return alternatives.get(term.lower(), "")

    def _get_enhanced_claim_patterns(self):
        return [
            (r'\d+\s*%\s*(?:off|discount|saving)', "Percentage discount claim – Appendix B hard fail"),
            (r'was\s+[\$£€]?\d', "Price comparison 'was/now' format – Appendix B hard fail"),
            (r'only\s+[\$£€]?\d', "Price callout with 'only' – Appendix B hard fail"),
            (r'[\$£€]\d+[\.,]?\d*', "Currency amount detected – price callout hard fail"),
            (r'clinically\s+(?:proven|tested)', "Clinical claim – Appendix B hard fail"),
            (r'no\s+\d+\s+in', "Ranking claim – Appendix B hard fail"),
            (r'doctor[s]?\s+(?:recommend|approved)', "Medical endorsement – hard fail"),
        ]
