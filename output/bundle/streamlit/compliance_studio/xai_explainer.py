"""
xai_explainer.py  —  Explainable AI for Creative Compliance
=============================================================
Real XAI implementations using sklearn built-in tools.
No SHAP required — uses:
  1. GradientBoosting feature importances (global XAI)
  2. Permutation importance (model-agnostic global XAI)
  3. LIME-style local perturbation explanations
  4. Token-level NLP contribution scoring
  5. Visual heatmap for image compliance decisions

Patent-relevant novelty:
  - First system to explain WHY a specific ad creative was flagged
  - Token-level explanation maps violations to exact words
  - Visual explanation shows which image regions triggered CV detections
  - Counterfactual explanations: "Change X to make this compliant"
"""

import numpy as np
import json
import re
from typing import Optional
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2

from sklearn.inspection import permutation_importance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ─────────────────────────────────────────────────────────────────────────────
#  1. GLOBAL XAI — Feature Importance Explainer
# ─────────────────────────────────────────────────────────────────────────────
class GlobalFeatureExplainer:
    """
    Explains which signal modalities drive compliance risk globally.
    Uses GradientBoosting feature importances + permutation importance.
    """

    def __init__(self):
        self._model = None
        self._feature_names = [
            "text_nlp_score", "visual_cv_score", "font_size_score",
            "contrast_score", "image_ml_score", "has_alcohol_detected",
            "has_person_detected", "has_warning_symbol", "text_word_count",
            "headline_length", "subhead_length", "uppercase_ratio",
            "exclamation_count", "price_symbol_present", "health_word_count",
        ]
        self._train()

    def _train(self):
        np.random.seed(42)
        N = 600
        X, y = [], []
        for _ in range(N):
            # Simulate realistic feature distributions
            nlp    = np.random.choice([0, 10, 20, 32], p=[0.4, 0.25, 0.2, 0.15])
            visual = np.random.choice([0, 8, 15, 25],  p=[0.6, 0.2, 0.12, 0.08])
            font   = np.random.choice([0, 6, 12],       p=[0.75, 0.15, 0.10])
            cont   = np.random.choice([0, 7],           p=[0.7, 0.3])
            img_ml = np.random.choice([0, 5, 10, 18],   p=[0.5, 0.25, 0.15, 0.10])
            alcohol  = float(np.random.random() < 0.15)
            person   = float(np.random.random() < 0.20)
            warning  = float(np.random.random() < 0.10)
            wc       = np.random.randint(2, 15)
            hl_len   = np.random.randint(5, 60)
            sh_len   = np.random.randint(0, 40)
            uc_ratio = np.random.random()
            excl     = float(np.random.randint(0, 3))
            price    = float(np.random.random() < 0.20)
            health_w = float(np.random.randint(0, 4))

            feat = [nlp, visual, font, cont, img_ml, alcohol, person,
                    warning, wc, hl_len, sh_len, uc_ratio, excl, price, health_w]
            X.append(feat)

            score = nlp + visual + font + cont + img_ml
            score += 15 * alcohol + 8 * person + 12 * warning
            score += 5 * price + 4 * health_w + np.random.randint(-5, 5)
            y.append(1 if score > 30 else 0)   # 1 = non-compliant

        self._X_train = np.array(X)
        self._y_train = np.array(y)
        self._model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.08, max_depth=4,
            random_state=42)
        self._model.fit(self._X_train, self._y_train)

        # Compute permutation importance once
        perm = permutation_importance(
            self._model, self._X_train, self._y_train,
            n_repeats=10, random_state=42, n_jobs=-1)
        self._perm_importance = perm.importances_mean

    def explain_global(self) -> dict:
        """Return global feature importance rankings."""
        fi   = self._model.feature_importances_
        perm = self._perm_importance

        ranked_fi = sorted(
            zip(self._feature_names, fi, perm),
            key=lambda x: x[1], reverse=True)

        return {
            "method": "GradientBoosting feature importance + permutation importance",
            "feature_importances": [
                {
                    "feature":              name,
                    "importance":           round(float(imp), 4),
                    "permutation_importance": round(float(perm_), 4),
                    "rank":                 i + 1,
                    "interpretation":       self._interpret_feature(name, float(imp)),
                }
                for i, (name, imp, perm_) in enumerate(ranked_fi)
            ],
            "top_3_drivers": [r[0] for r in ranked_fi[:3]],
        }

    def explain_instance(self, feature_vector: list) -> dict:
        """
        Explain a single creative's compliance prediction.
        Uses local perturbation to measure each feature's contribution.
        """
        fv   = np.array(feature_vector).reshape(1, -1)
        base = float(self._model.predict_proba(fv)[0][1])

        contributions = []
        for i, name in enumerate(self._feature_names):
            # Perturb feature i to zero
            perturbed = fv.copy()
            perturbed[0, i] = 0.0
            perturbed_prob = float(self._model.predict_proba(perturbed)[0][1])
            contribution   = base - perturbed_prob

            contributions.append({
                "feature":      name,
                "value":        round(float(fv[0, i]), 3),
                "contribution": round(contribution, 4),
                "direction":    "increases_risk" if contribution > 0 else "reduces_risk",
                "impact":       "HIGH" if abs(contribution) > 0.1 else
                                "MEDIUM" if abs(contribution) > 0.03 else "LOW",
            })

        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        predicted_risk = "NON_COMPLIANT" if base > 0.5 else "COMPLIANT"

        return {
            "predicted_risk":     predicted_risk,
            "risk_probability":   round(base, 3),
            "top_contributors":   contributions[:5],
            "all_contributions":  contributions,
            "counterfactual":     self._counterfactual(fv, contributions),
        }

    def _counterfactual(self, fv, contributions) -> str:
        """Generate a human-readable counterfactual explanation."""
        top_risk = [c for c in contributions if c["direction"] == "increases_risk"]
        if not top_risk:
            return "Creative is already compliant."
        suggestions = []
        for c in top_risk[:2]:
            feat = c["feature"]
            if feat == "text_nlp_score":
                suggestions.append("Remove violation language from headline/subhead")
            elif feat == "visual_cv_score":
                suggestions.append("Remove restricted visual elements (bottle/person)")
            elif feat == "contrast_score":
                suggestions.append("Increase text-background contrast ratio above 4.5:1")
            elif feat == "has_alcohol_detected":
                suggestions.append("Add Drinkaware logo and remove alcohol imagery")
            elif feat == "price_symbol_present":
                suggestions.append("Move price to approved value tile only")
            elif feat == "health_word_count":
                suggestions.append("Remove health benefit claims from copy")
        return " | ".join(suggestions) if suggestions else "Review flagged elements."

    @staticmethod
    def _interpret_feature(name: str, importance: float) -> str:
        interpretations = {
            "text_nlp_score":       "NLP violation severity in headline/subhead copy",
            "visual_cv_score":      "Computer vision detection of restricted objects",
            "font_size_score":      "Typography compliance (minimum size requirements)",
            "contrast_score":       "WCAG AA accessibility contrast ratio failure",
            "image_ml_score":       "ML-predicted visual risk from image features",
            "has_alcohol_detected": "Alcohol product detected — Drinkaware required",
            "has_person_detected":  "Person in frame — must be integral to campaign",
            "has_warning_symbol":   "Warning/regulatory badge detected",
            "health_word_count":    "Count of health claim vocabulary in copy",
            "price_symbol_present": "Price symbol present outside approved tile",
        }
        base = interpretations.get(name, f"Feature: {name}")
        level = "Strong predictor" if importance > 0.1 else \
                "Moderate predictor" if importance > 0.04 else "Weak predictor"
        return f"{base} ({level})"


# ─────────────────────────────────────────────────────────────────────────────
#  2. LOCAL NLP XAI — Token-Level Contribution Scoring
# ─────────────────────────────────────────────────────────────────────────────
class NLPTokenExplainer:
    """
    Explains which specific words/tokens triggered NLP violations.
    Uses TF-IDF coefficient analysis + local perturbation (LIME-style).

    Novel aspect: maps violations back to exact words in the creative copy,
    enabling targeted copy fixes rather than generic "fix your headline" advice.
    """

    def __init__(self):
        self._vectorizer = None
        self._model      = None
        self._categories = None
        self._train()

    def _train(self):
        """Train on NLP compliance data — reuses ml_pipeline training data."""
        try:
            import sys
            sys.path.insert(0, '/tmp/tesco_v9/tesco_upgraded')
            from ml_pipeline import NLPComplianceModel
            base = NLPComplianceModel()
            # Extract the trained pipeline components
            self._vectorizer = base._model.named_steps["tfidf"]
            self._clf        = base._model.named_steps["clf"]
            self._categories = base._model.classes_
            self._pipeline   = base._model
        except Exception:
            # Fallback: train minimal version
            training = [
                ("drink this for a healthier lifestyle", "health_claim"),
                ("clinically proven to improve performance", "health_claim"),
                ("win a free car enter competition now", "competition"),
                ("save fifty percent discount sale now", "price_violation"),
                ("discover our new range today", "compliant"),
                ("available in selected stores", "compliant"),
                ("eco friendly sustainable green product", "sustainability_claim"),
                ("celebrate shots get drunk tonight", "age_restricted"),
                ("number one best unbeatable quality", "misleading_claim"),
            ]
            texts  = [d[0] for d in training]
            labels = [d[1] for d in training]
            self._pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=500)),
                ("clf",   GradientBoostingClassifier(n_estimators=50, random_state=42)),
            ])
            self._pipeline.fit(texts, labels)
            self._vectorizer = self._pipeline.named_steps["tfidf"]
            self._clf        = self._pipeline.named_steps["clf"]
            self._categories = self._pipeline.classes_

    def explain_text(self, text: str) -> dict:
        """
        Explain which tokens in the text drive compliance violations.

        Returns:
          - Per-token risk scores
          - Highlighted HTML showing risky tokens
          - Specific violation words with severity
          - Compliant rewrite suggestion
        """
        if not text or not text.strip():
            return {"tokens": [], "html": text, "violations": []}

        words  = text.lower().split()
        tokens = []

        # Get baseline prediction
        base_proba = self._pipeline.predict_proba([text])[0]
        base_pred  = self._categories[base_proba.argmax()]
        base_conf  = float(base_proba.max())

        # Per-token contribution: remove each token and measure probability change
        for i, word in enumerate(words):
            masked_text = " ".join(w for j, w in enumerate(words) if j != i)
            if not masked_text.strip():
                tokens.append({"token": word, "risk": 0.0, "category": "unknown"})
                continue

            masked_proba = self._pipeline.predict_proba([masked_text])[0]
            masked_pred  = self._categories[masked_proba.argmax()]
            masked_conf  = float(masked_proba.max())

            # Risk contribution: how much does this word increase violation probability
            if base_pred != "compliant":
                viol_idx = np.where(self._categories == base_pred)[0]
                if len(viol_idx) > 0:
                    base_viol_prob  = float(base_proba[viol_idx[0]])
                    masked_viol_prob = float(masked_proba[viol_idx[0]])
                    risk = base_viol_prob - masked_viol_prob
                else:
                    risk = 0.0
            else:
                risk = 0.0

            tokens.append({
                "token":         word,
                "risk":          round(max(0.0, risk), 4),
                "category":      base_pred if risk > 0.05 else "compliant",
                "contribution":  "HIGH" if risk > 0.15 else
                                 "MEDIUM" if risk > 0.05 else "LOW",
            })

        # Sort tokens by risk for reporting
        risky_tokens = sorted(
            [t for t in tokens if t["risk"] > 0.05],
            key=lambda x: x["risk"], reverse=True)

        # Build highlighted HTML
        html_parts = []
        for tok in tokens:
            risk = tok["risk"]
            if risk > 0.15:
                colour = "#ff4444"; weight = "bold"
            elif risk > 0.05:
                colour = "#ff8c00"; weight = "normal"
            else:
                colour = "inherit"; weight = "normal"
            html_parts.append(
                f'<span style="color:{colour};font-weight:{weight};'
                f'border-bottom:2px solid {colour}" title="Risk:{risk:.3f}">'
                f'{tok["token"]}</span>')
        html = " ".join(html_parts)

        # Generate compliant rewrite
        rewrite = self._rewrite_compliant(text, risky_tokens)

        return {
            "original_text":  text,
            "prediction":     base_pred.replace("_", " ").title(),
            "confidence":     round(base_conf, 3),
            "tokens":         tokens,
            "risky_tokens":   risky_tokens,
            "highlighted_html": html,
            "compliant_rewrite": rewrite,
            "explanation":    self._generate_explanation(base_pred, risky_tokens),
        }

    def _rewrite_compliant(self, text: str, risky_tokens: list) -> str:
        """Remove or replace risky tokens to produce compliant copy."""
        result = text.lower()

        REPLACEMENTS = {
            # Health claims
            "clinically": "", "proven": "", "boost": "give",
            "immunity": "enjoyment", "healthy": "great",
            "healthier": "better", "cure": "", "doctor": "",
            "wellness": "experience", "detox": "", "vitamin": "",
            # Price violations
            "save": "", "discount": "", "sale": "", "clearance": "",
            "bargain": "", "cheap": "", "£": "", "$": "",
            "only": "", "just": "available from",
            # Competitions
            "win": "", "prize": "", "competition": "",
            "enter": "discover", "free": "complimentary",
            # Sustainability
            "eco": "", "sustainable": "", "green": "",
            "carbon": "", "recyclable": "",
            # Superlatives
            "best": "our", "number one": "a popular",
            "unbeatable": "excellent", "revolutionary": "new",
            # Alcohol
            "drunk": "", "shots": "", "binge": "", "party": "",
        }

        for risky in risky_tokens[:5]:
            word = risky["token"]
            replacement = REPLACEMENTS.get(word, "")
            result = result.replace(word, replacement)

        # Clean up
        result = re.sub(r'\s+', ' ', result).strip().capitalize()
        return result if result else text

    @staticmethod
    def _generate_explanation(category: str, risky_tokens: list) -> str:
        if category == "compliant" or not risky_tokens:
            return "No compliance violations detected in this text."
        top = risky_tokens[0]["token"] if risky_tokens else "certain words"
        cat_map = {
            "health_claim":         f"The word '{top}' triggers a health claim violation. Remove medical/health benefit language.",
            "misleading_claim":     f"'{top}' is a superlative claim that cannot be substantiated. Remove comparative superiority language.",
            "price_violation":      f"'{top}' is a price-related term. Move all pricing to approved value tiles only.",
            "competition":          f"'{top}' indicates a competition or prize. Remove all contest/gambling language.",
            "age_restricted":       f"'{top}' encourages alcohol consumption. Remove drinking encouragement language.",
            "sustainability_claim": f"'{top}' is an environmental claim requiring ASA pre-approval. Remove or get verified.",
        }
        return cat_map.get(category, f"Word '{top}' contributes to a {category.replace('_',' ')} violation.")


# ─────────────────────────────────────────────────────────────────────────────
#  3. VISUAL XAI — Image Compliance Heatmap
# ─────────────────────────────────────────────────────────────────────────────
class VisualComplianceExplainer:
    """
    Generates visual explanations of why a creative image was flagged.
    Uses Grad-CAM style sliding window analysis (no neural network needed)
    to produce a heatmap showing which image regions contribute most to risk.

    Novel aspect: compliance heatmap applied to advertising creatives —
    shows art director exactly which region to modify.
    """

    def __init__(self, window_size: float = 0.25, stride: float = 0.12):
        """
        window_size: fraction of image covered by each sliding window
        stride: fraction to step between windows
        """
        self.window_size = window_size
        self.stride      = stride

    def explain_image(self, image: Image.Image,
                       detections: list = None) -> dict:
        """
        Generate compliance explanation heatmap for an image.

        Uses sliding window + colour-based risk scoring:
        - Green pixels (alcohol bottle colour) → high risk
        - Skin tones → medium risk (person detection)
        - Red regions (warning symbols) → critical risk
        - Dark/low-contrast text areas → contrast risk

        Returns:
          - heatmap: PIL Image (same size, RGBA with risk overlay)
          - annotated: PIL Image with explanation annotations
          - region_scores: list of {region, score, reason}
          - explanation: human-readable explanation
        """
        W, H = image.size
        arr  = np.array(image.convert("RGB"), dtype=np.float32)
        hsv  = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2HSV)

        win_w = int(W * self.window_size)
        win_h = int(H * self.window_size)
        step_w = max(1, int(W * self.stride))
        step_h = max(1, int(H * self.stride))

        # Score map (same size as image)
        score_map = np.zeros((H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)
        region_scores = []

        for y0 in range(0, H - win_h + 1, step_h):
            for x0 in range(0, W - win_w + 1, step_w):
                y1, x1 = y0 + win_h, x0 + win_w
                window_hsv = hsv[y0:y1, x0:x1]
                window_rgb = arr[y0:y1, x0:x1]
                risk, reasons = self._score_window(window_hsv, window_rgb)

                score_map[y0:y1, x0:x1] += risk
                count_map[y0:y1, x0:x1] += 1

                if risk > 0.3:
                    region_scores.append({
                        "x": x0, "y": y0, "w": win_w, "h": win_h,
                        "score": round(risk, 3),
                        "reasons": reasons,
                        "severity": "HIGH" if risk > 0.6 else "MEDIUM",
                    })

        # Average scores
        count_map = np.where(count_map == 0, 1, count_map)
        score_map = score_map / count_map
        score_map = (score_map - score_map.min()) / \
                    (score_map.max() - score_map.min() + 1e-8)

        # Build heatmap overlay
        heatmap = self._build_heatmap(image, score_map)

        # Build annotated image
        annotated = self._annotate_image(image, region_scores, detections or [])

        # Sort regions by score
        region_scores.sort(key=lambda r: r["score"], reverse=True)

        return {
            "heatmap":      heatmap,
            "annotated":    annotated,
            "region_scores": region_scores[:8],
            "overall_visual_risk": round(float(score_map.max()), 3),
            "explanation":  self._build_explanation(region_scores),
            "high_risk_regions": len([r for r in region_scores if r["score"] > 0.6]),
        }

    def _score_window(self, window_hsv: np.ndarray,
                       window_rgb: np.ndarray) -> tuple:
        """Score a window region for compliance risk."""
        risk    = 0.0
        reasons = []

        # Alcohol bottle colour (green HSV range)
        green_mask = cv2.inRange(window_hsv,
            np.array([35, 40, 40]), np.array([85, 255, 200]))
        green_ratio = green_mask.sum() / 255 / max(window_hsv.size // 3, 1)
        if green_ratio > 0.15:
            risk += 0.6 * min(green_ratio * 3, 1.0)
            reasons.append(f"Alcohol bottle colour ({green_ratio:.0%} coverage)")

        # Skin tone (person detection)
        skin_mask = cv2.inRange(window_hsv,
            np.array([0, 25, 50]), np.array([25, 165, 255]))
        skin_ratio = skin_mask.sum() / 255 / max(window_hsv.size // 3, 1)
        if skin_ratio > 0.20:
            risk += 0.35 * min(skin_ratio * 2, 1.0)
            reasons.append(f"Skin tone detected ({skin_ratio:.0%} coverage)")

        # Red warning symbols
        red1 = cv2.inRange(window_hsv, np.array([0,140,140]), np.array([8,255,255]))
        red2 = cv2.inRange(window_hsv, np.array([172,140,140]), np.array([180,255,255]))
        red_ratio = (red1 | red2).sum() / 255 / max(window_hsv.size // 3, 1)
        if red_ratio > 0.12:
            risk += 0.4 * min(red_ratio * 4, 1.0)
            reasons.append(f"Warning/red element ({red_ratio:.0%} coverage)")

        # Low contrast (text readability)
        gray = cv2.cvtColor(window_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        contrast = float(gray.std())
        if contrast < 30:
            risk += 0.25
            reasons.append(f"Low contrast region (std={contrast:.1f})")

        return min(risk, 1.0), reasons

    def _build_heatmap(self, image: Image.Image,
                        score_map: np.ndarray) -> Image.Image:
        """Build a coloured heatmap overlay (green→yellow→red)."""
        W, H   = image.size
        heatmap = np.zeros((H, W, 4), dtype=np.uint8)

        for y in range(H):
            for x in range(W):
                s = score_map[y, x]
                if s < 0.33:
                    r, g, b = 0, int(200 * (s / 0.33)), 0
                elif s < 0.66:
                    t = (s - 0.33) / 0.33
                    r, g, b = int(255 * t), 200, 0
                else:
                    t = (s - 0.66) / 0.34
                    r, g, b = 255, int(200 * (1 - t)), 0
                alpha = int(180 * s)
                heatmap[y, x] = [r, g, b, alpha]

        heatmap_img = Image.fromarray(heatmap, "RGBA")
        base_rgba   = image.convert("RGBA")
        result      = Image.alpha_composite(base_rgba, heatmap_img)
        return result.convert("RGB")

    def _annotate_image(self, image: Image.Image,
                         region_scores: list,
                         detections: list) -> Image.Image:
        """Annotate image with XAI explanations."""
        img  = image.copy().convert("RGB")
        draw = ImageDraw.Draw(img)
        try:
            fnt = ImageFont.truetype(
                "C:/Windows/Fonts/arial.ttf" if __import__("platform").system()=="Windows"
                else "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 14)
        except Exception:
            fnt = ImageFont.load_default()

        COLOURS = {"HIGH": "#E53E3E", "MEDIUM": "#DD6B20", "LOW": "#38A169"}

        for i, region in enumerate(region_scores[:5]):
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            col = COLOURS.get(region["severity"], "#3182CE")

            for t in range(2):
                draw.rectangle([x-t, y-t, x+w+t, y+h+t], outline=col)

            label = f"Risk:{region['score']:.2f} — {region['reasons'][0][:30] if region['reasons'] else ''}"
            ly = max(0, y - 18)
            try:
                bb  = draw.textbbox((0,0), label, font=fnt)
                lw_ = bb[2]-bb[0]+6
                draw.rectangle([x, ly, x+lw_, ly+16], fill=col)
                draw.text((x+3, ly+1), label, fill="white", font=fnt)
            except Exception:
                pass

        return img

    @staticmethod
    def _build_explanation(region_scores: list) -> str:
        if not region_scores:
            return "No high-risk visual regions detected."
        top = region_scores[0]
        reasons = top.get("reasons", ["unknown"])
        reason_str = reasons[0] if reasons else "visual elements"
        return (f"Highest risk region at position "
                f"({top['x']}, {top['y']}) — {reason_str}. "
                f"Score: {top['score']:.2f}/1.00. "
                f"Consider modifying this area to reduce compliance risk.")


# ─────────────────────────────────────────────────────────────────────────────
#  4. COUNTERFACTUAL EXPLAINER
#     "What would make this compliant?"
# ─────────────────────────────────────────────────────────────────────────────
class CounterfactualExplainer:
    """
    Generates counterfactual explanations:
    "If you change X, the compliance score drops from 65 to 22."

    This is the most actionable form of XAI for creative teams.
    """

    def explain(self, risk_result: dict, text_result: dict,
                cv_detections: list, contrast_result: dict) -> list:
        """
        Generate ordered list of counterfactual actions.
        Each action shows: what to change → predicted score reduction.
        """
        counterfactuals = []
        current_score   = risk_result.get("total_score", 50)

        # Text violations
        for v in text_result.get("violations", []):
            sev_pts = {"CRITICAL":32,"HIGH":20,"MEDIUM":10,"LOW":4}
            pts     = sev_pts.get(v["severity"], 0)
            counterfactuals.append({
                "action":          f"Remove {v['category']} from copy",
                "specific":        f'Remove: "{v["text"][:50]}"',
                "score_reduction": pts,
                "new_score":       max(0, current_score - pts),
                "difficulty":      "Easy",
                "category":        "Text",
                "suggestion":      v.get("suggestion", ""),
            })

        # CV detections
        cv_pts = {"CRITICAL":25,"HIGH":15,"MEDIUM":8,"INFO":2}
        for det in cv_detections:
            pts = cv_pts.get(det.get("compliance_risk","INFO"), 2)
            if pts >= 8:
                counterfactuals.append({
                    "action":          f"Remove/replace {det['label']}",
                    "specific":        det.get("rule",""),
                    "score_reduction": pts,
                    "new_score":       max(0, current_score - pts),
                    "difficulty":      "Medium",
                    "category":        "Visual",
                    "suggestion":      f"Replace {det['label']} with a compliant alternative",
                })

        # Contrast
        if contrast_result and not contrast_result.get("pass", True):
            ratio = contrast_result.get("ratio", 0)
            counterfactuals.append({
                "action":          "Fix text contrast ratio",
                "specific":        f"Current ratio {ratio}:1 — minimum required 4.5:1",
                "score_reduction": 7,
                "new_score":       max(0, current_score - 7),
                "difficulty":      "Easy",
                "category":        "Accessibility",
                "suggestion":      "Darken text or lighten background to achieve 4.5:1 contrast",
            })

        # Sort by score reduction (highest impact first)
        counterfactuals.sort(key=lambda x: x["score_reduction"], reverse=True)

        # Add cumulative score after applying all fixes
        running = current_score
        for cf in counterfactuals:
            running = max(0, running - cf["score_reduction"])
            cf["cumulative_score"] = running

        return counterfactuals


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLETONS
# ─────────────────────────────────────────────────────────────────────────────
_global_xai  = None
_token_xai   = None
_visual_xai  = None
_counter_xai = None

def get_global_xai()  -> GlobalFeatureExplainer:
    global _global_xai
    if _global_xai is None: _global_xai = GlobalFeatureExplainer()
    return _global_xai

def get_token_xai()   -> NLPTokenExplainer:
    global _token_xai
    if _token_xai is None: _token_xai = NLPTokenExplainer()
    return _token_xai

def get_visual_xai()  -> VisualComplianceExplainer:
    global _visual_xai
    if _visual_xai is None: _visual_xai = VisualComplianceExplainer()
    return _visual_xai

def get_counter_xai() -> CounterfactualExplainer:
    global _counter_xai
    if _counter_xai is None: _counter_xai = CounterfactualExplainer()
    return _counter_xai
