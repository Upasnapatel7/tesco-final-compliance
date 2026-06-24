"""
policy_analyser.py — Custom Compliance Policy Analyser
=======================================================
Users upload brand/regulatory PDF policy documents.
System extracts text using pdfminer, embeds as vectors,
then analyses creatives against those policies using
cosine similarity + NLP matching + Claude API (if available).
"""

import os
import io
import json
import re
import sqlite3
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


# ─────────────────────────────────────────────────────────────────────────────
#  PDF TEXT EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────
def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pdfminer."""
    try:
        from pdfminer.high_level import extract_text_to_fp
        from pdfminer.layout import LAParams
        output = io.StringIO()
        extract_text_to_fp(
            io.BytesIO(pdf_bytes), output,
            laparams=LAParams(), output_type="text", codec=None)
        text = output.getvalue()
        return text.strip()
    except Exception:
        try:
            from pdfminer.high_level import extract_text
            return extract_text(io.BytesIO(pdf_bytes)).strip()
        except Exception as e:
            return f"PDF extraction failed: {e}"


# ─────────────────────────────────────────────────────────────────────────────
#  POLICY STORE
# ─────────────────────────────────────────────────────────────────────────────
_POLICY_DB = Path("policies.db")

def _init_policy_db():
    with sqlite3.connect(_POLICY_DB) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS policies (
            id INTEGER PRIMARY KEY,
            name TEXT, source TEXT, brand TEXT,
            full_text TEXT, sections TEXT,
            embedding TEXT, uploaded_at TEXT
        )""")
        c.commit()

def save_policy(name: str, source: str, brand: str,
                full_text: str, sections: list,
                embedding: list) -> int:
    _init_policy_db()
    with sqlite3.connect(_POLICY_DB) as c:
        cursor = c.execute("""INSERT INTO policies
            (name, source, brand, full_text, sections, embedding, uploaded_at)
            VALUES (?,?,?,?,?,?,?)""",
            (name, source, brand, full_text,
             json.dumps(sections), json.dumps(embedding),
             datetime.now().isoformat()))
        c.commit()
        return cursor.lastrowid

def load_policies() -> list:
    _init_policy_db()
    with sqlite3.connect(_POLICY_DB) as c:
        rows = c.execute("""SELECT id, name, source, brand,
            full_text, sections, embedding, uploaded_at
            FROM policies ORDER BY id DESC""").fetchall()
    result = []
    for row in rows:
        try:
            result.append({
                "id": row[0], "name": row[1], "source": row[2],
                "brand": row[3], "full_text": row[4],
                "sections": json.loads(row[5] or "[]"),
                "embedding": json.loads(row[6] or "[]"),
                "uploaded_at": row[7],
            })
        except Exception:
            pass
    return result

def delete_policy(policy_id: int):
    with sqlite3.connect(_POLICY_DB) as c:
        c.execute("DELETE FROM policies WHERE id=?", (policy_id,))
        c.commit()


# ─────────────────────────────────────────────────────────────────────────────
#  POLICY PARSER — extracts rules from raw text
# ─────────────────────────────────────────────────────────────────────────────
def parse_policy_sections(text: str) -> list:
    """
    Parse a policy document into structured rule sections.
    Identifies rules, forbidden terms, and requirements.
    """
    sections = []

    # Split by common section patterns
    patterns = [
        r'\n(?=\d+\.\s)',           # numbered sections
        r'\n(?=[A-Z]{2,}\s)',       # ALL CAPS headings
        r'\n(?=Section\s)',         # "Section X" headings
        r'\n(?=Clause\s)',          # "Clause X"
        r'\n(?=RULE\s)',            # "RULE X"
        r'\n(?=Appendix\s)',        # "Appendix X"
    ]

    combined = "|".join(patterns)
    chunks = re.split(combined, text)

    # Forbidden word patterns
    forbidden_signals = [
        "must not", "prohibited", "forbidden", "not permitted",
        "not allowed", "shall not", "cannot", "do not use",
        "avoid", "never use", "restricted", "banned",
    ]

    # Required word patterns
    required_signals = [
        "must", "required", "mandatory", "shall", "should include",
        "always", "ensure", "compulsory",
    ]

    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) < 20:
            continue

        # Classify chunk
        chunk_lower = chunk.lower()
        is_forbidden = any(sig in chunk_lower for sig in forbidden_signals)
        is_required  = any(sig in chunk_lower for sig in required_signals)

        # Extract key terms
        words = re.findall(r'\b[a-zA-Z]{4,}\b', chunk_lower)
        word_freq = {}
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1
        key_terms = sorted(word_freq, key=word_freq.get, reverse=True)[:5]

        sections.append({
            "text":       chunk[:500],
            "type":       "forbidden" if is_forbidden else
                          "required"  if is_required  else "general",
            "key_terms":  key_terms,
            "word_count": len(chunk.split()),
        })

    return sections[:50]  # cap at 50 sections


# ─────────────────────────────────────────────────────────────────────────────
#  POLICY EMBEDDER
# ─────────────────────────────────────────────────────────────────────────────
class PolicyEmbedder:
    """Embeds policy text into vector space for similarity search."""

    _DIM = 64

    def __init__(self):
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), max_features=2000,
            stop_words="english")
        self._svd = TruncatedSVD(n_components=self._DIM, random_state=42)
        self._fitted = False

        # Seed with compliance-domain vocabulary
        seed = [
            "prohibited forbidden not permitted advertising standards",
            "health claim nutrition benefit medical wellness",
            "price discount sale offer promotion clearance",
            "competition prize draw entry win gambling",
            "misleading deceptive false substantiation evidence",
            "sustainability eco green environmental carbon neutral",
            "alcohol responsible drinking drinkaware underage",
            "mandatory required must include logo disclaimer",
            "brand identity colour font typography logo guidelines",
            "consumer protection unfair commercial practices directive",
        ]
        X = self._vectorizer.fit_transform(seed)
        self._svd.fit(X)
        self._fitted = True

    def embed(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros(self._DIM, dtype=np.float32)
        X = self._vectorizer.transform([text.lower()])
        v = self._svd.transform(X)[0]
        n = np.linalg.norm(v)
        return (v / max(n, 1e-8)).astype(np.float32)

    def embed_sections(self, sections: list) -> np.ndarray:
        if not sections:
            return np.zeros((1, self._DIM), dtype=np.float32)
        texts = [s["text"] for s in sections]
        X = self._vectorizer.transform([t.lower() for t in texts])
        V = self._svd.transform(X)
        return normalize(V, norm="l2").astype(np.float32)


_embedder = PolicyEmbedder()


# ─────────────────────────────────────────────────────────────────────────────
#  CREATIVE vs POLICY ANALYSER
# ─────────────────────────────────────────────────────────────────────────────
class PolicyComplianceAnalyser:
    """
    Analyses a creative against uploaded policy documents.

    Steps:
    1. Embed creative copy as query vector
    2. Retrieve most relevant policy sections via cosine similarity
    3. Check creative copy against forbidden/required terms
    4. Generate violation report with specific policy references
    5. If Claude API available, get deeper reasoning
    """

    def __init__(self):
        self._api_key = os.getenv("ANTHROPIC_API_KEY", "")

    def analyse(self, headline: str, subhead: str,
                brand: str, policies: list) -> dict:
        """
        Analyse creative against all uploaded policies.

        Returns:
          - policy_violations: list of specific policy violations
          - policy_passes: list of policy checks passed
          - most_relevant_sections: top policy sections
          - overall_policy_score: 0-100
          - recommendations: actionable fixes
        """
        if not policies:
            return {
                "policy_violations": [],
                "policy_passes":     [],
                "most_relevant_sections": [],
                "overall_policy_score":   100,
                "recommendations":        [],
                "message":  "No policies uploaded. Upload brand/regulatory PDFs to analyse.",
            }

        creative_text = f"{headline} {subhead}".strip()
        query_vec     = _embedder.embed(creative_text)

        all_violations = []
        all_passes     = []
        all_sections   = []

        for policy in policies:
            result = self._check_against_policy(
                creative_text, headline, subhead, query_vec, policy)
            all_violations.extend(result["violations"])
            all_passes.extend(result["passes"])
            all_sections.extend(result["relevant_sections"])

        # Sort sections by relevance
        all_sections.sort(key=lambda x: x["similarity"], reverse=True)

        # Score
        penalty = sum(
            {"CRITICAL":30,"HIGH":20,"MEDIUM":10,"LOW":5}.get(
                v["severity"], 5) for v in all_violations)
        score = max(0, 100 - penalty)

        # Claude deeper analysis
        claude_analysis = None
        if self._api_key and len(self._api_key) > 10 and all_violations:
            claude_analysis = self._claude_analyse(
                headline, subhead, brand, all_violations, all_sections[:3])

        return {
            "policy_violations":       all_violations,
            "policy_passes":           all_passes,
            "most_relevant_sections":  all_sections[:5],
            "overall_policy_score":    score,
            "grade":  "A" if score>=85 else "B" if score>=70 else
                      "C" if score>=55 else "D" if score>=40 else "F",
            "recommendations":         self._build_recommendations(all_violations),
            "claude_analysis":         claude_analysis,
            "policies_checked":        len(policies),
        }

    def _check_against_policy(self, creative_text: str,
                               headline: str, subhead: str,
                               query_vec: np.ndarray,
                               policy: dict) -> dict:
        violations = []
        passes     = []
        relevant   = []

        sections = policy.get("sections", [])
        if not sections:
            return {"violations": [], "passes": [], "relevant_sections": []}

        # Embed sections
        section_vecs = _embedder.embed_sections(sections)
        sims = section_vecs @ query_vec

        # Top relevant sections
        top_idx = np.argsort(sims)[-5:][::-1]
        for idx in top_idx:
            if idx < len(sections) and float(sims[idx]) > 0.1:
                relevant.append({
                    "policy_name": policy["name"],
                    "section":     sections[idx]["text"][:200],
                    "type":        sections[idx]["type"],
                    "similarity":  round(float(sims[idx]), 3),
                })

        # Check forbidden sections against creative text
        creative_lower = creative_text.lower()
        for i, section in enumerate(sections):
            if section["type"] != "forbidden":
                continue

            # Find forbidden terms in policy section
            sec_text = section["text"].lower()
            found_violations = []

            for term in section["key_terms"]:
                if len(term) > 4 and term in creative_lower:
                    # Check if this term appears in a forbidden context
                    found_violations.append(term)

            if found_violations:
                violations.append({
                    "policy":      policy["name"],
                    "section":     section["text"][:150],
                    "terms_found": found_violations,
                    "severity":    "HIGH" if len(found_violations) > 2 else "MEDIUM",
                    "description": f"Creative contains terms restricted in {policy['name']}",
                })
            elif float(sims[i]) > 0.4:
                # High similarity to forbidden section is a warning
                violations.append({
                    "policy":      policy["name"],
                    "section":     section["text"][:150],
                    "terms_found": [],
                    "severity":    "LOW",
                    "description": f"Creative copy is similar to a restricted section in {policy['name']} (similarity: {sims[i]:.0%})",
                })

        # Check required sections
        for section in sections:
            if section["type"] != "required":
                continue
            # Check if required elements are present
            required_met = any(
                term in creative_lower for term in section["key_terms"])
            if required_met:
                passes.append({
                    "policy":      policy["name"],
                    "description": f"Required element found: {', '.join(section['key_terms'][:3])}",
                })

        return {
            "violations": violations,
            "passes":     passes,
            "relevant_sections": relevant,
        }

    def _build_recommendations(self, violations: list) -> list:
        recs = []
        for v in violations[:5]:
            terms = v.get("terms_found", [])
            if terms:
                recs.append(
                    f"Remove or replace '{', '.join(terms[:3])}' — "
                    f"restricted by {v['policy']}")
            else:
                recs.append(
                    f"Review copy against {v['policy']} — "
                    f"high similarity to restricted section detected")
        return recs

    def _claude_analyse(self, headline, subhead, brand,
                         violations, sections) -> str:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self._api_key)

            viol_text = "\n".join(
                f"- {v['description']} (Policy: {v['policy']})"
                for v in violations[:3])
            sec_text  = "\n".join(
                f"- [{s['policy_name']}]: {s['section'][:100]}"
                for s in sections[:2])

            prompt = f"""You are a compliance expert reviewing ad copy against brand policies.

Ad copy:
  Headline: "{headline}"
  Subhead: "{subhead}"
  Brand: {brand}

Policy violations detected:
{viol_text}

Relevant policy sections:
{sec_text}

Provide:
1. A brief assessment (2 sentences)
2. The most important fix (1 sentence)
3. A compliant rewrite of the headline (1 sentence)

Keep response under 100 words total."""

            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                messages=[{"role":"user","content":prompt}])
            return resp.content[0].text.strip()
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
#  STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
def render_policy_uploader():
    """Renders the policy upload section in Advanced AI tab."""
    import streamlit as st

    st.subheader("📋 Custom Compliance Policies")
    st.caption("Upload brand guidelines or regulatory PDFs — AI analyses creatives against them")

    # Upload new policy
    with st.expander("➕ Upload New Policy Document", expanded=True):
        pc1, pc2 = st.columns(2)
        pol_name  = pc1.text_input("Policy name", placeholder="e.g. Tesco Brand Guidelines 2024")
        pol_brand = pc2.text_input("Brand", placeholder="e.g. Tesco")
        pol_source= pc1.text_input("Source", placeholder="e.g. Tesco Internal")
        pol_file  = st.file_uploader(
            "Upload PDF or TXT",
            type=["pdf","txt"],
            key="policy_upload_file")

        if st.button("📤 Upload & Process Policy", type="primary",
                     disabled=not (pol_file and pol_name)):
            with st.spinner("Extracting and embedding policy…"):
                try:
                    raw_bytes = pol_file.read()

                    if pol_file.name.endswith(".pdf"):
                        text = extract_pdf_text(raw_bytes)
                    else:
                        text = raw_bytes.decode("utf-8", errors="replace")

                    if len(text) < 50:
                        st.error("Could not extract text from this file. Try a text-layer PDF.")
                    else:
                        sections  = parse_policy_sections(text)
                        embedding = _embedder.embed(text[:2000]).tolist()
                        pid = save_policy(pol_name, pol_source or "Custom",
                                         pol_brand or brand if 'brand' in dir() else "",
                                         text, sections, embedding)
                        st.success(f"✅ Policy '{pol_name}' uploaded! "
                                  f"Extracted {len(sections)} sections, "
                                  f"{len(text.split())} words.")
                except Exception as e:
                    st.error(f"Upload failed: {e}")

    # Show uploaded policies
    policies = load_policies()
    if policies:
        st.markdown(f"**{len(policies)} polic{'ies' if len(policies)>1 else 'y'} loaded:**")
        for pol in policies:
            pc1, pc2, pc3 = st.columns([3, 1, 1])
            pc1.markdown(
                f'<div style="padding:6px 0">'
                f'<b>{pol["name"]}</b> '
                f'<span style="color:#8b949e;font-size:11px">'
                f'· {pol["brand"] or "General"} · {len(pol["sections"])} sections'
                f'</span></div>', unsafe_allow_html=True)
            pc2.caption(pol["uploaded_at"][:10])
            if pc3.button("🗑️", key=f"del_pol_{pol['id']}"):
                delete_policy(pol["id"])
                st.rerun()
    else:
        st.info("No policies uploaded yet. Upload a PDF to get started.")

    return policies


def render_policy_analysis(headline: str, subhead: str,
                            brand: str, policies: list):
    """Renders policy analysis results."""
    import streamlit as st

    if not policies:
        st.info("Upload compliance policy PDFs above to analyse your creative against them.")
        return

    if not (headline or subhead):
        st.info("Enter a headline or subhead to analyse against policies.")
        return

    if st.button("🔍 Analyse Against Policies", type="primary",
                 use_container_width=True):
        with st.spinner("Analysing creative against uploaded policies…"):
            analyser = PolicyComplianceAnalyser()
            result   = analyser.analyse(headline, subhead, brand, policies)

        score = result["overall_policy_score"]
        grade = result["grade"]
        gcol  = {"A":"#238636","B":"#1f6feb","C":"#9e6a03",
                 "D":"#da3633","F":"#6e1818"}.get(grade,"#888")

        # Score display
        r1, r2, r3 = st.columns(3)
        r1.markdown(
            f'<div style="background:{gcol};color:white;padding:12px;'
            f'border-radius:8px;text-align:center">'
            f'<div style="font-size:28px;font-weight:700">{grade}</div>'
            f'<div style="font-size:11px">Policy Grade</div></div>',
            unsafe_allow_html=True)
        r2.metric("Policy Score", f"{score}/100")
        r3.metric("Policies Checked", result["policies_checked"])

        # Violations
        viols = result["policy_violations"]
        if viols:
            st.markdown(f"**⚠️ {len(viols)} Policy Violation(s) Found:**")
            for v in viols:
                sev_col = {"CRITICAL":"#6e1818","HIGH":"#da3633",
                           "MEDIUM":"#9e6a03","LOW":"#1f6feb"}.get(v["severity"],"#888")
                st.markdown(
                    f'<div style="border-left:4px solid {sev_col};padding:8px 12px;'
                    f'margin:4px 0;background:#0d1117;border-radius:4px">'
                    f'<b>[{v["severity"]}] {v["policy"]}</b><br>'
                    f'<span style="font-size:12px;color:#c9d1d9">{v["description"]}</span>'
                    + (f'<br><span style="font-size:11px;color:#ff8c00">Terms: {", ".join(v["terms_found"])}</span>'
                       if v.get("terms_found") else "")
                    + f'</div>', unsafe_allow_html=True)
        else:
            st.success("✅ No specific policy violations detected.")

        # Passes
        if result["policy_passes"]:
            with st.expander(f"✅ {len(result['policy_passes'])} checks passed"):
                for p in result["policy_passes"]:
                    st.markdown(f"· {p['description']} ({p['policy']})")

        # Recommendations
        if result["recommendations"]:
            st.markdown("**💡 Recommendations:**")
            for rec in result["recommendations"]:
                st.caption(f"→ {rec}")

        # Claude analysis
        if result.get("claude_analysis"):
            st.markdown("**🤖 Claude AI Analysis:**")
            st.info(result["claude_analysis"])

        # Relevant sections
        if result["most_relevant_sections"]:
            with st.expander("📄 Most Relevant Policy Sections"):
                for sec in result["most_relevant_sections"][:3]:
                    st.markdown(
                        f'<div style="border:1px solid #21262d;padding:10px;'
                        f'border-radius:6px;margin:4px 0;background:#161b22">'
                        f'<b>{sec["policy_name"]}</b> '
                        f'<span style="color:#8b949e;font-size:11px">'
                        f'[{sec["type"]}] Relevance: {sec["similarity"]:.0%}</span><br>'
                        f'<span style="font-size:12px">{sec["section"]}</span>'
                        f'</div>', unsafe_allow_html=True)
