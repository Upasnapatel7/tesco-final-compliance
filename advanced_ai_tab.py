"""
advanced_ai_tab.py
==================
Streamlit UI for the three new patent-grade AI features:
  1. Explainable AI (XAI) — why did the system flag this creative?
  2. Vector DB + Recommendation — what similar approved creatives exist?
  3. RAG Compliance — grounded guideline retrieval for LLM context

Add to dashboard.py:
    from advanced_ai_tab import render_advanced_ai_tab
    # Add "🧠 Advanced AI" to st.tabs(...)
    with tab_adv: render_advanced_ai_tab(audit_result, image, brand, headline, subhead)
"""

import io
import numpy as np
from PIL import Image


def render_advanced_ai_tab(
    audit_result: dict = None,
    image:        Image.Image = None,
    brand:        str = "",
    headline:     str = "",
    subhead:      str = "",
    brand_primary: str = "#00539F",
    brand_accent:  str = "#E31837",
):
    """
    Render the Advanced AI tab with XAI, Vector DB, and RAG features.
    Call this inside a Streamlit tab context.
    """
    import streamlit as st

    st.subheader("🧠 Advanced AI — XAI · Vector DB · RAG")
    st.caption("Explainable AI · Creative Recommendation · Retrieval-Augmented Compliance")

    # Load modules
    try:
        from xai_explainer import (get_global_xai, get_token_xai,
                                    get_visual_xai, get_counter_xai)
        XAI_OK = True
    except ImportError as e:
        XAI_OK = False
        st.error(f"XAI module unavailable: {e}")

    try:
        from vector_db import get_vector_db, get_recommender, get_rag
        VDB_OK = True
    except ImportError as e:
        VDB_OK = False
        st.error(f"Vector DB module unavailable: {e}")

    ai1, ai2, ai3, ai4 = st.tabs([
        "💡 Explainable AI",
        "🔍 Token Analysis",
        "🗺️ Visual Heatmap",
        "📚 Recommendations & RAG",
    ])

    # ── TAB 1: Global + Instance XAI ─────────────────────────────────────────
    with ai1:
        st.markdown("**Why did the AI give this compliance score?**")
        st.caption("Uses GradientBoosting feature importances + permutation importance")

        if not XAI_OK:
            st.warning("Place xai_explainer.py in your project folder.")
        else:
            gxai = get_global_xai()

            with st.expander("📊 Global Feature Importance — What drives compliance risk?",
                             expanded=True):
                global_exp = gxai.explain_global()
                st.caption(f"Method: {global_exp['method']}")

                import plotly.express as px
                import pandas as pd

                df_fi = pd.DataFrame([{
                    "Feature":  f["feature"].replace("_"," ").title(),
                    "Importance":        round(f["importance"]*100, 1),
                    "Permutation":       round(f["permutation_importance"]*100, 1),
                    "Interpretation":    f["interpretation"],
                } for f in global_exp["feature_importances"]])

                fig = px.bar(df_fi.head(10), x="Importance", y="Feature",
                             orientation="h",
                             title="Top 10 Compliance Risk Drivers (GradientBoosting)",
                             color="Importance",
                             color_continuous_scale=["#238636","#9e6a03","#da3633"])
                fig.update_layout(height=350, margin=dict(t=40,b=10,l=10,r=10),
                                  paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)",
                                  font_color="#c9d1d9", coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(f"**Top 3 risk drivers:** {' · '.join(global_exp['top_3_drivers'])}")

            # Instance explanation from audit result
            if audit_result:
                risk   = audit_result.get("risk", {})
                text   = audit_result.get("text", {})
                dets   = audit_result.get("detections", [])
                contrast = audit_result.get("contrast", {})

                nlp_score     = risk.get("breakdown",{}).get("NLP Text Analysis", 0)
                cv_score      = risk.get("breakdown",{}).get("Computer Vision", 0)
                font_score    = risk.get("breakdown",{}).get("Font Size Compliance", 0)
                contrast_score= risk.get("breakdown",{}).get("Contrast (WCAG AA)", 0)
                img_ml_score  = risk.get("breakdown",{}).get("Image ML Model", 0)
                has_alcohol   = any(d["label"] == "Alcohol Bottle" for d in dets)
                has_person    = any(d["label"] == "Person" for d in dets)
                has_warning   = any("Warning" in d["label"] for d in dets)
                viols         = text.get("violations", [])
                health_wc     = sum(1 for v in viols if "health" in v["category"].lower())
                price_sym     = any("price" in v["category"].lower() for v in viols)
                hl_text       = headline or ""

                fv = [nlp_score, cv_score, font_score, contrast_score,
                      img_ml_score, float(has_alcohol), float(has_person),
                      float(has_warning), len(hl_text.split()),
                      len(hl_text), len(subhead), 
                      sum(1 for c in hl_text if c.isupper())/max(len(hl_text),1),
                      float(hl_text.count("!")), float(price_sym),
                      float(health_wc)]

                with st.expander("🎯 Instance Explanation — Why THIS creative?",
                                 expanded=True):
                    inst = gxai.explain_instance(fv)
                    r1, r2, r3 = st.columns(3)
                    col = "#238636" if inst["predicted_risk"] == "COMPLIANT" else "#da3633"
                    r1.markdown(
                        f'<div style="background:{col};color:white;padding:8px 12px;'
                        f'border-radius:6px;text-align:center"><b>{inst["predicted_risk"]}</b></div>',
                        unsafe_allow_html=True)
                    r2.metric("Risk probability", f"{inst['risk_probability']:.0%}")
                    r3.metric("Actual score", risk.get("total_score", "?"))

                    st.markdown("**Per-feature contributions (perturbation analysis):**")
                    for c in inst["top_contributors"][:5]:
                        direction_col = "#da3633" if c["direction"]=="increases_risk" else "#238636"
                        bar_width = int(abs(c["contribution"]) * 500)
                        st.markdown(
                            f'<div style="margin:3px 0;display:flex;align-items:center;gap:8px">'
                            f'<span style="width:200px;font-size:12px">{c["feature"].replace("_"," ").title()}</span>'
                            f'<div style="width:{min(bar_width,200)}px;height:16px;background:{direction_col};border-radius:3px"></div>'
                            f'<span style="font-size:11px;color:#8b949e">{c["contribution"]:+.4f} ({c["impact"]})</span>'
                            f'</div>', unsafe_allow_html=True)

                    st.info(f"💡 **To make compliant:** {inst['counterfactual']}")

                # Counterfactual actions
                with st.expander("⚡ Counterfactual Actions — Highest impact fixes"):
                    counter = get_counter_xai()
                    cfs = counter.explain(risk, text, dets, contrast)
                    if cfs:
                        for i, cf in enumerate(cfs[:5]):
                            score_after = cf["new_score"]
                            reduction   = cf["score_reduction"]
                            st.markdown(
                                f'<div style="border-left:4px solid #1f6feb;padding:8px 12px;'
                                f'margin:4px 0;background:#0d1117;border-radius:4px">'
                                f'<b>#{i+1}</b> {cf["action"]} '
                                f'<span style="color:#238636">−{reduction} pts → score {score_after}</span><br>'
                                f'<span style="font-size:12px;color:#8b949e">{cf["specific"][:80]}</span><br>'
                                f'<span style="font-size:11px;color:#58a6ff">💡 {cf["suggestion"][:80]}</span>'
                                f'</div>', unsafe_allow_html=True)
                        if cfs:
                            final_score = cfs[-1]["cumulative_score"]
                            st.success(f"Applying all fixes: score drops from "
                                      f"{risk.get('total_score','?')} → {final_score}")
                    else:
                        st.success("No actionable fixes needed — creative appears compliant.")
            else:
                st.info("Run an AI Compliance audit first to see instance explanations.")

    # ── TAB 2: Token-Level NLP XAI ────────────────────────────────────────────
    with ai2:
        st.markdown("**Which exact words triggered the violation?**")
        st.caption("Token-level perturbation analysis — LIME-style local explanation")

        if not XAI_OK:
            st.warning("Place xai_explainer.py in your project folder.")
        else:
            txai = get_token_xai()

            col_t1, col_t2 = st.columns(2)
            test_hl = col_t1.text_input("Headline to analyse",
                value=headline or "Clinically proven to boost immunity save 50%")
            test_sh = col_t2.text_input("Subhead to analyse",
                value=subhead or "Win a free prize competition now")

            if st.button("🔍 Analyse Tokens", type="primary"):
                for label, text in [("Headline", test_hl), ("Subhead", test_sh)]:
                    if not text.strip(): continue
                    result = txai.explain_text(text)

                    st.markdown(f"**{label} Analysis**")
                    cat_col = {"Compliant":"#238636","Health Claim":"#da3633",
                               "Competition":"#da3633","Price Violation":"#9e6a03",
                               "Misleading Claim":"#9e6a03","Age Restricted":"#da3633",
                               "Sustainability Claim":"#9e6a03"}.get(result["prediction"],"#888")

                    st.markdown(
                        f'<div style="background:#161b22;border:1px solid #21262d;'
                        f'padding:12px;border-radius:8px;margin:8px 0">'
                        f'<span style="background:{cat_col};color:white;padding:2px 8px;'
                        f'border-radius:10px;font-weight:700;font-size:12px">'
                        f'{result["prediction"]}</span> '
                        f'Confidence: {result["confidence"]:.0%}<br><br>'
                        f'<div style="font-size:15px;line-height:2">{result["highlighted_html"]}</div>'
                        f'</div>', unsafe_allow_html=True)

                    st.caption("🔴 High risk token  🟠 Medium risk token  Normal = compliant")

                    if result["risky_tokens"]:
                        import pandas as pd
                        df_t = pd.DataFrame([{
                            "Token": t["token"],
                            "Risk Score": t["risk"],
                            "Impact": t["contribution"],
                            "Category": t["category"].replace("_"," ").title(),
                        } for t in result["risky_tokens"]])
                        st.dataframe(df_t, use_container_width=True, hide_index=True)

                    st.markdown(f"**Explanation:** {result['explanation']}")
                    if result["compliant_rewrite"] != text:
                        st.success(f"**Suggested rewrite:** {result['compliant_rewrite']}")
                    st.divider()

    # ── TAB 3: Visual Heatmap XAI ─────────────────────────────────────────────
    with ai3:
        st.markdown("**Which image regions are compliance risks?**")
        st.caption("Sliding-window visual risk analysis — shows which areas to modify")

        if not XAI_OK:
            st.warning("Place xai_explainer.py in your project folder.")
        else:
            vxai = get_visual_xai()

            vis_img_file = st.file_uploader(
                "Upload creative image for visual XAI",
                type=["png","jpg","jpeg","webp"],
                key="xai_vis_img")

            use_audit = st.checkbox("Use image from last audit result",
                                    value=(image is not None))

            if st.button("🗺️ Generate Compliance Heatmap", type="primary"):
                target_img = None
                if use_audit and image is not None:
                    target_img = image
                elif vis_img_file:
                    target_img = Image.open(vis_img_file)

                if target_img is None:
                    st.warning("Upload an image or run an audit first.")
                else:
                    with st.spinner("Generating visual explanation…"):
                        detections = audit_result.get("detections",[]) if audit_result else []
                        result     = vxai.explain_image(target_img, detections)

                    ic1, ic2 = st.columns(2)
                    with ic1:
                        st.image(result["annotated"],
                                 caption="Annotated — top risk regions",
                                 use_column_width=True)
                    with ic2:
                        st.image(result["heatmap"],
                                 caption="Risk heatmap (green→yellow→red)",
                                 use_column_width=True)

                    st.metric("Overall visual risk", result["overall_visual_risk"])
                    st.info(f"💡 {result['explanation']}")

                    if result["region_scores"]:
                        import pandas as pd
                        df_r = pd.DataFrame([{
                            "Position":  f"({r['x']}, {r['y']})",
                            "Risk Score": r["score"],
                            "Severity":  r["severity"],
                            "Reason":    r["reasons"][0][:50] if r["reasons"] else "—",
                        } for r in result["region_scores"][:6]])
                        st.dataframe(df_r, use_container_width=True, hide_index=True)

    # ── TAB 4: Vector DB Recommendations + RAG ────────────────────────────────
    with ai4:
        st.markdown("**Find similar approved creatives + grounded compliance guidelines**")

        if not VDB_OK:
            st.warning("Place vector_db.py in your project folder.")
        else:
            vdb = get_vector_db()
            rec = get_recommender()
            rag = get_rag()

            db_stats = vdb.stats()
            s1, s2, s3 = st.columns(3)
            s1.metric("Creatives in DB",  db_stats["total_creatives"])
            s2.metric("Approved",          db_stats["approved"])
            s3.metric("Embedding dim",     db_stats["embedding_dim"])

            st.caption(f"Index: {db_stats['index_type']}")

            # Recommendation
            with st.expander("🎯 Creative Recommendations", expanded=True):
                rc1, rc2 = st.columns(2)
                rec_brand = rc1.selectbox("Brand", db_stats.get("brands",["Tesco"]) or ["Tesco"],
                                          key="rec_brand")
                rec_hl    = rc2.text_input("Headline", headline or "Discover New Range",
                                           key="rec_hl")
                rec_sh    = rc2.text_input("Subhead",  subhead  or "Available now",
                                           key="rec_sh")
                rec_img_file = rc1.file_uploader("Packshot (optional)",
                    type=["png","jpg","jpeg"], key="rec_img")

                if st.button("🔍 Find Similar Approved Creatives", type="primary"):
                    query_img = Image.open(rec_img_file) if rec_img_file else image
                    recs = rec.recommend(query_img, rec_hl, rec_sh,
                                        rec_brand, "", top_k=5)

                    if recs["similar_creatives"]:
                        st.success(f"Found {len(recs['similar_creatives'])} similar approved creatives")
                        st.markdown("**Recommended settings (from approved creatives):**")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Layout",   recs["recommended_layout"]   or "—")
                        m2.metric("Font",     (recs["recommended_font"] or "—").split("(")[0].strip())
                        m3.metric("Avg risk", f"{recs['avg_risk_of_similar']:.0f}/100")
                        st.caption(f"Confidence: {recs['confidence']:.0%}")

                        for k, reason in recs["reasoning"].items():
                            st.caption(f"• {k.title()}: {reason}")

                        st.info(f"💡 {recs['brand_specific_tip']}")

                        # Show similar creatives
                        st.markdown("**Most similar approved creatives:**")
                        for i, sim in enumerate(recs["similar_creatives"][:3]):
                            st.markdown(
                                f'<div style="border:1px solid #21262d;border-radius:8px;'
                                f'padding:10px;margin:4px 0;background:#0d1117">'
                                f'<b>#{i+1}</b> {sim["brand"]} · {sim["layout"]} · '
                                f'Risk: {sim["risk_score"]} · Grade: {sim["grade"]}<br>'
                                f'<span style="color:#8b949e;font-size:12px">'
                                f'"{sim["headline"]}" / "{sim["subhead"]}"</span><br>'
                                f'Similarity: {sim["similarity"]:.0%}'
                                f'</div>', unsafe_allow_html=True)
                    else:
                        st.info("No similar approved creatives found. Add more approved creatives to build recommendations.")

                # Add current creative to DB
                if audit_result and image:
                    risk = audit_result.get("risk",{})
                    if st.button("✅ Save to Vector DB as Approved",
                                 disabled=risk.get("total_score",100) > 30):
                        if risk.get("total_score",100) <= 30:
                            vdb.add(image, headline, subhead, brand, "",
                                   risk_score=risk.get("total_score",0),
                                   grade=risk.get("grade","?"), approved=True)
                            st.success("Creative saved to vector database!")
                        else:
                            st.error("Only compliant creatives (score ≤30) can be saved as approved.")

            # RAG Guidelines
            with st.expander("📚 RAG — Retrieved Compliance Guidelines", expanded=True):
                rag_query = st.text_input("Search guidelines",
                    value=f"{headline} {subhead}".strip() or "compliance violations",
                    key="rag_q")
                category_filter = st.selectbox("Filter by category",
                    ["All","health","misleading","sustainability","alcohol",
                     "accessibility","deceptive","unfair_practices"],
                    key="rag_cat")

                if st.button("📖 Retrieve Guidelines", type="primary"):
                    cf = None if category_filter == "All" else category_filter
                    guidelines = rag.retrieve(rag_query, top_k=4, category_filter=cf)

                    if guidelines:
                        for g in guidelines:
                            rel_col = {"HIGH":"#238636","MEDIUM":"#9e6a03",
                                       "LOW":"#30363d"}.get(g["relevance"],"#333")
                            st.markdown(
                                f'<div style="border-left:4px solid {rel_col};'
                                f'padding:10px 14px;margin:6px 0;background:#161b22;'
                                f'border-radius:4px">'
                                f'<b>{g["guideline"]["source"]}</b> '
                                f'<span style="font-size:11px;color:#8b949e">'
                                f'[{g["guideline"]["category"]}] '
                                f'Relevance: {g["relevance"]} ({g["similarity"]:.0%})</span><br>'
                                f'{g["guideline"]["text"]}'
                                f'</div>', unsafe_allow_html=True)
                    else:
                        st.info("No guidelines found for this query.")

                # Add custom guideline
                with st.expander("➕ Add Custom Guideline"):
                    cg_source = st.text_input("Source (e.g. 'Company Brand Guidelines')")
                    cg_cat    = st.text_input("Category (e.g. 'brand_voice')")
                    cg_text   = st.text_area("Guideline text")
                    if st.button("Add Guideline") and cg_source and cg_text:
                        rag.add_guideline(cg_source, cg_cat, cg_text)
                        st.success("Guideline added to knowledge base!")

                # Show LLM-ready context
                if audit_result:
                    viols = audit_result.get("text",{}).get("violations",[])
                    if viols:
                        context = rag.build_compliance_context(
                            viols, f"{headline} {subhead}")
                        with st.expander("🤖 LLM-Ready Context (RAG augmentation)"):
                            st.caption("This context is automatically prepended to Claude API calls for grounded compliance reasoning")
                            st.code(context, language=None)
