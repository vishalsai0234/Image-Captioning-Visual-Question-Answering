import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import requests
from io import BytesIO
import json
from datetime import datetime

from models import (
    load_caption_model,
    load_vqa_model,
    generate_caption,
    answer_question,
    load_image_from_url,
    DEVICE,
)
from evaluation import (
    compute_bleu_scores,
    compute_meteor,
    compute_cider,
    compute_vqa_accuracy,
)


# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DA627 - VQA & Captioning",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f0f4f8; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1B2A 0%, #0E4D5C 100%);
    }
    [data-testid="stSidebar"] * { color: #e0f2f4 !important; }
    [data-testid="stSidebar"] .stRadio label { color: #e0f2f4 !important; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: white;
        border-radius: 12px;
        padding: 16px;
        border-left: 4px solid #0E9AA7;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    /* Caption output box */
    .caption-box {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        border-left: 5px solid #0E9AA7;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        font-size: 18px;
        font-style: italic;
        color: #1a2e3a;
        margin: 12px 0;
    }

    /* Answer box */
    .answer-box {
        background: #e8f9fb;
        border-radius: 10px;
        padding: 14px 20px;
        border-left: 4px solid #0E9AA7;
        font-size: 16px;
        font-weight: bold;
        color: #0D5E6E;
        margin: 8px 0;
    }

    /* Section header */
    .section-header {
        font-size: 22px;
        font-weight: 700;
        color: #0D1B2A;
        padding-bottom: 6px;
        border-bottom: 3px solid #0E9AA7;
        margin-bottom: 16px;
    }

    /* Info badge */
    .badge {
        display: inline-block;
        background: #0E9AA7;
        color: white;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
    }

    /* Correct / wrong tags */
    .tag-correct {
        background: #d1fae5; color: #065f46;
        border-radius: 6px; padding: 2px 10px;
        font-weight: 600; font-size: 13px;
    }
    .tag-wrong {
        background: #fee2e2; color: #991b1b;
        border-radius: 6px; padding: 2px 10px;
        font-weight: 600; font-size: 13px;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    #st.markdown("# DA627 Project")
    #st.markdown("## 🤖 Multimodal Image Captioning & VQA")
    #st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Home",
         "📷 Image Captioning",
         "❓ Visual Q&A",
         "📊 Evaluation",
         "📖 About"],
        label_visibility="collapsed",
    )

    #st.markdown("---")
    #st.markdown("**Device**")
    #device_color = "#22c55e" if DEVICE.type == "cuda" else "#f59e0b"
    #st.markdown(
    #    f'<span style="color:{device_color}">● </span>'
    #    f'`{str(DEVICE).upper()}`',
    #    unsafe_allow_html=True,
    #)

    #st.markdown("**Models**")
    #st.markdown("• `BLIP-Captioning-Base`")
    #st.markdown("• `BLIP-VQA-Base`")

    #st.markdown("---")
    #st.markdown("**Vishal · 220150029**")

    #st.markdown(
    #    "<small>Vishal · 220150029<br>DA627 Project Proposal</small>",
    #    unsafe_allow_html=True,
    #)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_image_from_upload(uploaded) -> Image.Image:
    return Image.open(uploaded).convert("RGB")


def image_input_widget(key_prefix: str):
    """
    Reusable image input: upload file OR paste URL.
    Returns (PIL Image, source label) or (None, None).
    """
    tab_upload, tab_url = st.tabs(["📂 Upload Image", "🔗 Image URL"])

    with tab_upload:
        uploaded = st.file_uploader(
            "Drop an image file",
            type=["jpg", "jpeg", "png", "webp"],
            key=f"{key_prefix}_upload",
        )
        if uploaded:
            img = load_image_from_upload(uploaded)
            return img, uploaded.name

    with tab_url:
        url = st.text_input(
            "Paste image URL",
            placeholder="https://example.com/image.jpg",
            key=f"{key_prefix}_url",
        )
        if url:
            try:
                img = load_image_from_url(url)
                return img, url
            except Exception as e:
                st.error(f"Could not load image from URL: {e}")

    return None, None


def metric_gauge(name: str, value: float, max_val: float = 1.0):
    """Small Plotly gauge for a single metric."""
    fig = go.Figure(go.Indicator(
        mode   = "gauge+number",
        value  = value,
        title  = {"text": name, "font": {"size": 14}},
        number = {"font": {"size": 20}, "valueformat": ".3f"},
        gauge  = {
            "axis"     : {"range": [0, max_val], "tickwidth": 1},
            "bar"      : {"color": "#0E9AA7"},
            "bgcolor"  : "white",
            "steps"    : [
                {"range": [0,          max_val * 0.4], "color": "#fef3c7"},
                {"range": [max_val * 0.4, max_val * 0.7], "color": "#d1fae5"},
                {"range": [max_val * 0.7, max_val],    "color": "#a7f3d0"},
            ],
            "threshold": {
                "line" : {"color": "#0D1B2A", "width": 3},
                "value": value,
            },
        },
    ))
    fig.update_layout(height=200, margin=dict(t=40, b=10, l=20, r=20))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Home":
    st.markdown("""
    <div style='background: linear-gradient(135deg, #0D1B2A 0%, #0E4D5C 100%);
                border-radius: 16px; padding: 40px 36px; margin-bottom: 28px;'>
        <h1 style='color:white; margin:0; font-size:34px;'>
            🤖 Multimodal Image Captioning & VQA
        </h1>
        <p style='color:#7BAAB2; margin-top:10px; font-size:16px;'>
            DA627 Project · Vishal (220150029)
        </p>
        <p style='color:#b0d4da; margin-top:6px; font-size:14px;'>
            A unified BLIP-based system for generating image captions
            and answering visual questions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div style='background:white; border-radius:12px; padding:24px;
                    border-top:4px solid #0E9AA7; box-shadow:0 2px 8px rgba(0,0,0,0.08);
                    text-align:center;'>
            <div style='font-size:36px;'>📷</div>
            <h3 style='color:#0D1B2A; margin:10px 0 6px;'>Image Captioning</h3>
            <p style='color:#666; font-size:13px;'>
                Generate natural language descriptions of images using BLIP.
            </p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style='background:white; border-radius:12px; padding:24px;
                    border-top:4px solid #0E9AA7; box-shadow:0 2px 8px rgba(0,0,0,0.08);
                    text-align:center;'>
            <div style='font-size:36px;'>❓</div>
            <h3 style='color:#0D1B2A; margin:10px 0 6px;'>Visual Q&A</h3>
            <p style='color:#666; font-size:13px;'>
                Ask natural language questions about any image.
            </p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div style='background:white; border-radius:12px; padding:24px;
                    border-top:4px solid #0E9AA7; box-shadow:0 2px 8px rgba(0,0,0,0.08);
                    text-align:center;'>
            <div style='font-size:36px;'>📊</div>
            <h3 style='color:#0D1B2A; margin:10px 0 6px;'>Evaluation</h3>
            <p style='color:#666; font-size:13px;'>
                Compute BLEU, METEOR, CIDEr & VQA Accuracy metrics.
            </p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Pipeline Overview</div>', unsafe_allow_html=True)

    steps = [
        ("1", "Data Preprocessing", "Resize, normalize images; tokenize text"),
        ("2", "Feature Extraction",  "CLIP / BLIP visual encoders"),
        ("3", "Multimodal Fusion",   "Shared vision-language embedding space"),
        ("4", "Caption Generation",  "BLIP generative decoder"),
        ("5", "VQA",                 "Image + question → answer"),
        ("6", "Evaluation",          "BLEU / METEOR / CIDEr / Accuracy"),
    ]

    cols = st.columns(6)
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div style='background:#0D1B2A; border-radius:10px; padding:16px 10px;
                        text-align:center; color:white; height:130px;'>
                <div style='font-size:24px; font-weight:900;
                            color:#F0C040;'>{num}</div>
                <div style='font-size:12px; font-weight:700;
                            margin:6px 0 4px; color:white;'>{title}</div>
                <div style='font-size:10px; color:#7BAAB2;
                            line-height:1.3;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📚 Datasets Used"):
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Flickr8k / Flickr30k**")
            st.markdown(
                "- 8,000 images × 5 captions = 40,000 descriptions\n"
                "- Standard benchmark for captioning\n"
                "- [Kaggle link](https://www.kaggle.com/datasets/adityajn105/flickr8k)"
            )
        with d2:
            st.markdown("**VQA Dataset**")
            st.markdown(
                "- 265,000+ images from MS-COCO\n"
                "- 750,000+ Q&A pairs\n"
                "- [visualqa.org](https://visualqa.org/download.html)"
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: IMAGE CAPTIONING
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📷 Image Captioning":
    st.markdown('<div class="section-header">📷 Image Captioning</div>',
                unsafe_allow_html=True)

    # Settings
    with st.expander("⚙️ Generation Settings", expanded=False):
        sc1, sc2 = st.columns(2)
        with sc1:
            num_beams = st.slider("Beam Width", 1, 10, 5,
                help="Higher = better quality but slower")
            max_tokens = st.slider("Max Caption Length (tokens)", 20, 100, 50)
        with sc2:
            conditional = st.text_input(
                "Conditional Prompt (optional)",
                placeholder="e.g.  a photo of",
                help="Guides the model to start with this phrase",
            )
            compare_beams = st.checkbox("Compare beam widths (1, 3, 5, 7)")

    image, source = image_input_widget("cap")

    if image:
        col_img, col_out = st.columns([1, 1.4])

        with col_img:
            st.image(image, caption=f"Size: {image.size[0]}×{image.size[1]}")

        with col_out:
            with st.spinner("Generating caption..."):
                cap = generate_caption(
                    image,
                    conditional_text=conditional or None,
                    max_new_tokens=max_tokens,
                    num_beams=num_beams,
                )

            st.markdown("**Generated Caption**")
            st.markdown(f'<div class="caption-box">"{cap}"</div>',
                        unsafe_allow_html=True)

            if conditional:
                st.markdown(f'<span class="badge">Prompt</span> `{conditional}`',
                            unsafe_allow_html=True)
            st.markdown(f'<span class="badge">Beams</span> `{num_beams}`  '
                        f'<span class="badge">Max tokens</span> `{max_tokens}`',
                        unsafe_allow_html=True)

            # Beam comparison
            if compare_beams:
                st.markdown("---")
                st.markdown("**Beam Width Comparison**")
                beam_data = []
                for b in [1, 3, 5, 7]:
                    c = generate_caption(image, max_new_tokens=max_tokens, num_beams=b)
                    beam_data.append({"Beams": b, "Caption": c})
                st.dataframe(pd.DataFrame(beam_data), hide_index=True)

    else:
        st.info("⬆️  Upload an image or paste a URL above to get started.")

        st.markdown("**Try a sample image:**")
        samples = {
            "🐶 Dog": "https://hips.hearstapps.com/hmg-prod/images/golden-retriever-relaxing-at-home-royalty-free-image-1752090274.pjpeg?crop=0.534xw:0.801xh;0.301xw,0.199xh",
            "🐱 Cat": "https://hips.hearstapps.com/hmg-prod/images/portrait-of-a-white-turkish-angora-cat-royalty-free-image-1718207522.jpg?crop=0.668xw:1.00xh;0.105xw,0",
        }
        s_cols = st.columns(len(samples))
        for col, (label, url) in zip(s_cols, samples.items()):
            with col:
                if st.button(label):
                    img = load_image_from_url(url)
                    cap = generate_caption(img)
                    st.image(img)
                    st.markdown(f'<div class="caption-box">"{cap}"</div>',
                                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: VISUAL Q&A
# ══════════════════════════════════════════════════════════════════════════════

elif page == "❓ Visual Q&A":
    st.markdown('<div class="section-header">❓ Visual Question Answering</div>',
                unsafe_allow_html=True)

    image, source = image_input_widget("vqa")

    if image:
        col_img, col_qa = st.columns([1, 1.4])

        with col_img:
            st.image(image, caption=f"Size: {image.size[0]}×{image.size[1]}")

        with col_qa:
            ## Single question
            #st.markdown("**Ask a Question**")
            #question = st.text_input(
            #    "Type your question",
            #    placeholder="What is in the image?",
            #    label_visibility="collapsed",
            #)

            #if st.button("🔍 Get Answer") and question:
            #    with st.spinner("Thinking..."):
            #        ans = answer_question(image, question)
            #    st.markdown(f'<div class="answer-box">💬 {ans}</div>',
            #                unsafe_allow_html=True)

            #st.markdown("---")

            # Batch questions
            #st.markdown("**Ask Multiple Questions**")
            st.markdown("**Ask Question(s)**")
            default_qs = (
                "What is in the image?\n"
                "What color is it?\n"
                "Is it indoors or outdoors?\n"
                "How many subjects are there?"
            )
            batch_input = st.text_area(
                "One question per line",
                value=default_qs,
                height=120,
                label_visibility="collapsed",
            )

            if st.button("🔍 Answer All"):
                questions = [q.strip() for q in batch_input.strip().splitlines() if q.strip()]
                if questions:
                    results = []
                    prog = st.progress(0)
                    for i, q in enumerate(questions):
                        a = answer_question(image, q)
                        results.append({"Question": q, "Answer": a})
                        prog.progress((i + 1) / len(questions))
                    prog.empty()

                    for r in results:
                        st.markdown(
                            f"**Q:** {r['Question']}",
                        )
                        st.markdown(
                            f'<div class="answer-box">💬 {r["Answer"]}</div>',
                            unsafe_allow_html=True,
                        )
    else:
        st.info("⬆️  Upload an image or paste a URL above to get started.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Evaluation":
    st.markdown('<div class="section-header">📊 Evaluation</div>', unsafe_allow_html=True)

    eval_tab1, eval_tab2 = st.tabs(["📐 Caption Metrics", "🎯 VQA Accuracy"])

    # ── CAPTION METRICS TAB ───────────────────────────────────────────────────
    with eval_tab1:
        st.markdown("Compute **BLEU-1 → 4**, **METEOR**, and **CIDEr** by comparing "
                    "the model's caption against your reference captions.")

        img_col, form_col = st.columns([1, 1.4])

        with img_col:
            image, _ = image_input_widget("eval_cap")
            if image:
                st.image(image)

        with form_col:
            st.markdown("**Reference Captions** *(one per line)*")
            refs_input = st.text_area(
                "references",
                height=160,
                placeholder=(
                    "a dog sitting on a wooden floor\n"
                    "a cute brown dog is resting\n"
                    "a small dog looking at the camera\n"
                    "a dog indoors on the floor\n"
                    "a brown dog sitting quietly"
                ),
                label_visibility="collapsed",
            )

            beam_eval = st.slider("Beam Width", 1, 10, 5, key="beam_eval")

            run_eval = st.button("▶  Run Evaluation",
                                 disabled=(image is None or not refs_input.strip()))

        if run_eval and image and refs_input.strip():
            references = [r.strip() for r in refs_input.strip().splitlines() if r.strip()]

            with st.spinner("Generating caption and computing metrics..."):
                pred_caption = generate_caption(image, num_beams=beam_eval)
                bleu   = compute_bleu_scores(pred_caption, references)
                meteor = compute_meteor(pred_caption, references)
                cider  = compute_cider(pred_caption, references)

            st.markdown("---")
            st.markdown(f'**Generated Caption:** <div class="caption-box">"{pred_caption}"</div>',
                        unsafe_allow_html=True)
            st.markdown(f"**References used:** {len(references)}")

            # Metric gauges
            st.markdown("#### Scores")
            g_cols = st.columns(6)
            gauge_data = [
                ("BLEU-1", bleu["BLEU-1"], 1.0),
                ("BLEU-2", bleu["BLEU-2"], 1.0),
                ("BLEU-3", bleu["BLEU-3"], 1.0),
                ("BLEU-4", bleu["BLEU-4"], 1.0),
                ("METEOR", meteor,          1.0),
                ("CIDEr",  cider,           10.0),
            ]
            for col, (name, val, mx) in zip(g_cols, gauge_data):
                with col:
                    st.plotly_chart(metric_gauge(name, val, mx),
                                    use_column_width=True)

            # Bar chart
            fig = px.bar(
                x=["BLEU-1","BLEU-2","BLEU-3","BLEU-4","METEOR"],
                y=[bleu["BLEU-1"], bleu["BLEU-2"], bleu["BLEU-3"], bleu["BLEU-4"], meteor],
                color_discrete_sequence=["#0E9AA7"],
                labels={"x": "Metric", "y": "Score"},
                title="Captioning Metrics (0 – 1 scale)",
            )
            fig.update_layout(
                plot_bgcolor="white",
                yaxis_range=[0, 1],
                height=320,
            )
            st.plotly_chart(fig)

            # Score table
            score_df = pd.DataFrame([{
                "Metric": k, "Score": v,
                "Interpretation": (
                    "Excellent" if v >= 0.5 else
                    "Good" if v >= 0.3 else
                    "Fair" if v >= 0.15 else "Low"
                )
            } for k, v in {**bleu, "METEOR": meteor, "CIDEr": cider}.items()])
            st.dataframe(score_df, hide_index=True)

            # Download
            result_json = json.dumps({
                "timestamp": datetime.now().isoformat(),
                "predicted_caption": pred_caption,
                "references": references,
                "scores": {**bleu, "METEOR": meteor, "CIDEr": cider},
            }, indent=2)
            st.download_button(
                "💾 Download Results (JSON)",
                data=result_json,
                file_name="caption_eval.json",
                mime="application/json",
            )

    # ── VQA ACCURACY TAB ──────────────────────────────────────────────────────
    with eval_tab2:
        st.markdown("Enter image, questions, and ground truth answers to compute **VQA Accuracy**.")

        vqa_img_col, vqa_form_col = st.columns([1, 1.4])

        with vqa_img_col:
            vqa_image, _ = image_input_widget("eval_vqa")
            if vqa_image:
                st.image(vqa_image)

        with vqa_form_col:
            st.markdown("**Q&A Pairs** *(question | ground_truth, one per line)*")
            qa_input = st.text_area(
                "qa pairs",
                height=180,
                placeholder=(
                    "What animal is in the image? | dog\n"
                    "What color is it? | brown\n"
                    "Is it indoors? | yes\n"
                    "How many animals? | 1"
                ),
                label_visibility="collapsed",
            )
            run_vqa = st.button("▶  Run VQA Evaluation",
                                disabled=(vqa_image is None or not qa_input.strip()))

        if run_vqa and vqa_image and qa_input.strip():
            pairs = []
            for line in qa_input.strip().splitlines():
                if "|" in line:
                    q, gt = line.split("|", 1)
                    pairs.append({"question": q.strip(), "ground_truth": gt.strip()})

            if not pairs:
                st.warning("No valid Q|A pairs found. Use the format: `question | answer`")
            else:
                results = []
                prog = st.progress(0)
                for i, pair in enumerate(pairs):
                    pred = answer_question(vqa_image, pair["question"])
                    score = compute_vqa_accuracy(pred, pair["ground_truth"])
                    results.append({
                        "Question"    : pair["question"],
                        "Ground Truth": pair["ground_truth"],
                        "Predicted"   : pred,
                        "Score"       : score,
                        "Result"      : "✅ Correct" if score == 1.0 else "❌ Wrong",
                    })
                    prog.progress((i + 1) / len(pairs))
                prog.empty()

                correct = sum(r["Score"] for r in results)
                accuracy = correct / len(results)

                # Accuracy display
                st.markdown("---")
                m1, m2, m3 = st.columns(3)
                m1.metric("VQA Accuracy",  f"{accuracy:.1%}")
                m2.metric("Correct",       f"{int(correct)} / {len(results)}")
                m3.metric("Total Questions", len(results))

                # Accuracy gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode  = "gauge+number+delta",
                    value = accuracy * 100,
                    delta = {"reference": 50},
                    title = {"text": "VQA Accuracy (%)"},
                    number = {"suffix": "%", "font": {"size": 30}},
                    gauge = {
                        "axis" : {"range": [0, 100]},
                        "bar"  : {"color": "#0E9AA7"},
                        "steps": [
                            {"range": [0,  40], "color": "#fee2e2"},
                            {"range": [40, 70], "color": "#fef3c7"},
                            {"range": [70, 100],"color": "#d1fae5"},
                        ],
                    },
                ))
                fig_gauge.update_layout(height=260, margin=dict(t=50, b=10, l=20, r=20))
                st.plotly_chart(fig_gauge)

                # Per-question table
                df_results = pd.DataFrame(results).drop(columns=["Score"])
                st.dataframe(df_results, hide_index=True)

                # Horizontal bar chart
                fig_bar = px.bar(
                    pd.DataFrame(results),
                    x="Score", y="Question",
                    orientation="h",
                    color="Result",
                    color_discrete_map={"✅ Correct": "#0E9AA7", "❌ Wrong": "#EF4444"},
                    title="Per-Question Accuracy",
                )
                fig_bar.update_layout(height=60 + len(results) * 45,
                                      xaxis_range=[0, 1.3],
                                      plot_bgcolor="white")
                st.plotly_chart(fig_bar)

                # Download
                vqa_json = json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "vqa_accuracy": accuracy,
                    "correct": int(correct),
                    "total": len(results),
                    "results": results,
                }, indent=2, default=str)
                st.download_button(
                    "💾 Download VQA Results (JSON)",
                    data=vqa_json,
                    file_name="vqa_eval.json",
                    mime="application/json",
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📖 About":
    st.markdown('<div class="section-header">📖 About This Project</div>',
                unsafe_allow_html=True)

    st.markdown("""
    ### Objective
    Develop a multimodal AI system capable of **image captioning** and
    **visual question answering (VQA)** by jointly processing visual and
    textual information using deep learning.

    ### Models Used
    | Model | Task | Source |
    |-------|------|--------|
    | `BLIP-Image-Captioning-Base` | Caption generation | Salesforce / HuggingFace |
    | `BLIP-VQA-Base` | Visual question answering | Salesforce / HuggingFace |

    ### Evaluation Metrics
    | Metric | Task | Description |
    |--------|------|-------------|
    | BLEU-1 → 4 | Captioning | N-gram overlap with references |
    | METEOR | Captioning | Synonym-aware alignment |
    | CIDEr | Captioning | TF-IDF weighted consensus |
    | Accuracy | VQA | Exact match (normalized) |

    ### References
    1. Vinyals et al. (2015) — *Show and Tell* — IEEE CVPR
    2. Radford et al. (2021) — *CLIP* — ICML
    3. Li et al. (2022) — *BLIP* — ICML
    4. Antol et al. (2015) — *VQA* — IEEE ICCV
    """)

    with st.expander("⚙️ System Info"):
        st.code(f"""
Device      : {DEVICE}
PyTorch     : {torch.__version__}
CUDA        : {torch.cuda.is_available()}
GPU         : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}
        """)