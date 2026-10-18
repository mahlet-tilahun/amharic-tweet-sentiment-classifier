import os
import sys
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from utils import clean_text  # noqa: E402

LOCAL_MODEL_PATH = ROOT / "models" / "best_model"
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str(LOCAL_MODEL_PATH)
    if (LOCAL_MODEL_PATH / "config.json").is_file()
    and (LOCAL_MODEL_PATH / "model.safetensors").is_file()
    else "mahlet-tilahun/amharic-sentiment-model",
)


st.set_page_config(page_title="Amharic Sentiment Classifier", layout="centered")


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return model, tokenizer


def classify_text(text: str):
    if not text or not text.strip():
        return {}, "Please type an Amharic tweet."

    cleaned = clean_text(text)
    model, tokenizer = load_model()
    inputs = tokenizer(cleaned, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)

    labels = {0: "negative", 1: "neutral", 2: "positive"}
    scores = {labels[i]: float(probs[i].item()) for i in range(len(probs))}

    ethiopic = sum(1 for ch in cleaned if "ሀ" <= ch <= "፿")
    letters = sum(1 for ch in cleaned if ch.isalpha())
    note = f"Processed text: {cleaned}"
    if letters and ethiopic / letters < 0.5:
        note += "\n\n⚠️ This input is mostly not written in Ge'ez script, so the prediction may be unreliable."
    if max(scores.values()) < 0.5:
        note += "\n\nℹ️ The model is not confident (no class above 50%)."

    return scores, note


st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f4f8ff 0%, #edfaf4 45%, #f9f4ff 100%);
    }
    div[data-testid="stAppViewContainer"] > div:first-child {
        padding-top: 1rem;
    }
    .block-container {
        max-width: 980px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #1f3b5b;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    div[data-testid="stHeading"] {
        padding-top: 1rem;
        margin-top: 0.6rem;
    }
    .stTextArea textarea {
        border-radius: 14px;
        border: 1px solid #d8e3f0;
        background: rgba(255,255,255,0.9);
        box-shadow: 0 8px 24px rgba(31, 59, 91, 0.08);
    }
    .stButton > button {
        border-radius: 12px;
        background: linear-gradient(90deg, #1d4ed8, #2563eb);
        color: white;
        font-weight: 600;
        border: none;
        box-shadow: 0 10px 22px rgba(37, 99, 235, 0.25);
        height: 3em;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #1e40af, #1d4ed8);
    }
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    .example-card {
        background: rgba(255,255,255,0.8);
        border: 1px solid #dfeaf7;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
    }
    .example-label {
        font-weight: 700;
        color: #234a72;
        margin-bottom: 0.35rem;
        display: block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Amharic Tweet Sentiment Classifier")
st.caption("Negative • Neutral • Positive")

example_text = "አማርኛ በጣም ጥሩ እና አስደሳች ነው"
text = st.text_area("Type or paste an Amharic tweet", value=example_text, height=180)
classify_clicked = st.button("Classify tweet", use_container_width=True)

examples = [
    ("Positive", "ዛሬ ስብሰባው በጣም ጥሩ እና አስደሳች ነው", "Today’s meeting was very good and exciting."),
    ("Neutral", "በዚህ ስብሰባ ላይ ብዙ ሰዎች ተገኙ እና ሂደቱ እንደተለመደው ተካሄደ", "Many people attended this meeting and the process went as usual."),
    ("Negative", "ይህ ስብሰባ የማይጠቅም እና አስቸጋሪ ነው", "This meeting is useless and stressful."),
]

st.markdown("### Quick examples")
quick_cols = st.columns(3)
for i, (label, amharic, english) in enumerate(examples):
    with quick_cols[i]:
        box_color = {
            "Positive": "#e8f8ee",
            "Negative": "#fde7e7",
            "Neutral": "#fff3d6",
        }.get(label, "#eef2ff")
        st.markdown(
            f"""
            <div style="background: {box_color}; border: 1px solid rgba(0,0,0,0.06); border-radius: 12px; padding: 0.9rem; min-height: 210px;">
                <div style="font-weight: 700; margin-bottom: 0.5rem; color: #0f172a;">{label}</div>
                <div style="font-size: 0.95rem; margin-bottom: 0.6rem; color: #0f172a;">{amharic}</div>
                <div style="font-size: 0.8rem; color: #475569;">English: {english}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

if classify_clicked:
    scores, note = classify_text(text)
    if not scores:
        st.warning(note)
    else:
        best_label, best_score = max(scores.items(), key=lambda x: x[1])
        sentiment_colors = {
            "negative": "#dc2626",
            "neutral": "#f59e0b",
            "positive": "#16a34a",
        }
        badge_color = sentiment_colors.get(best_label.lower(), "#64748b")
        st.markdown(
            f"""
            <div style="background: rgba(255,255,255,0.88); border: 1px solid #dfeaf7; border-radius: 16px; padding: 1.1rem 1.2rem; box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06); margin-top: 1rem; margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; justify-content: space-between; gap: 1rem; margin-bottom: 0.5rem;">
                    <div style="font-size: 0.9rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.08em;">Prediction</div>
                    <span style="background: {badge_color}; color: white; border-radius: 999px; padding: 0.38rem 0.8rem; font-weight: 700; text-transform: capitalize;">{best_label}</span>
                </div>
                <div style="font-size: 1.7rem; font-weight: 700; color: #0f172a;">{best_score:.2%} confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("Probability breakdown")
        st.bar_chart({label: score for label, score in scores.items()})

        st.info(note)
