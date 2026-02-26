"""
Streamlit Demo - Factuality Detection in AI-Generated Educational Content
Data4Good Competition | The White Hatters

Flow: User question → LLM generates answer → Ensemble classifies as Factual / Incorrect / Irrelevant
"""
import streamlit as st
import json
import os
import random
import pandas as pd
from typing import Optional
from model_pipeline import train_model, predict

st.set_page_config(
    page_title="Factuality Detection Demo",
    page_icon="📚",
    layout="centered"
)

# Custom styling
st.markdown("""
<style>
    .stApp { max-width: 800px; margin: 0 auto; }
    h1 { color: #1e3a5f; font-family: 'Georgia', serif; }
    .pred-factual { color: #0d6b0d; font-weight: bold; }
    .pred-contradiction { color: #b91c1c; font-weight: bold; }
    .pred-irrelevant { color: #b45309; font-weight: bold; }
    .overview-box { background: #f8fafc; padding: 1rem 1.25rem; border-radius: 8px; margin: 1rem 0; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

st.title("📚 Factuality Detection Demo")
st.markdown("**Classifying AI-Generated Educational Content** — *Data4Good Competition | The White Hatters*")

# Overview of the LLM Judge / Ensemble Model
st.markdown("### How it works")
st.markdown("""
<div class="overview-box">
<strong>What the LLM Judge (Ensemble Model) does</strong><br><br>
We combine an <b>LLM</b> to generate answers with an <b>ensemble classifier</b> to judge them. 
You enter a <b>question</b> and <b>context</b> (reference material). An LLM produces an answer. 
The ensemble then classifies whether that answer is:
<ul>
  <li><b>Factual</b> — accurate and relevant to the context</li>
  <li><b>Contradiction</b> — incorrect or conflicts with the context</li>
  <li><b>Irrelevant</b> — does not address the question</li>
</ul>
The ensemble uses Random Forest, XGBoost, and Logistic Regression with features like semantic similarity, 
word overlap, and text structure — trained on 20k+ examples.
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# Load and train model (cached)
@st.cache_resource
def load_model():
    with st.spinner("Loading model (first run trains on sample data, ~30-60 sec)..."):
        with open("data/train.json", "r") as f:
            train_data = json.load(f)
        train_df = pd.DataFrame(train_data)
        train_df.columns = [c.lower() for c in train_df.columns]
        pipeline = train_model(train_df, sample_size=5000)
    return pipeline

def generate_answer_with_llm(question: str, context: str) -> Optional[str]:
    """Use an LLM to generate an answer given question and context."""
    api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer the question using only the provided context. Be concise. If the context doesn't contain the answer, say so briefly."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"}
            ],
            max_tokens=256
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"LLM error: {e}")
        return None

@st.cache_data
def load_preset_examples(n: int = 200):
    """Load sample examples from training data for preset selection."""
    with open("data/train.json", "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df.columns = [c.lower() for c in df.columns]
    return df.sample(n=min(n, len(df)), random_state=42).to_dict("records")

def _show_result(pred: str, proba: dict):
    st.markdown("---")
    st.subheader("Classification")
    pred_lower = pred.lower()
    if "factual" in pred_lower:
        st.markdown(f'**Result:** <span class="pred-factual">✓ {pred}</span>', unsafe_allow_html=True)
    elif "contradiction" in pred_lower:
        st.markdown(f'**Result:** <span class="pred-contradiction">✗ {pred}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'**Result:** <span class="pred-irrelevant">◇ {pred}</span>', unsafe_allow_html=True)
    st.markdown("**Confidence:**")
    for label, p in proba.items():
        pct = p * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        st.markdown(f"  {label}: {bar} {pct:.1f}%")

try:
    pipeline = load_model()
except FileNotFoundError:
    st.error("Training data not found. Please ensure `data/train.json` exists.")
    st.stop()

# Demo inputs
st.subheader("Try it yourself")
st.markdown("Enter a **Question** and **Context** (reference material). We’ll generate an answer with an LLM, then classify it.")

if "preset" not in st.session_state:
    st.session_state.preset = {"question": "", "context": "", "answer": ""}

preset_examples = load_preset_examples()
if st.button("Load random preset from training data"):
    ex = random.choice(preset_examples)
    st.session_state.preset = {
        "question": ex.get("question", ""),
        "context": ex.get("context", ""),
        "answer": ex.get("answer", ""),
    }
    st.rerun()

with st.form("prediction_form"):
    question = st.text_area(
        "Question",
        value=st.session_state.preset["question"],
        placeholder="e.g., What is the capital of France?",
        height=80
    )
    context = st.text_area(
        "Context (reference material)",
        value=st.session_state.preset["context"],
        placeholder="e.g., France is a country in Western Europe. Paris is its capital and largest city.",
        height=100
    )
    use_llm = st.checkbox("Use LLM to generate answer (requires OPENAI_API_KEY)", value=True)
    if not use_llm:
        answer = st.text_area(
            "Answer (if not using LLM)",
            value=st.session_state.preset.get("answer", ""),
            placeholder="e.g., Paris is the capital of France.",
            height=100
        )
    else:
        answer = None
    submitted = st.form_submit_button("Generate & Classify")

if submitted:
    if not (question and context):
        st.warning("Please fill in Question and Context.")
    elif use_llm:
        with st.spinner("Generating answer with LLM..."):
            answer = generate_answer_with_llm(question, context)
        if answer is None:
            st.warning("LLM not configured. Set OPENAI_API_KEY in env or Streamlit secrets, or uncheck 'Use LLM' and enter an answer manually.")
        else:
            st.markdown("**Generated answer:**")
            st.info(answer)
            with st.spinner("Classifying..."):
                pred, proba = predict(pipeline, question, context, answer)
            _show_result(pred, proba)
    else:
        if not answer:
            st.warning("Please enter an answer.")
        else:
            with st.spinner("Classifying..."):
                pred, proba = predict(pipeline, question, context, answer)
            _show_result(pred, proba)

st.markdown("---")
st.caption("Built for the 4th Annual Data4Good Competition | UW Foster MSBA | The White Hatters")
