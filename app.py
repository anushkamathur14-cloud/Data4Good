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
from llm_judge import judge as llm_judge_classify

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
    .overview-box { background: #f8fafc; padding: 1.5rem 1.5rem; border-radius: 8px; margin: 1rem 0; font-size: 1.05rem; line-height: 1.6; color: #1e293b; }
</style>
""", unsafe_allow_html=True)

st.title("📚 Factuality Detection Demo")
st.markdown("**Classifying AI-Generated Educational Content** — *Data4Good Competition | The White Hatters*")

# Overview of the LLM Judge / Ensemble Model
st.markdown("### How it works")
st.markdown("""
<div class="overview-box">
<style>
.flowchart { font-family: system-ui, -apple-system, sans-serif; font-size: 1.05rem; }
.flow-step { display: flex; align-items: center; gap: 8px; margin: 12px 0; flex-wrap: wrap; line-height: 1.5; }
.flow-box { padding: 10px 18px; border-radius: 8px; font-weight: 600; font-size: 1rem; }
.flow-input { background: #e0f2fe; border: 2px solid #0ea5e9; }
.flow-llm { background: #fef3c7; border: 2px solid #f59e0b; }
.flow-clf { background: #d1fae5; border: 2px solid #10b981; }
.flow-out { padding: 8px 14px; border-radius: 6px; margin: 0 6px; font-size: 0.95rem; display: inline-block; font-weight: 600; }
.flow-factual { background: #dcfce7; color: #166534; }
.flow-contra { background: #fee2e2; color: #991b1b; }
.flow-irrel { background: #ffedd5; color: #9a3412; }
.flow-arrow { color: #64748b; font-size: 1.1rem; }
</style>
<div class="flowchart">
<div class="flow-step"><span class="flow-box flow-input">📝 Question</span> + <span class="flow-box flow-input">📄 Context</span></div>
<div class="flow-step"><span class="flow-arrow">↓</span> <span class="flow-box flow-llm">🤖 LLM generates Answer</span> <span style="font-size: 0.95rem; color: #64748b;">(optional)</span></div>
<div class="flow-step"><span class="flow-arrow">↓</span> <span class="flow-box flow-clf">⚖️ Classifier</span> (Ensemble or LLM-as-Judge)</div>
<div class="flow-step"><span class="flow-arrow">↓</span> <span class="flow-out flow-factual">✓ Factual</span> <span class="flow-out flow-contra">✗ Contradiction</span> <span class="flow-out flow-irrel">◇ Irrelevant</span></div>
</div>
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

def generate_answer_with_llm(question: str, context: str, api_key: Optional[str] = None) -> Optional[str]:
    """Use an LLM to generate an answer given question and context."""
    key = api_key or st.session_state.get("openai_api_key", "") or os.environ.get("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else "")
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
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

def _get_api_key() -> str:
    return (
        st.session_state.get("openai_api_key", "")
        or os.environ.get("OPENAI_API_KEY", "")
        or (st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else "")
    )


def _classify(question: str, context: str, answer: str, classifier: str, pipeline):
    """Dispatch to Ensemble or LLM-as-Judge."""
    if "LLM-as-Judge" in classifier:
        api_key = _get_api_key()
        if not api_key:
            raise ValueError("API key required for LLM-as-Judge. Enter it in the expander above.")
        return llm_judge_classify(question, context, answer, api_key)
    return predict(pipeline, question, context, answer)


def _show_result(pred: str, proba: Optional[dict]):
    st.markdown("---")
    st.subheader("Classification")
    pred_lower = pred.lower()
    if "factual" in pred_lower:
        st.markdown(f'**Result:** <span class="pred-factual">✓ {pred}</span>', unsafe_allow_html=True)
    elif "contradiction" in pred_lower:
        st.markdown(f'**Result:** <span class="pred-contradiction">✗ {pred}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'**Result:** <span class="pred-irrelevant">◇ {pred}</span>', unsafe_allow_html=True)
    if proba:
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

with st.expander("Add your OpenAI API key to try the LLM", expanded=False):
    st.caption("Optional. Your key is stored only in this session and never sent anywhere except OpenAI. [Get a key](https://platform.openai.com/api-keys)")
    user_key = st.text_input(
        "OpenAI API key",
        type="password",
        placeholder="sk-...",
        key="openai_api_key_input",
        label_visibility="collapsed"
    )
    if user_key:
        st.session_state.openai_api_key = user_key.strip()
        st.success("Key saved for this session.")
    else:
        st.session_state.openai_api_key = ""

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
    use_llm = st.checkbox("Use LLM to generate answer (enter your API key above)", value=True)
    classifier = st.radio(
        "Classifier",
        ["Ensemble (local ML)", "LLM-as-Judge (OpenAI)"],
        horizontal=True,
        help="Ensemble: trained RF/XGB/LR. LLM-as-Judge: 2-step reasoning prompt (requires API key).",
    )
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
            st.warning("Enter your OpenAI API key in the expander above to use the LLM, or uncheck 'Use LLM' and enter an answer manually.")
        else:
            st.markdown("**Generated answer:**")
            st.info(answer)
            try:
                with st.spinner("Classifying..."):
                    pred, proba = _classify(question, context, answer, classifier, pipeline)
                _show_result(pred, proba)
            except ValueError as e:
                st.warning(str(e))
            except Exception as e:
                st.error(f"Classification error: {e}")
    else:
        if not answer:
            st.warning("Please enter an answer.")
        else:
            try:
                with st.spinner("Classifying..."):
                    pred, proba = _classify(question, context, answer, classifier, pipeline)
                _show_result(pred, proba)
            except ValueError as e:
                st.warning(str(e))
            except Exception as e:
                st.error(f"Classification error: {e}")

st.markdown("---")
st.caption("Built for the 4th Annual Data4Good Competition | UW Foster MSBA | The White Hatters")
