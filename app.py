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
from model_pipeline import train_model, predict, load_pipeline
from llm_judge import judge as llm_judge_classify

st.set_page_config(
    page_title="Live Demo on LLM-based Factuality",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom styling
st.markdown("""
<style>
    .stApp { max-width: 1100px; margin: 0 auto; }
    [data-testid="stVerticalBlock"] > div { gap: 0.5rem; }
    h1 { color: #1e3a5f; font-family: 'Georgia', serif; }
    .pred-factual { color: #0d6b0d; font-weight: bold; }
    .pred-contradiction { color: #b91c1c; font-weight: bold; }
    .pred-irrelevant { color: #b45309; font-weight: bold; }
    .overview-box { background: #f8fafc; padding: 1.5rem 1.5rem; border-radius: 8px; margin: 1rem 0; font-size: 1.05rem; line-height: 1.6; color: #1e293b; }
</style>
""", unsafe_allow_html=True)

st.title("📚 Live Demo on LLM-based Factuality")
st.markdown("**Classifying AI-Generated Educational Content** — *Data4Good Competition | The White Hatters*")

# Session tracker for Ensemble & LLM-as-Judge
if "tracker_ensemble" not in st.session_state:
    st.session_state.tracker_ensemble = {"factual": 0, "contradiction": 0, "irrelevant": 0}
if "tracker_llm" not in st.session_state:
    st.session_state.tracker_llm = {"factual": 0, "contradiction": 0, "irrelevant": 0}

def _track_result(label: str, classifier: str) -> None:
    key = label.lower()
    if "factual" in key:
        key = "factual"
    elif "contradiction" in key:
        key = "contradiction"
    else:
        key = "irrelevant"
    if "LLM-as-Judge" in classifier:
        st.session_state.tracker_llm[key] += 1
    else:
        st.session_state.tracker_ensemble[key] += 1

def _tracker_pcts(tracker: dict) -> dict:
    total = sum(tracker.values())
    if total == 0:
        return {"factual": 0, "contradiction": 0, "irrelevant": 0}
    return {k: round(100 * v / total, 1) for k, v in tracker.items()}

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

# Load model: use pre-trained file if present, else train (cached)
MODEL_PATH = "model_pipeline.joblib"

@st.cache_resource
def load_model():
    pipeline = load_pipeline(MODEL_PATH)
    if pipeline is not None:
        return pipeline
    with st.spinner("Loading model (training on sample data, ~30-60 sec)..."):
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
        _show_friendly_error(e)
        return None

@st.cache_data
def load_preset_examples(n: int = 200):
    """Load sample examples from training data for preset selection."""
    with open("data/train.json", "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df.columns = [c.lower() for c in df.columns]
    return df.sample(n=min(n, len(df)), random_state=42).to_dict("records")

def _show_friendly_error(e: Exception) -> None:
    """Show a user-friendly error message instead of raw API/JSON output."""
    err = str(e).lower()
    if "401" in err or "invalid_api_key" in err or "incorrect api key" in err:
        st.error("Invalid OpenAI API key. Please check your key in the expander above and ensure it's correct. [Get a key](https://platform.openai.com/account/api-keys)")
    elif "429" in err or "rate limit" in err:
        st.error("Rate limit exceeded. Please wait a moment and try again.")
    elif "api" in err and ("key" in err or "auth" in err):
        st.error("API key issue. Please verify your OpenAI API key in the expander above.")
    else:
        st.error(f"Error: {e}")


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


def _show_result(pred: str, proba: Optional[dict], classifier: str = "", title: str = "Classification"):
    st.markdown("---")
    st.subheader(title)
    pred_lower = pred.lower()
    if "factual" in pred_lower:
        st.markdown(f'**Result:** <span class="pred-factual">✓ {pred}</span>', unsafe_allow_html=True)
    elif "contradiction" in pred_lower:
        st.markdown(f'**Result:** <span class="pred-contradiction">✗ {pred}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'**Result:** <span class="pred-irrelevant">◇ {pred}</span>', unsafe_allow_html=True)

    res_col1, res_col2 = st.columns([1, 1])
    with res_col1:
        if proba:
            st.markdown("**Confidence**")
            for label, p in proba.items():
                pct = p * 100
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                st.markdown(f"  {label}: {bar} {pct:.1f}%")
    with res_col2:
        if proba:
            best_label = max(proba, key=proba.get)
            best_pct = proba[best_label] * 100
            st.markdown("**Note**")
            st.info(f"Most likely **{best_label}** {best_pct:.1f}% (based on highest percentile)")
        else:
            st.markdown("**Note**")
            st.info(f"Classification: **{pred}** (LLM single-label)")

# Demo inputs
st.subheader("Try it yourself")
st.markdown("1) Load a sample, or enter Question & Context. 2) Get an answer (generate with LLM or enter manually). 3) Capture it, then classify.")

with st.expander("🔑 OpenAI API key (for LLM features)", expanded=False):
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
if st.button("📋 Load sample"):
    ex = random.choice(preset_examples)
    st.session_state.preset = {
        "question": ex.get("question", ""),
        "context": ex.get("context", ""),
        "answer": ex.get("answer", ""),
    }
    st.session_state.generated_answer = ex.get("answer", "")
    st.session_state.generated_q = ex.get("question", "")
    st.session_state.generated_c = ex.get("context", "")
    st.session_state.just_loaded_sample = True
    st.rerun()

with st.form("prediction_form"):
    col_q, col_c = st.columns(2)
    with col_q:
        question = st.text_area(
            "Question",
            value=st.session_state.preset["question"],
            placeholder="e.g., What is the capital of France?",
            height=280,
            help="The question to evaluate."
        )
    with col_c:
        context = st.text_area(
            "Context (reference)",
            value=st.session_state.preset["context"],
            placeholder="e.g., France is a country in Western Europe. Paris is its capital and largest city.",
            height=280,
            help="Reference material the answer should be based on."
        )
    st.markdown("**Answer** — *generate with LLM or enter manually*")
    manual_answer = st.text_area(
        "Manual answer",
        value="",
        height=120,
        placeholder="Type your answer here, then click Capture answer...",
        key="manual_answer_input",
        label_visibility="collapsed",
    )
    btn_gen, btn_capture = st.columns(2)
    with btn_gen:
        submitted_gen = st.form_submit_button("▶ Generate with LLM")
    with btn_capture:
        submitted_manual = st.form_submit_button("✓ Capture answer (use manual)")

if submitted_gen:
    if not (question and context):
        st.warning("Please fill in Question and Context.")
    else:
        with st.spinner("Generating answer with LLM..."):
            answer = generate_answer_with_llm(question, context)
        if answer is None:
            st.warning("Enter your OpenAI API key in the expander above to generate an answer.")
        else:
            st.session_state.generated_answer = answer
            st.session_state.generated_q = question
            st.session_state.generated_c = context
            st.success("✓ **New answer generated** and ready for classification.")

if submitted_manual:
    if not (question and context):
        st.warning("Please fill in Question and Context first.")
    elif not manual_answer.strip():
        st.warning("Enter an answer manually, or generate with LLM.")
    else:
        st.session_state.generated_answer = manual_answer.strip()
        st.session_state.generated_q = question
        st.session_state.generated_c = context
        st.session_state.just_loaded_sample = False
        st.success("✓ **Answer captured** — ready for classification.")

if st.session_state.get("just_loaded_sample"):
    st.success("✓ **Sample loaded** — question, context, and answer ready.")
    st.session_state.just_loaded_sample = False
if "generated_answer" in st.session_state and st.session_state.generated_answer:
    st.markdown("**Answer** — *you can edit before classifying*")
    answer_edited = st.text_area(
        "Answer",
        value=st.session_state.generated_answer,
        height=150,
        key="answer_edit",
        label_visibility="collapsed",
    )
    st.session_state.generated_answer = answer_edited  # keep in sync
    st.caption(f"✓ **Answer captured** ({len(answer_edited)} chars) — ready for classification")
    st.markdown("**Choose classifier** — *run both to verify they work*")
    classifier = st.radio(
        "Classifier",
        ["Ensemble (local ML)", "LLM-as-Judge (OpenAI)"],
        horizontal=True,
        key="classifier_choice",
        help="Run this answer through Ensemble or LLM-as-Judge.",
    )
    run_one = st.button("Classify with chosen model")
    run_both = st.button("Run both classifiers (verify both work)")
    if run_one or run_both:
        try:
            pipeline = load_model()
        except FileNotFoundError:
            st.error("Training data not found. Please ensure `data/train.json` exists.")
        else:
            q = st.session_state.generated_q
            c = st.session_state.generated_c
            a = answer_edited
            if run_one:
                try:
                    with st.spinner("Classifying..."):
                        pred, proba = _classify(q, c, a, classifier, pipeline)
                    _track_result(pred, classifier)
                    _show_result(pred, proba, classifier)
                except ValueError as e:
                    st.warning(str(e))
                except Exception as e:
                    _show_friendly_error(e)
            elif run_both:
                results = []
                with st.spinner("Running both classifiers..."):
                    for clf in ["Ensemble (local ML)", "LLM-as-Judge (OpenAI)"]:
                        try:
                            pred, proba = _classify(q, c, a, clf, pipeline)
                            _track_result(pred, clf)
                            results.append((clf, pred, proba, None))
                        except ValueError as e:
                            results.append((clf, None, None, str(e)))
                        except Exception as e:
                            err = str(e).lower()
                            if "401" in err or "invalid_api_key" in err or "incorrect api key" in err:
                                results.append((clf, None, None, "Invalid API key. Add your OpenAI key in the expander above."))
                            else:
                                results.append((clf, None, None, str(e)))
                ok_count = sum(1 for r in results if r[1] is not None)
                if ok_count == 2:
                    st.success("✓ **Both models ran** — Ensemble and LLM-as-Judge classified successfully.")
                    preview = (a[:80] + "…") if len(a) > 80 else a
                    st.caption(f"Both classifiers used this answer ({len(a)} chars):")
                    st.code(preview, language=None)
                elif ok_count == 1:
                    st.warning("One model ran; the other had an error. See results below.")
                bc1, bc2 = st.columns(2)
                with bc1:
                    clf, pred, proba, err = results[0]
                    if err:
                        st.warning(f"**{clf}:** {err}")
                    elif pred is not None:
                        _show_result(pred, proba, clf, title=clf)
                with bc2:
                    clf, pred, proba, err = results[1]
                    if err:
                        st.warning(f"**{clf}:** {err}")
                    elif pred is not None:
                        _show_result(pred, proba, clf, title=clf)

st.markdown("---")
st.markdown("**Session tracker** — *% by classifier*")
tr_col1, tr_col2 = st.columns(2)
with tr_col1:
    e_total = sum(st.session_state.tracker_ensemble.values())
    e_pct = _tracker_pcts(st.session_state.tracker_ensemble)
    st.markdown("**Ensemble** (n={})".format(e_total))
    if e_total > 0:
        st.markdown("✓ Factual {:.1f}%".format(e_pct["factual"]))
        st.progress(e_pct["factual"] / 100, text=None)
        st.markdown("✗ Contradiction {:.1f}%".format(e_pct["contradiction"]))
        st.progress(e_pct["contradiction"] / 100, text=None)
        st.markdown("◇ Irrelevant {:.1f}%".format(e_pct["irrelevant"]))
        st.progress(e_pct["irrelevant"] / 100, text=None)
    else:
        st.caption("No classifications yet")
with tr_col2:
    l_total = sum(st.session_state.tracker_llm.values())
    l_pct = _tracker_pcts(st.session_state.tracker_llm)
    st.markdown("**LLM-as-Judge** (n={})".format(l_total))
    if l_total > 0:
        st.markdown("✓ Factual {:.1f}%".format(l_pct["factual"]))
        st.progress(l_pct["factual"] / 100, text=None)
        st.markdown("✗ Contradiction {:.1f}%".format(l_pct["contradiction"]))
        st.progress(l_pct["contradiction"] / 100, text=None)
        st.markdown("◇ Irrelevant {:.1f}%".format(l_pct["irrelevant"]))
        st.progress(l_pct["irrelevant"] / 100, text=None)
    else:
        st.caption("No classifications yet")

st.markdown("---")
st.caption("Built for the 4th Annual Data4Good Competition | UW Foster MSBA | The White Hatters")
