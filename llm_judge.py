"""
LLM-as-Judge for factuality classification.
From Data4Good Colab / Final Deck: 2-step reasoning + relevance-first gating.
"""
from typing import Optional, Tuple

LABELS = ["contradiction", "factual", "irrelevant"]

JUDGE_SYSTEM = """You are a strict evaluator for a QA task.
Your job is to assign exactly ONE of the following labels:
- factual
- contradiction
- irrelevant

Follow this decision procedure exactly:
Step 1 — Relevance:
Decide whether the ANSWER attempts to answer the QUESTION (is it on-topic?).
You may use the CONTEXT only to understand the topic of the question.
If the answer is off-topic, a non-sequitur, or does not attempt to answer the question,
return: irrelevant
and stop.

Step 2 — Correctness (only if relevant):
Use the CONTEXT as the primary evidence.
- If the answer is supported by the context, return: factual
- If the answer is wrong, return: contradiction

If the context is missing or does not clearly address the answer:
- Only return contradiction if the answer is clearly false, internally inconsistent,
or obviously wrong by general knowledge.
- Otherwise, default to: factual

Important rules:
- Do NOT return contradiction just because the context is long, different, or contains
unrelated information.
- An answer that does not address the question is irrelevant, not contradiction.
- If the answer is on-topic and makes an attempt (even if incorrect), it is NOT irrelevant.
Irrelevant is only for off-topic / non-sequitur / no attempt to answer.

Return ONLY the label."""


def make_judge_prompt(question: str, context: str, answer: str) -> str:
    """Format (Q, C, A) for the judge."""
    return f"""QUESTION:
{question or ""}

CONTEXT:
{context or ""}

ANSWER:
{answer or ""}
"""


def judge(
    question: str,
    context: str,
    answer: str,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> Tuple[str, Optional[dict]]:
    """
    Classify (question, context, answer) as factual / contradiction / irrelevant.
    Returns (predicted_label, proba_dict).
    For LLM Judge, proba is None (no probabilities).
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": make_judge_prompt(question, context, answer)},
        ],
        max_tokens=20,
        stream=False,
    )
    content = resp.choices[0].message.content.strip().lower()

    # Normalize to valid label
    for label in LABELS:
        if label in content:
            pred = label
            break
    else:
        pred = "factual"  # fallback

    return pred, None
