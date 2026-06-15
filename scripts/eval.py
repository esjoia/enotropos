"""enotropos — Evaluation script.

Computes RAG quality metrics using LLM-as-judge via DeepSeek V4 Flash:
  - Faithfulness: Are claims in the answer supported by contexts?
  - Answer Relevancy: How relevant is the answer to the question?
  - Context Relevancy: How relevant are retrieved contexts to the question?

Each metric returns a score in [0, 1]. Uses the existing OpenAI client
(OpenCode Go) + Jina embeddings. No extra dependencies.
"""
from __future__ import annotations

import json
import logging
import re
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from winegpt.config import LLM_BASE_URL, LLM_MODEL, OPENCODE_GO_API_KEY
from winegpt.embed import embed_texts
from winegpt.rag import query_rag

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

EVAL_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_questions.json"


def get_llm_client():
    from openai import OpenAI

    return OpenAI(base_url=LLM_BASE_URL, api_key=OPENCODE_GO_API_KEY)


def _extract_json(raw: str) -> Any:
    """Robust JSON extraction from LLM output (handles markdown fences)."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines)
    elif "```json" in raw:
        raw = raw.split("```json", 1)[1].split("```", 1)[0]
    return json.loads(raw)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    da = sum(x * x for x in a) ** 0.5
    db = sum(x * x for x in b) ** 0.5
    if da == 0 or db == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (da * db)


# ---------------------------------------------------------------------------
# Metric 1: Faithfulness
# ---------------------------------------------------------------------------

FAITHFULNESS_PROMPT = """Extract all factual claims from the answer below. For each claim, decide whether the context SUPPORTS or CONTRADICTS it, or if the context is UNVERIFIABLE.

Context:
{context}

Answer:
{answer}

Return a JSON list. Each item must have: "claim", "verdict" (one of: SUPPORTED, CONTRADICTED, UNVERIFIABLE).

JSON:"""


def compute_faithfulness(answer: str, contexts: list[str], client) -> float:
    context_text = "\n\n---\n\n".join(contexts)
    # Truncate context to ~3000 chars to avoid token limits
    if len(context_text) > 3000:
        context_text = context_text[:3000] + "\n\n[... truncated ...]"

    prompt = FAITHFULNESS_PROMPT.format(context=context_text, answer=answer)

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=16000,
        )
        raw = response.choices[0].message.content or ""
    except Exception as e:
        logger.error("Faithfulness LLM error: %s", e)
        return 0.0

    if not raw:
        logger.debug("Faithfulness empty response (thinking loop)")
        return 0.0

    try:
        claims = _extract_json(raw)
    except (json.JSONDecodeError, ValueError):
        logger.debug("Faithfulness parse failed, raw: %s", raw[:200])
        return 0.0

    if not claims or not isinstance(claims, list):
        return 0.0
    supported = sum(1 for c in claims if isinstance(c, dict) and c.get("verdict") == "SUPPORTED")
    return supported / len(claims)


# ---------------------------------------------------------------------------
# Metric 2: Answer Relevancy
# ---------------------------------------------------------------------------

ANSWER_RELEVANCY_PROMPT = """Rate how well the answer addresses the question on a scale from 1 to 5.
- 1: completely irrelevant, does not address the question at all
- 3: somewhat relevant, partially addresses the question
- 5: perfectly relevant, directly and fully answers the question

Question: {question}
Answer: {answer}

Return only a single integer (1-5)."""


def compute_answer_relevancy(question: str, answer: str, client) -> float:
    prompt = ANSWER_RELEVANCY_PROMPT.format(question=question, answer=answer)

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4000,
        )
        raw = response.choices[0].message.content or ""
    except Exception as e:
        logger.error("Answer relevancy LLM error: %s", e)
        return 0.0

    match = re.search(r"[1-5]", raw)
    if not match:
        logger.debug("Answer relevancy parse failed, raw: %s", raw)
        return 0.0
    return (int(match.group(0)) - 1) / 4.0  # Normalize to [0, 1]


# ---------------------------------------------------------------------------
# Metric 3: Context Relevancy
# ---------------------------------------------------------------------------

CONTEXT_RELEVANCY_PROMPT = """Rate how relevant each context passage is to the question on a scale from 1 to 3.
- 1: not relevant
- 2: somewhat relevant
- 3: highly relevant

Question: {question}

Context passages:
{contexts}

Return a JSON list of objects with: "passage" (first 60 chars of the passage), "score" (1, 2, or 3).

JSON:"""


def compute_context_relevancy(question: str, contexts: list[str], client) -> float:
    # Format numbered contexts (truncated)
    parts = []
    for i, ctx in enumerate(contexts, 1):
        snippet = ctx[:300].replace("\n", " ") + ("..." if len(ctx) > 300 else "")
        parts.append(f"[{i}] {snippet}")
    context_text = "\n".join(parts)

    prompt = CONTEXT_RELEVANCY_PROMPT.format(question=question, contexts=context_text)

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=16000,
        )
        raw = response.choices[0].message.content or ""
    except Exception as e:
        logger.error("Context relevancy LLM error: %s", e)
        return 0.0

    if not raw:
        logger.debug("Context relevancy empty response (thinking loop)")
        return 0.0

    try:
        scores = _extract_json(raw)
    except (json.JSONDecodeError, ValueError):
        logger.debug("Context relevancy parse failed, raw: %s", raw[:200])
        return 0.0

    if not scores or not isinstance(scores, list):
        return 0.0

    values = []
    for s in scores:
        if isinstance(s, dict) and isinstance(s.get("score"), (int, float)):
            values.append(s["score"])
    if not values:
        return 0.0
    # Normalize 1-3 to 0-1
    normalized = [(v - 1) / 2.0 for v in values]
    return statistics.mean(normalized)


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------


def load_eval_questions() -> list[dict[str, str]]:
    with open(EVAL_PATH, encoding="utf-8") as f:
        return json.load(f)


def run_eval(limit: int = 0) -> dict[str, Any]:
    questions = load_eval_questions()
    if limit > 0:
        questions = questions[:limit]

    client = get_llm_client()
    results: dict[str, list[float]] = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_relevancy": [],
    }
    per_question: list[dict[str, Any]] = []

    for i, q in enumerate(questions, 1):
        question = q["question"]
        gi = q.get("gi", "")
        logger.info("[%d/%d] %s — %s", i, len(questions), gi, question[:80])

        rag_result = query_rag(question)
        answer = rag_result["answer"]
        chunks = rag_result["context_chunks"]

        if not answer or not chunks:
            logger.warning("Skipping: no answer or no chunks")
            per_question.append({
                "question": question, "gi": gi,
                "faithfulness": None, "answer_relevancy": None,
                "context_relevancy": None, "skipped": True,
            })
            continue

        contexts = [c.get("document", "") for c in chunks if c.get("document")]

        f_score = compute_faithfulness(answer, contexts, client)
        ar_score = compute_answer_relevancy(question, answer, client)
        cr_score = compute_context_relevancy(question, contexts, client)

        results["faithfulness"].append(f_score)
        results["answer_relevancy"].append(ar_score)
        results["context_relevancy"].append(cr_score)

        per_question.append({
            "question": question, "gi": gi,
            "faithfulness": round(f_score, 3),
            "answer_relevancy": round(ar_score, 3),
            "context_relevancy": round(cr_score, 3),
        })

        logger.info(
            "  faithfulness=%.3f  answer_relevancy=%.3f  context_relevancy=%.3f",
            f_score, ar_score, cr_score,
        )

    means = {
        metric: round(statistics.mean(scores), 3) if scores else 0.0
        for metric, scores in results.items()
    }

    return {"means": means, "per_question": per_question}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="RAG evaluation with LLM-as-judge")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only N questions")
    parser.add_argument("--json", action="store_true", help="Output full JSON")
    parser.add_argument("--debug", action="store_true", help="Show LLM raw output for debugging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    print("=" * 60)
    print("enotropos — RAG Evaluation (LLM-as-judge)")
    print("=" * 60)

    result = run_eval(limit=args.limit)

    print("\n--- Aggregate Scores (0-1, higher is better) ---")
    for metric, score in result["means"].items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {metric:25s}  {score:.3f}  {bar}")

    print(f"\n  Evaluated {len(result['per_question'])} questions.")

    if args.json:
        print("\n--- Full Results (JSON) ---")
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
