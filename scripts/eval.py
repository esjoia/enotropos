"""enotropos — Evaluation script (unified).

Computes RAG quality metrics using an LLM-as-judge via DeepSeek V4 Flash:

  - Faithfulness: are claims in the answer supported by the contexts?
  - Answer Relevancy: how relevant is the answer to the question?
  - Context Relevancy: how relevant are retrieved contexts to the question?
  - Ground-truth Accuracy: does the answer agree with the reference answer?
    (only when the dataset item provides ``ground_truth``)

Each metric returns a score in [0, 1]. Uses the shared OpenCode Go LLM client
and the canonical dataset at ``scripts/eval_dataset.json``. No extra
dependencies beyond the core stack (replaces the previous Ragas-based
``run_eval.py`` and the hand-written ``eval_dataset.py``).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from winegpt.config import DATA_DIR, LLM_MODEL
from winegpt.llm import get_llm_client
from winegpt.rag import _extract_gi_names, query_rag

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).resolve().parent / "eval_dataset.json"
DEFAULT_OUT_PATH = DATA_DIR / "eval_results.json"


def _extract_json(raw: str) -> Any:
    """Robust JSON extraction from LLM output (handles markdown fences)."""
    raw = raw.strip()
    if raw.startswith("```"):
        # Strip an opening fence (with optional language) and a closing fence.
        first_nl = raw.find("\n")
        if first_nl != -1:
            raw = raw[first_nl + 1:]
        raw = raw.rsplit("```", 1)[0]
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Metric 1: Faithfulness
# ---------------------------------------------------------------------------

FAITHFULNESS_PROMPT = """Extract all factual claims from the answer below. \
For each claim, decide whether the context SUPPORTS or CONTRADICTS it, \
or if the context is UNVERIFIABLE.

Context:
{context}

Answer:
{answer}

Return a JSON list. Each item must have: \
"claim", "verdict" (one of: SUPPORTED, CONTRADICTED, UNVERIFIABLE).

JSON:"""


def compute_faithfulness(answer: str, contexts: list[str], client: OpenAI) -> float:
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
    supported = sum(
        1 for c in claims if isinstance(c, dict) and c.get("verdict") == "SUPPORTED"
    )
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


def _extract_scale(raw: str, low: int, high: int) -> int | None:
    """Extract a standalone integer score in [low, high] from LLM output."""
    match = re.search(rf"(?<![\d])([{low}-{high}])(?![\d])", raw)
    if not match:
        return None
    return int(match.group(1))


def compute_answer_relevancy(question: str, answer: str, client: OpenAI) -> float:
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

    score = _extract_scale(raw, 1, 5)
    if score is None:
        logger.debug("Answer relevancy parse failed, raw: %s", raw)
        return 0.0
    return (score - 1) / 4.0  # Normalize to [0, 1]


# ---------------------------------------------------------------------------
# Metric 3: Context Relevancy
# ---------------------------------------------------------------------------

CONTEXT_RELEVANCY_PROMPT = """Rate how relevant each context passage is to the question \
on a scale from 1 to 3.
- 1: not relevant
- 2: somewhat relevant
- 3: highly relevant

Question: {question}

Context passages:
{contexts}

Return a JSON list of objects with: "passage" (first 60 chars of the passage), "score" (1, 2, or 3).

JSON:"""


def compute_context_relevancy(question: str, contexts: list[str], client: OpenAI) -> float:
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
    return float(sum(normalized) / len(normalized))


# ---------------------------------------------------------------------------
# Metric 4: Ground-truth Accuracy (replaces Ragas answer_correctness role)
# ---------------------------------------------------------------------------

GROUND_TRUTH_PROMPT = """Compara la resposta donada amb la resposta de referència \
i puntua la seva exactitud factual en una escala d'1 a 5.
- 1: completament incorrecta o contradictòria
- 3: parcialment correcta (alguns fets coincidents, omissions o errors menors)
- 5: correcta i completa, coincident amb la referència

Pregunta: {question}
Resposta de referència: {ground_truth}
Resposta donada: {answer}

Retorna només un sol nombre enter (1-5)."""


def compute_ground_truth_accuracy(
    question: str, answer: str, ground_truth: str, client: OpenAI,
) -> float:
    prompt = GROUND_TRUTH_PROMPT.format(
        question=question, ground_truth=ground_truth, answer=answer,
    )
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4000,
        )
        raw = response.choices[0].message.content or ""
    except Exception as e:
        logger.error("Ground-truth accuracy LLM error: %s", e)
        return 0.0
    score = _extract_scale(raw, 1, 5)
    if score is None:
        logger.debug("Ground-truth accuracy parse failed, raw: %s", raw)
        return 0.0
    return (score - 1) / 4.0


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------


def load_eval_dataset() -> list[dict[str, str]]:
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return cast("list[dict[str, str]]", data)


def run_eval(limit: int = 0, use_ground_truth: bool = True) -> dict[str, Any]:
    questions = load_eval_dataset()
    if limit > 0:
        questions = questions[:limit]

    client = get_llm_client()
    metric_keys = ["faithfulness", "answer_relevancy", "context_relevancy"]
    if use_ground_truth and any("ground_truth" in q for q in questions):
        metric_keys.append("ground_truth_accuracy")
    results: dict[str, list[float]] = {m: [] for m in metric_keys}
    per_question: list[dict[str, Any]] = []

    for i, q in enumerate(questions, 1):
        question = q["question"]
        gi_names = _extract_gi_names(question)
        gi = gi_names[0] if gi_names else ""
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

        row: dict[str, Any] = {
            "question": question, "gi": gi,
            "faithfulness": round(f_score, 3),
            "answer_relevancy": round(ar_score, 3),
            "context_relevancy": round(cr_score, 3),
        }

        gt = q.get("ground_truth")
        if "ground_truth_accuracy" in metric_keys and gt:
            gt_score = compute_ground_truth_accuracy(question, answer, gt, client)
            results["ground_truth_accuracy"].append(gt_score)
            row["ground_truth_accuracy"] = round(gt_score, 3)

        per_question.append(row)
        metrics_str = "  ".join(f"{k}={v:.3f}" for k, v in row.items() if isinstance(v, float))
        logger.info("  %s", metrics_str)

    means = {
        metric: round(sum(scores) / len(scores), 3) if scores else 0.0
        for metric, scores in results.items()
    }

    return {"means": means, "per_question": per_question}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG evaluation with LLM-as-judge")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only N questions")
    parser.add_argument("--json", action="store_true", help="Output full JSON")
    parser.add_argument("--debug", action="store_true", help="Show LLM raw output for debugging")
    parser.add_argument(
        "--out", type=str, default=str(DEFAULT_OUT_PATH),
        help="Path to write the metrics JSON (default: data/eval_results.json).",
    )
    parser.add_argument(
        "--no-ground-truth", action="store_true",
        help="Skip the ground-truth accuracy metric.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    print("=" * 60)
    print("enotropos — RAG Evaluation (LLM-as-judge)")
    print("=" * 60)

    result = run_eval(limit=args.limit, use_ground_truth=not args.no_ground_truth)

    print("\n--- Aggregate Scores (0-1, higher is better) ---")
    for metric, score in result["means"].items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {metric:25s}  {score:.3f}  {bar}")

    print(f"\n  Evaluated {len(result['per_question'])} questions.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"  Saved metrics to {out_path}")

    if args.json:
        print("\n--- Full Results (JSON) ---")
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
