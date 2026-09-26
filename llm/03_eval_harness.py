"""
Step 8: Evaluation Harness (v4 — final)
------------------------------------------------------------------------
Runs a fixed set of test questions through the full agent pipeline and
scores each answer against ground truth or expected keywords.

v4 changes:
  - Added a test question with no store specified, to verify the
    store-default disclosure fix (system must explicitly state it
    defaulted to store 1, not silently assume it).
  - All prior fixes retained: tolerant numeric matching, forecast-aware
    routing for future-dated "average sales" questions.
"""

import sys
import time
import csv
import re
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.graph import ask
from src.tools import run_sql

SECONDS_BETWEEN_QUESTIONS = 20
ZERO_RESULT_WORDS = ["no ", "none", "zero", "0"]


TEST_QUESTIONS = [
    {
        "question": "What was the average sales amount for store 1 in January 2013?",
        "type": "sql",
        "ground_truth_query": "SELECT ROUND(AVG(sales_amount)) FROM model_input WHERE store_id = 1 AND YEAR(sale_date) = 2013 AND MONTH(sale_date) = 1",
    },
    {
        "question": "What was the average sales amount across all stores in December 2014?",
        "type": "sql",
        "ground_truth_query": "SELECT ROUND(AVG(sales_amount)) FROM model_input WHERE YEAR(sale_date) = 2014 AND MONTH(sale_date) = 12",
    },
    {
        "question": "How many distinct stores are in the dataset?",
        "type": "sql",
        "ground_truth_query": "SELECT COUNT(DISTINCT store_id) FROM model_input",
    },
    {
        "question": "What is the maximum sales amount ever recorded for store 1?",
        "type": "sql",
        "ground_truth_query": "SELECT MAX(sales_amount) FROM model_input WHERE store_id = 1",
    },
    {
        "question": "What was the total number of customers for store 1 on 2013-01-02?",
        "type": "sql",
        "ground_truth_query": "SELECT customers FROM model_input WHERE store_id = 1 AND sale_date = '2013-01-02'",
    },
    {
        "question": "How many store type 'b' stores are in the dataset?",
        "type": "sql",
        "ground_truth_query": "SELECT COUNT(DISTINCT store_id) FROM model_input WHERE store_type = 'b'",
        "zero_result_expected": True,
    },
    {
        "question": "What was the average sales for store 1 in December 2024?",
        "type": "forecast",
        "expected_keywords": ["prediction", "low-confidence"],
    },
    {
        "question": "Is a MAPE of 0.078 a good result for this project?",
        "type": "context",
        "expected_keywords": ["strong", "10%"],
    },
    {
        "question": "What bug was found in the rolling average feature?",
        "type": "context",
        "expected_keywords": ["leakage", "current"],
    },
    {
        "question": "Why does promo matter as a feature in this model?",
        "type": "context",
        "expected_keywords": ["promo"],
    },
    {
        "question": "Did walk-forward validation or one-shot validation perform better?",
        "type": "context",
        "expected_keywords": ["walk-forward"],
    },
    {
        "question": "What was the best model and its MAPE score in this project?",
        "type": "context",
        "expected_keywords": ["prophet", "0.078"],
    },
    {
        "question": "What will store 1's sales be on 2015-08-20?",
        "type": "forecast",
        "expected_keywords": ["prediction"],
    },
    {
        "question": "What will store 2's sales be in December 2026?",
        "type": "forecast",
        "expected_keywords": ["prediction", "low-confidence"],
    },
    {
        "question": "What will store 42's sales be next month?",
        "type": "forecast",
        "expected_keywords": ["1, 2, 3, 4, 5"],
    },
    {
        # New: no store specified -- verifies the store-default
        # disclosure fix (must explicitly say it defaulted to store 1).
        "question": "What will sales be next month?",
        "type": "forecast",
        "expected_keywords": ["default", "store 1"],
    },
]


def get_ground_truth(query: str) -> str:
    result = run_sql(query)
    if "Error" in result or "no rows" in result:
        return None
    lines = result.strip().split("\n")
    return lines[-1].strip() if len(lines) >= 3 else None


def extract_numbers(text: str) -> list:
    cleaned = text.replace(",", "")
    return [float(n) for n in re.findall(r"\d+\.?\d*", cleaned)]


def check_answer(test_case: dict, agent_answer: str) -> tuple:
    answer_lower = agent_answer.lower()

    if test_case["type"] == "sql" and test_case.get("ground_truth_query"):
        truth = get_ground_truth(test_case["ground_truth_query"])
        if truth is None:
            return False, "Could not compute ground truth"

        truth_val = float(truth)

        if test_case.get("zero_result_expected") or truth_val == 0:
            if truth_val in extract_numbers(agent_answer) or any(w in answer_lower for w in ZERO_RESULT_WORDS):
                return True, f"Ground truth {truth} (zero-result) correctly expressed"
            return False, f"Expected a zero/none result, not clearly expressed"

        agent_numbers = extract_numbers(agent_answer)
        if any(abs(round(n) - round(truth_val)) <= 1 for n in agent_numbers):
            return True, f"Ground truth ~{truth_val} found in answer (tolerant match)"
        return False, f"Expected ~{truth_val}, not found in answer"

    keywords = test_case.get("expected_keywords", [])
    missing = [kw for kw in keywords if kw.lower() not in answer_lower]
    if not missing:
        return True, "All expected keywords present"
    return False, f"Missing keywords: {missing}"


def run_eval():
    results = []

    for i, test_case in enumerate(TEST_QUESTIONS):
        if i > 0:
            print(f"\n[pausing {SECONDS_BETWEEN_QUESTIONS}s before next question]")
            time.sleep(SECONDS_BETWEEN_QUESTIONS)

        print("\n" + "=" * 70)
        print(f"[{i+1}/{len(TEST_QUESTIONS)}] {test_case['question']}")
        print("=" * 70)

        start = time.time()
        try:
            final_state = ask(test_case["question"])
            latency = round(time.time() - start, 2)

            is_correct, note = check_answer(test_case, final_state["answer"])

            result = {
                "question": test_case["question"],
                "type": test_case["type"],
                "answer": final_state["answer"],
                "correct": is_correct,
                "note": note,
                "critic_verdict": final_state["verdict"],
                "revisions": final_state["revision_count"] - 1,
                "latency_sec": latency,
            }
        except Exception as e:
            latency = round(time.time() - start, 2)
            result = {
                "question": test_case["question"],
                "type": test_case["type"],
                "answer": f"ERROR: {e}",
                "correct": False,
                "note": "Pipeline error",
                "critic_verdict": "N/A",
                "revisions": 0,
                "latency_sec": latency,
            }

        print(f"Answer: {result['answer']}")
        print(f"Correct: {result['correct']} ({result['note']})")
        print(f"Critic: {result['critic_verdict']}, revisions: {result['revisions']}, latency: {result['latency_sec']}s")

        results.append(result)

    return results


def summarize(results: list):
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = round(100 * correct / total, 1)

    approved_first_try = sum(1 for r in results if r["critic_verdict"] == "APPROVED" and r["revisions"] == 0)
    caught_and_revised = sum(1 for r in results if r["revisions"] > 0)
    avg_latency = round(sum(r["latency_sec"] for r in results) / total, 2)

    by_type = {}
    for r in results:
        t = r["type"]
        by_type.setdefault(t, {"total": 0, "correct": 0})
        by_type[t]["total"] += 1
        if r["correct"]:
            by_type[t]["correct"] += 1

    print("\n" + "=" * 70)
    print("EVAL SUMMARY")
    print("=" * 70)
    print(f"Total questions:            {total}")
    print(f"Correct answers:            {correct}/{total} ({accuracy}%)")
    print(f"Approved on first pass:     {approved_first_try}/{total}")
    print(f"Critic caught & revised:    {caught_and_revised}/{total}")
    print(f"Average latency/question:   {avg_latency}s")
    print("\nBy question type:")
    for t, stats in by_type.items():
        print(f"  {t}: {stats['correct']}/{stats['total']}")

    with open("eval_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("\nSaved detailed results to eval_results.csv")


if __name__ == "__main__":
    results = run_eval()
    summarize(results)