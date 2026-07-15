"""
evaluate.py
-----------
Bonus: CUAD ships with human-annotated gold spans for 41 clause categories.
Two of our three target clauses map onto CUAD categories directly:
  - termination_clause -> "Termination for Convenience"
  - liability_clause   -> "Cap on Liability" / "Uncapped Liability"
(CUAD has no dedicated "Confidentiality" category, so that clause has no
ground truth here and is excluded from this quantitative check.)

We score with token-level F1 against the gold span (standard span-extraction
metric, same family as SQuAD scoring) rather than exact match, since the LLM
may paraphrase clause boundaries slightly.
"""

import re
from collections import Counter


def _tokenize(text: str):
    return re.findall(r"\w+", text.lower())


def token_f1(pred: str, gold: str) -> float:
    pred_tokens, gold_tokens = _tokenize(pred), _tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_f1_against_gold(pred: str, gold_list: list) -> float:
    """A contract can have >1 valid gold span; score against the best match."""
    if pred == "NOT FOUND" or not gold_list:
        return None  
    return max(token_f1(pred, g) for g in gold_list)


def evaluate_batch(results: list) -> dict:
    """
    results: list of dicts each with keys termination_clause, liability_clause,
    gold_termination (list), gold_liability (list).
    Returns mean F1 per clause type over contracts that had gold labels.
    """
    scores = {"termination_clause": [], "liability_clause": []}
    for r in results:
        t_f1 = best_f1_against_gold(r.get("termination_clause", "NOT FOUND"), r.get("gold_termination", []))
        l_f1 = best_f1_against_gold(r.get("liability_clause", "NOT FOUND"), r.get("gold_liability", []))
        if t_f1 is not None:
            scores["termination_clause"].append(t_f1)
        if l_f1 is not None:
            scores["liability_clause"].append(l_f1)

    return {
        "termination_clause_mean_f1": sum(scores["termination_clause"]) / len(scores["termination_clause"])
        if scores["termination_clause"] else None,
        "liability_clause_mean_f1": sum(scores["liability_clause"]) / len(scores["liability_clause"])
        if scores["liability_clause"] else None,
        "n_scored_termination": len(scores["termination_clause"]),
        "n_scored_liability": len(scores["liability_clause"]),
    }
