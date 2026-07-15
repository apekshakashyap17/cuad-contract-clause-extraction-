"""
main.py
-------
End-to-end pipeline entrypoint.

  1. Load + sample 50 CUAD contracts, normalize text.
  2. For each contract: extract termination/confidentiality/liability clauses
     (Part A) and generate a 100-150 word summary (Part B).
  3. Write results to output/results.csv and output/results.json.
  4. (Bonus) Build a FAISS semantic search index over the extracted clauses.
  5. (Bonus) Score termination/liability extraction against CUAD's own gold
     labels and print an accuracy report.

Usage:
    python main.py --n 50 --limit 5   # --limit for a quick smoke test on 5 contracts
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from src import data_loader, evaluate, preprocessing, semantic_search
from src.batch_extractor import extract_all as batch_extract_all
from src.clause_extractor import extract_clauses
from src.combined_extractor import extract_and_summarize
from src.summarizer import summarize_contract

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def run_pipeline(n_contracts: int = 50, limit: int = None, build_index: bool = True,
                  mode: str = "classic", batch_size: int = 5):
    """
    mode:
      classic  - 2 LLM calls per contract (clause_extractor + summarizer separately).
                 Most "focused" prompting, but most calls: 50 contracts = 100 calls.
      combined - 1 LLM call per contract (clauses + summary together).
                 50 contracts = 50 calls. Use this to fit OpenRouter's free 50/day cap.
      batch    - 1 LLM call per N contracts (batch_size), all sharing one prompt.
                 50 contracts / batch_size=5 = 10 calls total. Lowest call count,
                 highest risk of the model mixing up contracts - check eval_report.json
                 afterward to confirm accuracy held up.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading {n_contracts} contracts from CUAD ...")
    contracts = data_loader.sample_subset(n=n_contracts)
    data_loader.save_subset_to_disk(contracts)
    if limit:
        contracts = contracts[:limit]

    results = []

    if mode == "batch":
        print(f"Running in BATCH mode: {len(contracts)} contracts in groups of {batch_size} "
              f"= {-(-len(contracts) // batch_size)} total LLM calls")
        batch_results = batch_extract_all(contracts, batch_size=batch_size)
        gold_by_id = {c.contract_id: c for c in contracts}
        for r in batch_results:
            c = gold_by_id[r["contract_id"]]
            results.append({
                "contract_id": r["contract_id"],
                "summary": r["summary"].strip() if r["summary"] else "",
                "termination_clause": r["termination_clause"],
                "confidentiality_clause": r["confidentiality_clause"],
                "liability_clause": r["liability_clause"],
                "gold_termination": c.gold_termination,
                "gold_liability": c.gold_liability,
                "seconds": None,  
            })
    else:
        for i, c in enumerate(contracts, 1):
            print(f"[{i}/{len(contracts)}] Processing: {c.title[:60]}")
            clean_text = preprocessing.normalize_text(c.full_text)

            t0 = time.time()
            try:
                if mode == "combined":
                    combined = extract_and_summarize(clean_text)
                    clauses = {k: combined[k] for k in ["termination_clause", "confidentiality_clause", "liability_clause"]}
                    summary = combined["summary"]
                else:  
                    clauses = extract_clauses(clean_text)
                    summary = summarize_contract(clean_text)
            except Exception as e:
                print(f"  !! Failed on this contract: {e} - marking as FAILED and continuing")
                clauses = {"termination_clause": "FAILED", "confidentiality_clause": "FAILED", "liability_clause": "FAILED"}
                summary = "FAILED"
            elapsed = time.time() - t0

            results.append({
                "contract_id": c.contract_id,
                "summary": summary.strip(),
                "termination_clause": clauses["termination_clause"],
                "confidentiality_clause": clauses["confidentiality_clause"],
                "liability_clause": clauses["liability_clause"],
                "gold_termination": c.gold_termination,
                "gold_liability": c.gold_liability,
                "seconds": round(elapsed, 1),
            })

    
    csv_path = os.path.join(OUTPUT_DIR, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "contract_id", "summary", "termination_clause",
            "confidentiality_clause", "liability_clause",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in writer.fieldnames})
    print(f"Wrote {csv_path}")

   
    json_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {json_path}")

    if build_index:
        records = []
        for r in results:
            for clause_type in ["termination_clause", "confidentiality_clause", "liability_clause"]:
                records.append({
                    "contract_id": r["contract_id"],
                    "clause_type": clause_type,
                    "text": r[clause_type],
                })
        idx = semantic_search.ClauseIndex()
        idx.build(records)
        if idx.index is not None:
            idx.save()
            print(f"Saved semantic search index ({len(idx.metadata)} clauses) to {semantic_search.INDEX_DIR}")


    report = evaluate.evaluate_batch(results)
    print("\n=== Extraction accuracy vs. CUAD gold labels (token F1) ===")
    print(json.dumps(report, indent=2))
    with open(os.path.join(OUTPUT_DIR, "eval_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Number of contracts to sample from CUAD")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N of the sample (quick test)")
    parser.add_argument("--no-index", action="store_true", help="Skip building the semantic search index")
    parser.add_argument("--mode", choices=["classic", "combined", "batch"], default="classic",
                         help="classic=2 calls/contract (100 total). combined=1 call/contract (50 total). "
                              "batch=1 call per N contracts (fewest calls, use --batch-size)")
    parser.add_argument("--batch-size", type=int, default=5,
                         help="Contracts per LLM call when --mode batch (default 5 -> 10 calls for 50 contracts)")
    args = parser.parse_args()

    run_pipeline(n_contracts=args.n, limit=args.limit, build_index=not args.no_index,
                 mode=args.mode, batch_size=args.batch_size)
