import json
import os

from src.semantic_search import ClauseIndex

OUTPUT_DIR = "output"

with open(os.path.join(OUTPUT_DIR, "results.json"), "r", encoding="utf-8") as f:
    results = json.load(f)

records = []

for r in results:
    for clause_type in [
        "termination_clause",
        "confidentiality_clause",
        "liability_clause",
    ]:
        records.append(
            {
                "contract_id": r["contract_id"],
                "clause_type": clause_type,
                "text": r[clause_type],
            }
        )

idx = ClauseIndex()
idx.build(records)

if idx.index is not None:
    idx.save()
    print(f"Saved semantic search index ({len(idx.metadata)} clauses)")
else:
    print("No clauses available to index.")