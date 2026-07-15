import json
import os

from src.evaluate import evaluate_batch

OUTPUT_DIR = "output"

with open(os.path.join(OUTPUT_DIR, "results.json"), "r", encoding="utf-8") as f:
    results = json.load(f)

report = evaluate_batch(results)

print("\n=== Extraction accuracy vs. CUAD gold lab els (token F1) ===")
print(json.dumps(report, indent=2))

with open(os.path.join(OUTPUT_DIR, "eval_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\nSaved output/eval_report.json")