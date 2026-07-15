"""
data_loader.py
--------------
Downloads the public CUAD (Contract Understanding Atticus Dataset) release,
loads the full contract texts, and samples a reproducible subset of contracts
for this pipeline.

CUAD ships as a single SQuAD-style JSON (CUADv1.json) rather than 510 loose
PDFs: each "paragraph" contains the FULL contract text under `context`, plus
41 clause-type Q&A pairs with gold answer spans under `qas`. We use the raw
contract text as our document source, and separately keep the gold spans for
"Termination for Convenience" and "Cap on Liability" / "Uncapped Liability"
so the evaluate.py module can score our LLM extraction against ground truth.
"""

import json
import os
import random
import urllib.request
import zipfile
from dataclasses import dataclass, field
from typing import Optional

CUAD_ZIP_URL = "https://raw.githubusercontent.com/TheAtticusProject/cuad/master/data.zip"
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
SUBSET_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "subset")

GOLD_CATEGORY_MAP = {
    "termination_clause": "Termination for Convenience",
    "liability_clause": ["Cap on Liability", "Uncapped Liability"],
}


@dataclass
class Contract:
    contract_id: str
    title: str
    full_text: str
    gold_termination: list = field(default_factory=list)
    gold_liability: list = field(default_factory=list)


def _download_and_extract():
    os.makedirs(RAW_DIR, exist_ok=True)
    zip_path = os.path.join(RAW_DIR, "data.zip")
    json_path = os.path.join(RAW_DIR, "CUADv1.json")
    if os.path.exists(json_path):
        return json_path
    print(f"Downloading CUAD dataset from {CUAD_ZIP_URL} ...")
    urllib.request.urlretrieve(CUAD_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW_DIR)
    return json_path


def _gold_spans_for_question(qas, question_substring):
    for qa in qas:
        if question_substring.lower() in qa["question"].lower():
            return [a["text"] for a in qa["answers"]]
    return []


def load_all_contracts() -> list:
    """Parse CUADv1.json into a list of Contract objects (full text + gold spans)."""
    json_path = _download_and_extract()
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    contracts = []
    for entry in raw["data"]:
        title = entry["title"]
        para = entry["paragraphs"][0]
        text = para["context"]
        qas = para["qas"]

        gold_term = _gold_spans_for_question(qas, GOLD_CATEGORY_MAP["termination_clause"])
        gold_liab = []
        for cat in GOLD_CATEGORY_MAP["liability_clause"]:
            gold_liab.extend(_gold_spans_for_question(qas, cat))

        contracts.append(
            Contract(
                contract_id=title,
                title=title,
                full_text=text,
                gold_termination=gold_term,
                gold_liability=gold_liab,
            )
        )
    return contracts


def sample_subset(n: int = 50, seed: int = 42) -> list:
    """Deterministically sample n contracts so the run is reproducible."""
    contracts = load_all_contracts()
    rng = random.Random(seed)
    return rng.sample(contracts, min(n, len(contracts)))


def save_subset_to_disk(contracts: list, out_dir: str = SUBSET_DIR):
    """Write each contract's raw text to disk (mirrors 'extracted from PDF' step)."""
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for c in contracts:
        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in c.contract_id)[:100]
        path = os.path.join(out_dir, f"{safe_name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(c.full_text)
        manifest.append({"contract_id": c.contract_id, "path": path})
    return manifest


if __name__ == "__main__":
    subset = sample_subset(50)
    manifest = save_subset_to_disk(subset)
    print(f"Saved {len(manifest)} contracts to {SUBSET_DIR}")
