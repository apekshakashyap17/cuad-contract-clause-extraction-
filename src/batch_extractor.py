"""
batch_extractor.py
--------------------
Processes multiple contracts in a SINGLE LLM call to minimize total API
calls (useful for free-tier daily request caps). Each contract is tagged
with its contract_id in the prompt so the model can't lose track of which
clause belongs to which contract, and the response is a JSON array keyed
by contract_id.

Trade-off vs. one-call-per-contract (combined_extractor.py): fewer calls,
but higher risk of the model mixing up contracts or truncating output when
batch_size is too large. batch_size=5 is a reasonable default; keep it lower
for longer contracts, higher only if you've verified quality holds up via
evaluate.py.
"""

from . import llm_client
from .preprocessing import chunk_text

SYSTEM_PROMPT = (
    "You are a meticulous contracts lawyer's assistant processing MULTIPLE "
    "contracts in one pass. Keep every contract's information strictly "
    "separate - never blend clauses or facts from one contract into "
    "another's result. You never invent text that is not in the contract."
)

USER_TEMPLATE = """You will be given {n} separate contracts, each wrapped in
<contract id="CONTRACT_ID">...</contract> tags. For EACH contract independently, do:

1. Extract these three clause types verbatim (or "NOT FOUND" if absent):
   - termination_clause
   - confidentiality_clause
   - liability_clause
2. Write a 100-150 word plain-English summary (purpose, key obligations of
   each party, notable risks/penalties).

Return ONLY a JSON array, one object per contract, in this exact shape:
[
  {{
    "contract_id": "...",
    "termination_clause": "...",
    "confidentiality_clause": "...",
    "liability_clause": "...",
    "summary": "..."
  }}
]

Do not merge or cross-reference information between contracts - treat each
<contract> block as fully independent.

{contracts_block}
"""


def _build_contracts_block(contracts: list) -> str:
    parts = []
    for c in contracts:
        first_chunk = chunk_text(c.full_text, max_chars=10000)[0]
        parts.append(f'<contract id="{c.contract_id}">\n{first_chunk}\n</contract>')
    return "\n\n".join(parts)


def extract_batch(contracts: list) -> list:
    """
    contracts: list of Contract objects (see data_loader.py).
    Returns a list of dicts: {contract_id, termination_clause,
    confidentiality_clause, liability_clause, summary} - one per input contract.
    """
    prompt = USER_TEMPLATE.format(
        n=len(contracts),
        contracts_block=_build_contracts_block(contracts),
    )
    raw = llm_client.chat(SYSTEM_PROMPT, prompt, max_tokens=400 * len(contracts))

    import json
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("["), cleaned.rfind("]")
        parsed = json.loads(cleaned[start : end + 1])


    if isinstance(parsed, dict):
        if "results" in parsed and isinstance(parsed["results"], list):
            parsed = parsed["results"]
        else:
            parsed = [parsed]

    by_id = {r.get("contract_id"): r for r in parsed if isinstance(r, dict)}
    results = []
    for c in contracts:
        r = by_id.get(c.contract_id, {})
        results.append({
            "contract_id": c.contract_id,
            "termination_clause": r.get("termination_clause", "NOT FOUND"),
            "confidentiality_clause": r.get("confidentiality_clause", "NOT FOUND"),
            "liability_clause": r.get("liability_clause", "NOT FOUND"),
            "summary": r.get("summary", ""),
        })
    return results


def extract_all(contracts: list, batch_size: int = 5) -> list:
    """Splits contracts into batches and processes each with one LLM call."""
    all_results = []
    for i in range(0, len(contracts), batch_size):
        batch = contracts[i : i + batch_size]
        print(f"  Batch {i // batch_size + 1}: contracts {i+1}-{i+len(batch)} ({len(batch)} in 1 call)")
        all_results.extend(extract_batch(batch))
    return all_results
