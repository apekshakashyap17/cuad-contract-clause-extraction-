"""
clause_extractor.py
--------------------
Part A - Clause Extraction.

Prompts the LLM to pull out (verbatim, where possible) the termination,
confidentiality, and liability clauses from a contract. Long contracts are
split into overlapping chunks (preprocessing.chunk_text); each chunk is
checked independently and the first clause found per category wins, since
these clauses are usually self-contained sections rather than split across
the document.
"""

from . import llm_client
from .preprocessing import chunk_text

SYSTEM_PROMPT = (
    "You are a meticulous contracts lawyer's assistant. You extract specific "
    "clause types from commercial contracts verbatim (or as close to verbatim "
    "as the excerpt allows). You never invent text that is not in the contract. "
    "If a clause type genuinely does not appear in the given text, say so."
)

FEW_SHOT_EXAMPLE = """
Example input excerpt:
"9. TERM AND TERMINATION. This Agreement shall remain in effect for two (2) years \
from the Effective Date. Either party may terminate this Agreement for convenience \
upon sixty (60) days written notice. 10. CONFIDENTIALITY. Each party agrees to hold \
the other's Confidential Information in strict confidence and not disclose it to any \
third party without prior written consent. 11. LIMITATION OF LIABILITY. Neither \
party's aggregate liability shall exceed the fees paid in the preceding twelve months."

Example output:
{
  "termination_clause": "Either party may terminate this Agreement for convenience upon sixty (60) days written notice.",
  "confidentiality_clause": "Each party agrees to hold the other's Confidential Information in strict confidence and not disclose it to any third party without prior written consent.",
  "liability_clause": "Neither party's aggregate liability shall exceed the fees paid in the preceding twelve months."
}
""".strip()

CLAUSE_KEYS = ["termination_clause", "confidentiality_clause", "liability_clause"]


def _build_user_prompt(chunk: str) -> str:
    return f"""{FEW_SHOT_EXAMPLE}

Now extract from this contract excerpt. Return ONLY a JSON object with exactly
these three keys: termination_clause, confidentiality_clause, liability_clause.
For each key, return the relevant clause text verbatim from the excerpt, or the
exact string "NOT FOUND" if that clause type does not appear in this excerpt.

Contract excerpt:
\"\"\"
{chunk}
\"\"\"
"""


def extract_clauses(contract_text: str) -> dict:
    """Returns {"termination_clause": str, "confidentiality_clause": str, "liability_clause": str}."""
    result = {k: "NOT FOUND" for k in CLAUSE_KEYS}

    for chunk in chunk_text(contract_text):
        if all(result[k] != "NOT FOUND" for k in CLAUSE_KEYS):
            break  
        parsed = llm_client.chat_json(SYSTEM_PROMPT, _build_user_prompt(chunk), max_tokens=700)
        for k in CLAUSE_KEYS:
            if result[k] == "NOT FOUND" and parsed.get(k, "NOT FOUND") != "NOT FOUND":
                result[k] = parsed[k]

    return result
