"""
combined_extractor.py
----------------------
Same output as running clause_extractor + summarizer separately, but in a
SINGLE LLM call per contract instead of two. This halves your total API
calls (50 contracts -> 50 calls instead of 100), which matters if you're on
OpenRouter's free tier (capped at 50 requests/day on :free models).

Use this instead of clause_extractor.py + summarizer.py when you want to
stay under a free-tier daily cap. Trade-off: slightly less focused prompting
than doing each task separately, but well within acceptable quality for this
assignment.
"""

from . import llm_client
from .preprocessing import chunk_text

SYSTEM_PROMPT = (
    "You are a meticulous contracts lawyer's assistant. You extract specific "
    "clause types from commercial contracts verbatim (or as close to verbatim "
    "as the excerpt allows) and write plain-English summaries for a "
    "non-lawyer audience. You never invent text that is not in the contract."
)

USER_TEMPLATE = """You will be given a contract excerpt. Do TWO things and return ONLY a JSON object:

1. Extract these three clause types verbatim from the excerpt (or "NOT FOUND" if a clause type doesn't appear):
   - termination_clause
   - confidentiality_clause
   - liability_clause

2. Write a 100-150 word plain-English summary covering: the purpose of the
   agreement, key obligations of each party, and notable risks or penalties.
   Put this under the key "summary".

Return ONLY this JSON shape, no other text:
{{
  "termination_clause": "...",
  "confidentiality_clause": "...",
  "liability_clause": "...",
  "summary": "..."
}}

Contract excerpt:
\"\"\"
{text}
\"\"\"
"""

CLAUSE_KEYS = ["termination_clause", "confidentiality_clause", "liability_clause"]


def extract_and_summarize(contract_text: str) -> dict:
    """
    One LLM call per contract. Uses only the first chunk for a long contract
    (same rationale as summarizer.py: recitals/parties/obligations live up
    front, and clause sections like termination/confidentiality/liability
    are usually complete within a single reasonably-sized chunk too).
    If you need multi-chunk clause search for very long contracts, use
    clause_extractor.py + summarizer.py separately instead (costs 2x calls).
    """
    chunk = chunk_text(contract_text, max_chars=16000)[0]

    parsed = llm_client.chat_json(
        SYSTEM_PROMPT,
        USER_TEMPLATE.format(text=chunk),
        max_tokens=1500
    )

    result = {k: parsed.get(k, "NOT FOUND") for k in CLAUSE_KEYS}
    result["summary"] = parsed.get("summary", "")
    return result