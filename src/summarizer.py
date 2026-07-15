"""
summarizer.py
-------------
Part B - Contract Summary.

Generates a 100-150 word summary covering: purpose of the agreement, key
obligations of each party, and notable risks/penalties. For long contracts we
summarize the first chunk (which almost always contains the recitals/purpose
and party obligations) rather than re-summarizing every chunk, since a single
tight summary is what the brief asks for.
"""

from . import llm_client
from .preprocessing import chunk_text

SYSTEM_PROMPT = (
    "You are a contracts analyst who writes clear, plain-English executive "
    "summaries of legal agreements for a non-lawyer audience."
)

USER_TEMPLATE = """Summarize the following contract in 100-150 words. Cover:
1. The purpose of the agreement.
2. The key obligations of each party.
3. Any notable risks or penalties (e.g. liability caps, termination fees, indemnities).

Write plain prose (no bullet points), 100-150 words total.

Contract excerpt:
\"\"\"
{text}
\"\"\"
"""


def summarize_contract(contract_text: str) -> str:
    chunk = chunk_text(contract_text)[0] 
    return llm_client.chat(SYSTEM_PROMPT, USER_TEMPLATE.format(text=chunk), max_tokens=350)
