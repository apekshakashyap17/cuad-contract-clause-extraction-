"""
llm_client.py
-------------
OpenRouter client using openai/gpt-4.1-mini.
Set OPENROUTER_API_KEY as an environment variable (see .env.example).
"""

import json
import os
import re
import time

from openai import OpenAI

MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
PROVIDER = os.environ.get("LLM_PROVIDER", "openrouter")


def _call_openrouter(system: str, user: str, max_tokens: int) -> str:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=0,
    )
    return resp.choices[0].message.content


def _call_mock(system: str, user: str, max_tokens: int) -> str:
    """No-API-key mode for smoke-testing the pipeline for free before spending credits."""
    if "<contract id=" in user:
        ids = re.findall(r'<contract id="([^"]+)">(.*?)</contract>', user, re.DOTALL)
        out = []
        for cid, text in ids:
            entry = {"contract_id": cid}
            for key, kw in [
                ("termination_clause", r"terminat"),
                ("confidentiality_clause", r"confidential"),
                ("liability_clause", r"liabilit"),
            ]:
                match = re.search(rf"([^.]*{kw}[^.]*\.)", text, re.IGNORECASE)
                entry[key] = match.group(1).strip() if match else "NOT FOUND"
            entry["summary"] = f"[MOCK SUMMARY for {cid}]"
            out.append(entry)
        return json.dumps(out)
    elif "termination_clause" in user and "confidentiality_clause" in user:
        m = re.search(r'excerpt:\s*"""(.*)"""', user, re.DOTALL)
        text = m.group(1) if m else user
        out = {}
        for key, kw in [
            ("termination_clause", r"terminat"),
            ("confidentiality_clause", r"confidential"),
            ("liability_clause", r"liabilit"),
        ]:
            match = re.search(rf"([^.]*{kw}[^.]*\.)", text, re.IGNORECASE)
            out[key] = match.group(1).strip() if match else "NOT FOUND"
        return json.dumps(out)
    return "[MOCK SUMMARY] Placeholder summary for pipeline smoke-testing only."


_CALLERS = {
    "openrouter": _call_openrouter,
    "mock": _call_mock,
}


def chat(system: str, user: str, max_tokens: int = 800, retries: int = 4) -> str:
    caller = _CALLERS[PROVIDER]
    last_err = None
    for attempt in range(retries):
        try:
            return caller(system, user, max_tokens)
        except Exception as e:
            last_err = e
            wait = _extract_retry_after(e) or min(2 ** attempt * 5, 60)
            print(f"  [retry {attempt + 1}/{retries}] {type(e).__name__} - waiting {wait}s before retrying...")
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")


def _extract_retry_after(exc: Exception):
    text = str(exc)
    match = re.search(r'"retry_after_seconds":\s*([\d.]+)', text)
    if match:
        return float(match.group(1)) + 2
    return None


def chat_json(system: str, user: str, max_tokens: int = 800) -> dict:
    raw = chat(system, user, max_tokens=max_tokens)
    cleaned = raw.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError as e:
        print("\n========== RAW MODEL OUTPUT ==========\n")
        print(cleaned)
        print("\n======================================\n")

        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass

        raise e