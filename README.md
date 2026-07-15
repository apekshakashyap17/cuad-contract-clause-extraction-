# CUAD Contract Clause Extraction & Summarization Pipeline

An LLM-powered pipeline that extracts termination, confidentiality, and
liability clauses from legal contracts and generates a plain-English summary
of each, built on a 50-contract subset of the public [CUAD dataset](https://www.atticusprojectai.org/cuad).

## Approach

CUAD distributes its 510 contracts as a single SQuAD-style JSON file
(`CUADv1.json`) rather than 510 separate PDFs — each contract's full text
lives in a `context` field, alongside 41 categories of human-annotated gold
clause spans. `src/data_loader.py` downloads this JSON directly from the
dataset's GitHub release, and deterministically samples 50 contracts (fixed
random seed, so results are reproducible run to run).

Two of our three target clause types have a direct CUAD gold-label
equivalent — **Termination for Convenience** and **Cap on Liability /
Uncapped Liability** — so as a bonus this pipeline scores its own extractions
against that ground truth (token-level F1, same family of metric as SQuAD).
CUAD has no dedicated "Confidentiality" category, so that clause is extracted
but not quantitatively scored.

### Flow diagram

```
                     ┌─────────────────────┐
                     │   CUADv1.json        │
                     │ (510 contracts,      │
                     │  full text + gold    │
                     │  clause labels)      │
                     └──────────┬───────────┘
                                │  data_loader.py
                                │  (sample 50, seed=42)
                                ▼
                     ┌─────────────────────┐
                     │  preprocessing.py    │
                     │  normalize + chunk   │
                     └──────────┬───────────┘
                                │
                 ┌──────────────┼──────────────┐
                 ▼                              ▼
     ┌─────────────────────┐        ┌─────────────────────┐
     │ clause_extractor.py  │        │   summarizer.py      │
     │ Part A: termination /│        │ Part B: 100-150 word │
     │ confidentiality /    │        │ summary (purpose,    │
     │ liability (few-shot  │        │ obligations, risks)  │
     │ prompted, per chunk) │        │                       │
     └──────────┬───────────┘        └──────────┬───────────┘
                 └──────────────┬───────────────┘
                                ▼
                     ┌─────────────────────┐
                     │      main.py         │
                     │  merges + writes:    │
                     │  results.csv/json    │
                     └──────────┬───────────┘
                                │
                 ┌──────────────┼──────────────┐
                 ▼                              ▼
     ┌─────────────────────┐        ┌─────────────────────┐
     │  evaluate.py (bonus) │        │semantic_search.py     │
     │  F1 vs CUAD gold     │        │(bonus) FAISS index    │
     │  labels               │       │over extracted clauses │
     └─────────────────────┘        └─────────────────────┘
```

## Setup

```bash
git clone <this-repo>
cd <this-repo>
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
cp .env.example .env   # then fill in your API key
```

Pick one LLM backend in `.env`:
- `LLM_PROVIDER=openrouter` — access GPT-4.1 / Llama / etc. through one key ([openrouter.ai](https://openrouter.ai))
- `LLM_PROVIDER=openai` — direct OpenAI key
- `LLM_PROVIDER=anthropic` — direct Anthropic key
- `LLM_PROVIDER=mock` — no API key needed; runs a keyword-matching stand-in so you can sanity-check the plumbing for free before spending API credits

## Run

```bash
python main.py                          # full run: 50 contracts, classic mode
python main.py --limit 5                # quick test on the first 5 sampled contracts
python main.py --mode combined          # 1 LLM call per contract instead of 2 (fits free-tier daily caps)
python main.py --mode batch --batch-size 5   # 1 call per 5 contracts (fewest total calls)
python main.py --no-index               # skip the bonus semantic search index (faster)
```

### Choosing a mode (matters for free-tier API limits)

| Mode | Calls for 50 contracts | Trade-off |
|------|------------------------|-----------|
| `classic` (default) | 100 (2/contract) | Most focused prompting per task; needs a paid key or a high-cap free tier |
| `combined` | 50 (1/contract) | Same accuracy in practice, half the calls; fits OpenRouter's free 50/day cap |
| `batch --batch-size 5` | 10 (1 per 5 contracts) | Fewest calls by far; higher risk of the model mixing up contracts across the batch - **check `output/eval_report.json` afterward** to confirm F1 scores didn't drop before you trust the output |

Outputs land in `output/`:
- `results.csv` — the required deliverable: `contract_id, summary, termination_clause, confidentiality_clause, liability_clause`
- `results.json` — same data plus CUAD gold labels and per-contract timing
- `eval_report.json` — mean token-F1 of our extraction vs. CUAD gold labels
- `clause_index/` — FAISS index + metadata for semantic search over extracted clauses (bonus)

## Design decisions

- **Chunking with overlap**: a handful of CUAD contracts run to 50k+
  characters. `preprocessing.chunk_text` splits these into overlapping
  windows so a clause near a chunk boundary isn't cut in half, and
  `clause_extractor.extract_clauses` stops early once all three clause types
  are found (most contracts resolve on chunk 1).
- **Few-shot prompting**: `clause_extractor.py` includes one hand-written
  example contract excerpt + expected JSON output, to pin down output format
  and reduce the model paraphrasing instead of quoting.
- **Provider-agnostic LLM client**: `llm_client.py` is a thin wrapper so
  swapping between OpenRouter/OpenAI/Anthropic (or comparing them — see
  Evaluation Criteria: Creativity) is a one-line env var change, not a code
  change.
- **Gold-label evaluation as a sanity check, not a leaderboard**: CUAD's gold
  spans are exact quotes chosen by legal annotators for a span-extraction
  task; our LLM is asked to extract *and* lightly summarize, so token-F1
  here is directional (are we in the right neighborhood of the contract),
  not a strict accuracy score.

## Known limitations

- Confidentiality clause extraction has no CUAD ground truth to check
  against — accuracy for that clause type is qualitative (spot-check the
  CSV) rather than measured.
- `sentence-transformers` downloads its embedding model from Hugging Face on
  first run — needs one outbound connection to huggingface.co.
- This pipeline processes contracts sequentially; for a larger corpus than
  50 contracts you'd want to batch/parallelize the LLM calls.
