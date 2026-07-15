# CUAD Contract Clause Extraction and Summarization

I built this project as part of an AI contract analysis assignment using the CUAD (Contract Understanding Atticus Dataset).

The goal was to process a sample of 50 legal contracts, extract three important clauses (termination, confidentiality and liability), generate a short summary for each contract using an LLM, and compare the extracted clauses with CUAD's annotated data wherever possible.

I also added two extra features:
- semantic search over the extracted clauses using FAISS
- evaluation against CUAD's gold annotations using token-level F1

---

### Dataset

The project uses the CUAD dataset, which contains 510 real-world legal contracts along with manually annotated clause spans.

Instead of storing each contract as a separate document, CUAD stores everything inside a single JSON file. Each contract contains the full contract text along with annotations for 41 different legal clause categories.

For this assignment I randomly sampled 50 contracts using a fixed seed so the same contracts are selected every time the project is run.

Since CUAD only has annotations for some clause types, only termination and liability extraction can be evaluated automatically. Confidentiality clauses are still extracted, but there isn't any ground truth available to score them.

---

### How the pipeline works

The pipeline follows these steps:

1. Load and sample contracts from CUAD
2. Clean and normalize the contract text
3. Split very large contracts into overlapping chunks when necessary
4. Extract termination, confidentiality and liability clauses
5. Generate a 100–150 word summary
6. Save everything to CSV and JSON
7. Compare extracted clauses against CUAD annotations
8. Build a semantic search index over the extracted clauses

---

### Project structure

```text
cuad-clause-extraction/

data/
output/
src/
main.py
requirements.txt
README.md
```

Most of the logic lives inside the `src` folder.

- `data_loader.py` loads the dataset
- `preprocessing.py` cleans and chunks contract text
- `clause_extractor.py` extracts the required clauses
- `summarizer.py` generates summaries
- `combined_extractor.py` combines extraction and summarization into one API call
- `batch_extractor.py` processes multiple contracts together to reduce API usage
- `evaluate.py` compares extracted clauses with CUAD annotations
- `semantic_search.py` builds the FAISS index

---

### Installation

Clone the repository.

```bash
git clone <repository-url>
cd cuad-contract-clause-extraction
```

Create a virtual environment.

```bash
python -m venv venv
```

Activate it.

Windows

```bash
venv\Scripts\activate
```

Install the required packages.

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root and add your OpenRouter API key.

```text
OPENROUTER_API_KEY=your_api_key
```

---

### Running the project

Run the full pipeline.

```bash
python main.py
```

Only process a few contracts.

```bash
python main.py --limit 5
```

Use one API call per contract.

```bash
python main.py --mode combined
```

Use batch mode to reduce API usage even further.

```bash
python main.py --mode batch --batch-size 5
```

Skip semantic search.

```bash
python main.py --no-index
```

---

### Different execution modes

The project supports three modes because API usage can become expensive.

| Mode | API calls for 50 contracts |
|------|----------------------------:|
| classic | 100 |
| combined | 50 |
| batch (5 contracts per request) | 10 |

I mainly used the combined mode since it stayed within OpenRouter's free daily limit.

---

### Output

After the pipeline finishes, the `output` folder contains:

- `results.csv` – extracted clauses and summaries
- `results.json` – same results with additional metadata
- `eval_report.json` – evaluation scores
- `clause_index/` – semantic search index

---

### Evaluation

Termination and liability clauses are evaluated using token-level F1 against CUAD's annotated spans.

On my run, the results were:

| Clause | Mean F1 |
|---------|---------:|
| Termination | 0.29 |
| Liability | 0.49 |

Confidentiality clauses are not evaluated because CUAD doesn't provide annotations for them.

---

### Some design decisions

A few things I added while building the project:

- Long contracts are split into overlapping chunks so clauses near chunk boundaries aren't missed.
- I used few-shot prompting to encourage the model to return structured JSON instead of free-form text.
- The extraction and summarization pipeline can run in separate, combined or batch modes depending on API limits.
- The LLM client is kept separate from the extraction code so changing models doesn't require changing the rest of the pipeline.

---

### Limitations

There are still a few limitations.

- CUAD doesn't provide labels for confidentiality clauses.
- Occasionally the LLM returns malformed JSON, so those contracts are marked as `FAILED` instead of stopping the whole pipeline.
- The sentence embedding model is downloaded from Hugging Face the first time semantic search is used.
- The pipeline currently processes contracts sequentially.

---

### Tools used

- Python
- OpenRouter
- GPT OSS 120B
- Sentence Transformers
- FAISS
- NumPy

---

Built by Apeksha Kashyap as part of an AI contract analysis assignment.