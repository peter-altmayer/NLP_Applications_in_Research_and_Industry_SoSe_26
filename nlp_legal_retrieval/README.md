# NLP Legal Document Retrieval

Retrieval system for legal documents using BM25 and dense retrieval methods.

## Project Structure

```
nlp_legal_retrieval/
├── data/
│   ├── raw/          # Original datasets (CUAD, LEDGAR, etc.)
│   └── processed/    # Tokenized / chunked documents
├── docs/             # Framework notes, plans, specs
├── experiments/      # Standalone experiment scripts
├── notebooks/        # Colab or local Jupyter notebooks
├── results/          # CSV metrics and plots
├── src/              # Core library code
└── tests/            # Unit tests
```

## Datasets (TBD)

Candidates:
- **CUAD** – Contract Understanding Atticus Dataset (510 contracts, 41 legal clause types)
- **LEDGAR** – SEC filing provisions (60k+ labeled segments)
- **EUR-Lex** – EU legislative documents

## Retrieval Methods

- BM25 (lexical baseline)
- Dense retrieval (sentence-transformers / legal-BERT variants)

## Evaluation Metrics

- MRR@k, NDCG@k, Recall@k

## Setup

```bash
pip install -r requirements.txt
```
