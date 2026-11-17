# Vietnamese Travel RAG

Retrieval-augmented question answering over Vietnamese Wikipedia articles about
provinces, destinations, heritage sites, food and festivals, with cited source
passages. Work in progress.

## Status

| Step | State |
|---|---|
| Data selection (500 articles) and chunking | done |
| Embeddings + FAISS retrieval | planned |
| Answer generation with citations (Gemini) | planned |
| Evaluation set (~150 questions, 50 human-reviewed) | planned |
| Retrieval / answer evaluation | planned |
| Streamlit demo | planned |

## Data

* Source: [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia),
  config `20231101.vi` (Vietnamese Wikipedia dump of 2023-11-01).
* License: text is CC BY-SA 4.0. Every chunk keeps the URL of its article so
  answers can attribute the source.
* The dump predates the 2025 province mergers, so it describes the 63-province map.
* Selection (`src/travel_rag/select_articles.py`) is rule-based because the dump
  has no categories: the 63 provinces, a curated list of well-known places,
  dishes and festivals, and prefix-matched titles ("Vịnh ", "Chùa ", "Bánh ",
  "Lễ hội ", ...) filtered by their lead section. Result: 63 provinces,
  160 destinations, 135 heritage sites, 85 dishes, 57 festivals.
  The rules still let some loosely related pages through; this is a known limitation.

## Chunking

`src/travel_rag/chunking.py` splits each article into section-aware,
sentence-aligned chunks with sentence-level overlap, measured in
whitespace-separated words (Vietnamese syllables). Reference sections are dropped.

| chunk size / overlap (words) | chunks | mean words |
|---|---|---|
| 100 / 20 | 18,469 | 75.3 |
| 200 / 40 | 10,393 | 138.4 |
| 400 / 80 | 6,968 | 204.2 |

## Reproduce

```bash
pip install -r requirements.txt
python scripts/download_wikipedia.py                 # ~730 MB parquet into data/raw/
PYTHONPATH=src python -m travel_rag.select_articles   # -> data/processed/articles.jsonl
PYTHONPATH=src python -m travel_rag.build_chunks --size 200 --overlap 40
python -m pytest
```
