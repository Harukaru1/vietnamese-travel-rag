# Vietnamese Travel RAG

Retrieval-augmented question answering over Vietnamese Wikipedia articles about
provinces, destinations, heritage sites, food and festivals, with cited source
passages. Work in progress.

## Status

| Step | State |
|---|---|
| Data selection (500 articles) and chunking | done |
| Embeddings + FAISS retrieval | done |
| Answer generation with citations (Gemini) | done |
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

## Retrieval

* Embedding model: [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3)
  (multilingual, 1024 dims). Each chunk is embedded as `title — section` plus
  its text; vectors are L2-normalised so inner product = cosine similarity.
* Index: FAISS `IndexFlatIP` (exact search). At ~10k vectors a brute-force
  scan takes milliseconds, so an approximate index (HNSW/IVF) is unnecessary.
* CPU embedding speed, measured on 128 sampled chunks with 4 threads
  (`scripts/benchmark_embedders.py`, `results/embedding_benchmark.json`):

| model | dim | chunks/s | est. time for 10,393 chunks |
|---|---|---|---|
| BAAI/bge-m3 | 1024 | 2.97 | ~58 min |
| intfloat/multilingual-e5-base | 768 | 10.34 | ~17 min |
| intfloat/multilingual-e5-small | 384 | 36.54 | ~5 min |

  bge-m3 is used for the main index. Because it is ~12x slower than
  multilingual-e5-small on CPU, the chunk-size comparison is run with
  e5-small, and the two models are compared at the default chunk size.
* Chunks average ~1.23 tokens per word with these tokenizers
  (median 204 tokens for 166 words), well under the 512-token cap.

```bash
PYTHONPATH=src python -m travel_rag.build_index --size 200 --overlap 40 --model BAAI/bge-m3
PYTHONPATH=src python -m travel_rag.search "Phở có nguồn gốc từ đâu?" --k 5
```

## Answer generation

`src/travel_rag/rag.py` sends the top-k chunks as numbered sources to Gemini
(`gemini-flash-lite-latest`, temperature 0) with a JSON response schema
`{found, answer, citations}`. Hallucination controls:

* answer only from the sources; outside knowledge is forbidden;
* explicit abstention: `found=false` becomes the fixed string
  `Không có trong tài liệu.` so abstention can be measured;
* citation numbers outside `1..k` are dropped and counted; kept ones map to the
  chunk id and Wikipedia URL;
* the sources are a November 2023 snapshot, so answers about "now" must state
  the time the source describes;
* source text is treated as data, not instructions.

The client (`src/travel_rag/llm.py`) retries 429/5xx with exponential backoff
and caches responses on disk, so evaluation reruns are repeatable and free.

Observed on a few hand-picked questions (not a measurement; see Evaluation):

| question | RAG | no RAG |
|---|---|---|
| Fansipan cao bao nhiêu mét? | 3147,3 m, cited | 3.147 m |
| Giá vé tham quan Vịnh Hạ Long năm 2024? | `Không có trong tài liệu.` | invents specific 2024 prices |
| Ai là huấn luyện viên đội tuyển bóng đá Việt Nam? | 2012 coach from a city article, flagged as dated; one citation does not support the claim | current coach |

The last row shows two failure modes the evaluation has to measure: stale
sources and citations that do not support the claim.

```bash
PYTHONPATH=src python -m travel_rag.ask "Phở có nguồn gốc từ đâu?" --closed-book
```

## Reproduce

```bash
pip install -r requirements.txt
python scripts/download_wikipedia.py                 # ~730 MB parquet into data/raw/
PYTHONPATH=src python -m travel_rag.select_articles   # -> data/processed/articles.jsonl
PYTHONPATH=src python -m travel_rag.build_chunks --size 200 --overlap 40
python -m pytest
```
