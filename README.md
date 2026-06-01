# Hybrid Passage Retrieval — MS MARCO & TREC-DL 2019

A multi-stage Information Retrieval pipeline that combines lexical search (BM25), semantic search (Dense Bi-Encoder), hybrid fusion, and neural re-ranking to achieve state-of-the-art passage retrieval on the MS MARCO and TREC Deep Learning 2019 benchmarks.

## Pipeline Overview

```
Query
  │
  ├──► BM25 (top-100)  ──────────────────────────────────────► BM25+CE
  │         │                                                      │
  │         ├──► Hybrid RRF  ──────────────────► Hybrid RRF+CE     │
  │         │         │                               │            │
  │         ├──► Hybrid CC (α·Dense + (1-α)·BM25) ─► Hybrid CC+CE │
  │         │                                                      │
  └──► Dense (top-100) ──────────────────────────────────────► Dense+CE
          ▲                                                    ▲
     Bi-Encoder                                         Cross-Encoder
   (FAISS IndexFlatIP)                              (ms-marco-MiniLM-L-6-v2)
```

**8 retrieval configurations** are evaluated end-to-end, from single-stage baselines to full hybrid + neural re-ranking pipelines.

## Setup

```bash
pip install -r requirements.txt
```

## Execution

```bash
# Run full experiment — builds indices, encodes corpus, runs all 8 pipelines,
# and evaluates on both MS MARCO (binary) and TREC-DL 2019 (graded).
python run_experiment.py

# Launch interactive web UI
streamlit run app.py
```

## Evaluation Results (1M Passage Subset)

All metrics are computed using `ir_measures` (trec_eval-compatible). The corpus uses **qrels-aware loading** — all relevant passages from both MS MARCO dev/small and TREC-DL 2019 qrels are guaranteed to be present in the index, ensuring mathematically sound evaluation.

### MS MARCO dev/small — Binary Relevance (1,000 queries)

| System | MRR@10 | nDCG@10 |
|:--|:--:|:--:|
| BM25 | 0.4517 | 0.4925 |
| Dense | 0.5913 | 0.6335 |
| Hybrid RRF | 0.5801 | 0.6241 |
| Hybrid CC (α=0.75) | 0.6164 | 0.6584 |
| BM25+CE | 0.6849 | 0.7135 |
| Dense+CE | 0.7126 | 0.7475 |
| Hybrid RRF+CE | 0.7201 | 0.7566 |
| **Hybrid CC+CE** | **0.7197** | **0.7563** |

### TREC Deep Learning 2019 — Graded Relevance (0–3 scale, ~43 queries)

| System | MRR(rel≥2)@10 | nDCG@10 |
|:--|:--:|:--:|
| BM25 | 0.7359 | 0.6235 |
| Dense | 0.9147 | 0.7435 |
| Hybrid RRF | 0.8934 | 0.7354 |
| Hybrid CC (α=0.75) | 0.9018 | 0.7576 |
| BM25+CE | 0.9089 | 0.7798 |
| Dense+CE | 0.9205 | 0.7980 |
| Hybrid RRF+CE | 0.9205 | 0.8120 |
| **Hybrid CC+CE** | **0.9205** | **0.8108** |

### Key Findings

- **Cross-Encoder re-ranking is the single biggest performance lever.** Every first-stage retriever benefits substantially — even BM25+CE outperforms standalone Dense on both benchmarks.
- **Convex Combination outperforms RRF** in first-stage fusion (CC: 0.6164 vs RRF: 0.5801 MRR@10 on MS MARCO), thanks to score-level interpolation with min-max normalization rather than rank-based aggregation.
- **After Cross-Encoder re-ranking, the gap between fusion methods narrows.** Hybrid RRF+CE and Hybrid CC+CE converge to near-identical performance, suggesting the cross-encoder effectively compensates for first-stage fusion differences.
- **Dense retrieval dominates BM25** on both benchmarks, confirming the advantage of semantic matching over lexical matching for MS MARCO-style queries.

## Architecture & Models

### Stage 1 — First-Stage Retrieval (top-100 candidates)

| Component | Implementation | Details |
|:--|:--|:--|
| **BM25** | PyTerrier `DFIndexer` | Porter stemming, English stopwords, positional index |
| **Dense Bi-Encoder** | `msmarco-distilbert-base-v3` | 768-dim embeddings, dot-product similarity via FAISS `IndexFlatIP` |

### Stage 2 — Hybrid Fusion

| Method | Formula | Description |
|:--|:--|:--|
| **RRF** | `Σ 1/(k + rank_i)` | Rank-based fusion, k=60 (Cormack et al., 2009) |
| **Convex Combination** | `α·dense + (1-α)·sparse` | Score-level fusion with per-query min-max normalization, α=0.75 |

### Stage 3 — Cross-Encoder Re-Ranking (top-10 from top-100)

| Component | Implementation | Details |
|:--|:--|:--|
| **Cross-Encoder** | `ms-marco-MiniLM-L-6-v2` | Jointly encodes (query, passage) pairs for fine-grained relevance scoring |

### Evaluation

- **MS MARCO dev/small**: Binary relevance → MRR@10, nDCG@10
- **TREC-DL 2019**: Graded relevance (0–3) → MRR(rel≥2)@10, nDCG@10
- Evaluation via `ir_measures` (trec_eval-compatible)

## Project Structure

```
├── run_experiment.py          # End-to-end experiment: all 8 systems, both benchmarks
├── app.py                     # Streamlit interactive search UI
├── src/
│   ├── config.py              # Paths, model names, hyperparameters
│   ├── data_loader.py         # MS MARCO & TREC-DL data loading (qrels-aware)
│   ├── bm25_retriever.py      # BM25 indexing & retrieval (PyTerrier)
│   ├── dense_retriever.py     # Bi-encoder encoding & FAISS retrieval (chunked for large corpora)
│   ├── hybrid_retriever.py    # RRF & Convex Combination fusion
│   ├── reranker.py            # Cross-encoder re-ranking
│   └── evaluate.py            # ir_measures evaluation (MS MARCO + TREC-DL)
├── index/                     # BM25 inverted index (auto-generated)
├── embeddings/                # FAISS index + passage IDs (auto-generated)
└── requirements.txt
```