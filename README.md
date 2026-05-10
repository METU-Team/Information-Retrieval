# Hybrid Passage Retrieval - MS MARCO & TREC-DL

This repository contains the development of a multi-stage Information Retrieval (IR) pipeline using the MS MARCO Passage dataset. The project currently implements Lexical Search (BM25), Semantic Search (Dense Bi-Encoder), and Hybrid Retrieval (Reciprocal Rank Fusion) pipelines.

## 🚀 Setup

```bash
pip install -r requirements.txt
```
*(Note: `ir_measures` for evaluation ships natively with PyTerrier).*

## 🛠️ Execution

The project provides an end-to-end script that runs both MS MARCO binary relevance and TREC-DL 2019 graded relevance benchmarks automatically.

```bash
# Builds indices (if missing), runs BM25, Dense, and Hybrid pipelines, 
# and evaluates them using trec_eval compatible metrics.
python run_experiment.py
```

## 📊 Evaluation Metrics (200k Subset)

*Note: Metrics are measured on a 200k-passage subset using **qrels-aware loading**, ensuring all relevant documents are present in the index. This guarantees mathematically sound evaluation metrics while speeding up iterative development.*

### MS MARCO dev/small (Binary Relevance, 1,000 queries)
| System | MRR@10 | nDCG@10 |
|---|---|---|
| BM25 Baseline | 0.6351 | 0.6737 |
| Dense Bi-Encoder | 0.7739 | 0.8033 |
| Hybrid (RRF) | 0.7515 | 0.7902 |

### TREC Deep Learning 2019 (Graded Relevance 0-3, ~43 queries)
| System | MRR(rel≥2)@10 |nDCG@10 | 
|---|---|---|
| BM25 Baseline | 0.7618 |0.6838 | 
| Dense Bi-Encoder | 0.9181 |0.7881 | 
| Hybrid (RRF) |0.8973 | 0.7815 | 

## 🤖 Architecture & Models
- **BM25**: PyTerrier (`DFIndexer`) with Porter stemming + English stopwords
- **Dense Bi-Encoder**: `sentence-transformers/msmarco-distilbert-base-v3` + FAISS `IndexFlatIP`
- **Hybrid Fusion**: Reciprocal Rank Fusion (RRF)
- **Evaluation**: Custom Python implementations + `ir_measures` (`trec_eval` compatible)

---

## 🗺️ Roadmap (Upcoming Features for Final Submission)
The following features are planned for the final phase of the project:
- Add Cross-Encoder Re-Ranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) on top of the Hybrid candidate pool.
- Develop an interactive Web UI using Streamlit.
- Scaling experiments with larger corpus sizes (up to 1M).