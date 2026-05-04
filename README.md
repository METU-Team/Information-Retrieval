# MS MARCO Passage Retrieval - Progress Report

This repository contains the ongoing development of a multi-stage Information Retrieval (IR) pipeline using the MS MARCO Passage dataset. Currently, the project implements and evaluates isolated baseline models: Lexical Search (BM25) and Semantic Search (Dense Bi-Encoder).

## 🚀 Setup

```bash
pip install -r requirements.txt
```

## 🛠️ Step-by-step Execution

Currently, the project is structured to test models individually.

### Step 1: Build BM25 Index
```python
from src.data_loader import load_corpus
from src.bm25_retriever import build_index

corpus_df = load_corpus()
build_index(corpus_df, overwrite=True)
```

### Step 2: Build FAISS Dense Index
```python
from src.data_loader import load_corpus
from src.dense_retriever import load_biencoder, encode_corpus

corpus_df = load_corpus()
model = load_biencoder()
encode_corpus(model, corpus_df, save=True)
```

### Step 3: Run Experiments & Evaluation
```bash
# Run the main experiment script to evaluate MRR@10 and nDCG@10
python run_experiment.py
```

## 📊 Evaluation Metrics (Subset: 100k)

*Note: The current metrics reflect performance on a hardware-constrained 100k document subset, causing absolute MRR scores to be lower than full-corpus benchmarks. However, the relative performance accurately demonstrates the advantage of semantic search.*

| System | MRR@10 | nDCG@10 |
|---|---|---|
| BM25 Baseline | 0.001 | — |
| Dense (Bi-encoder) | 0.002 | — |

## 🤖 Models Used (Current Stage)
- **Bi-encoder**: `sentence-transformers/msmarco-distilbert-base-v3`
- **BM25**: PyTerrier with Porter stemming + English stopwords

---

## 🗺️ Roadmap (Upcoming Features for Final Submission)
The following features are planned for the next phases of the project:
- Implement Hybrid Retrieval (BM25 + Dense combination).
- Add Cross-Encoder Re-Ranking (`ms-marco-MiniLM-L-6-v2`).
- Develop an interactive Web UI using Streamlit.