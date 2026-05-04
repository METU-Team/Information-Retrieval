import os
import pyterrier as pt
from src.data_loader import load_corpus, load_queries_and_qrels
from src.bm25_retriever import build_index, load_index, get_bm25_retriever, retrieve
from src.dense_retriever import load_biencoder, encode_corpus, load_faiss_index, dense_retrieve
from src.evaluate import evaluate_all
from src.config import INDEX_DIR, FAISS_INDEX_PATH

if not pt.started():
    pt.init()

# 1. Load data
print("--- Phase 1: Loading Dataset ---")
corpus_df = load_corpus()
queries_df, qrels_df = load_queries_and_qrels()
print(f"Corpus: {len(corpus_df)} passages | Queries: {len(queries_df)} | Qrels: {len(qrels_df)}")

# 2. BM25
print("\n--- Phase 2: BM25 Baseline ---")
index_path = os.path.join(INDEX_DIR, "data.properties")
if not os.path.exists(index_path):
    print("Building index (first run)...")
    index_ref = build_index(corpus_df, overwrite=True)
else:
    print("Loading existing index...")
    index_ref = load_index()

bm25_retriever = get_bm25_retriever(index_ref)
bm25_results = retrieve(bm25_retriever, queries_df)
print(f"BM25 done. Sample:\n{bm25_results.head(30)}")

# 3. Dense
print("\n--- Phase 3: Dense Retrieval ---")
bi_model = load_biencoder()

print(f"Running on: {str(bi_model.device).upper()}")

if not os.path.exists(FAISS_INDEX_PATH):
    print("Encoding corpus (this takes ~30 mins on CPU)...")
    encode_corpus(bi_model, corpus_df, save=True)
else:
    print("FAISS index found, skipping encoding.")

faiss_idx, passage_ids = load_faiss_index()
dense_results = dense_retrieve(bi_model, faiss_idx, passage_ids, queries_df)
print(f"Dense done. Sample:\n{dense_results.head(3)}")

# 4. Evaluate
print("\n--- Phase 4: Evaluation ---")
summary = evaluate_all(
    {"BM25_Baseline": bm25_results, "Dense_BiEncoder": dense_results},
    qrels_df
)
print("\n=== RESULTS ===")
print(summary.to_string(index=False))