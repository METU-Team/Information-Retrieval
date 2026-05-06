import os
import pandas as pd
import pyterrier as pt
from src.data_loader import load_corpus, load_queries_and_qrels, load_trec_dl_queries_and_qrels
from src.bm25_retriever import build_index, load_index, get_bm25_retriever, retrieve
from src.dense_retriever import load_biencoder, encode_corpus, load_faiss_index, dense_retrieve
from src.hybrid_retriever import reciprocal_rank_fusion
from src.evaluate import evaluate_all, evaluate_all_ir_measures, evaluate_trec_dl
from src.config import INDEX_DIR, FAISS_INDEX_PATH

if not pt.started():
    pt.init()

# 1. Load data (queries/qrels first so corpus can guarantee relevant docs)
print("--- Phase 1: Loading Dataset ---")
queries_df, qrels_df = load_queries_and_qrels()

# Also load TREC-DL queries/qrels (graded relevance)
trec_dl_queries, trec_dl_qrels = load_trec_dl_queries_and_qrels()

# Merge qrels from both sets for qrels-aware corpus loading
# This ensures TREC-DL relevant passages are also in the corpus
combined_qrels = pd.concat([qrels_df, trec_dl_qrels], ignore_index=True).drop_duplicates(
    subset=["qid", "docno"], keep="first"
)

corpus_df = load_corpus(qrels_df=combined_qrels)
print(f"Corpus: {len(corpus_df)} passages | MS MARCO Queries: {len(queries_df)} | TREC-DL Queries: {len(trec_dl_queries)}")

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
print(f"BM25 done. Sample:\n{bm25_results.head(5)}")

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

# 4. Hybrid Retrieval
print("\n--- Phase 4: Hybrid Retrieval (RRF) ---")
hybrid_results = reciprocal_rank_fusion([bm25_results, dense_results])
print(f"Hybrid done. Sample:\n{hybrid_results.head(3)}")

# 5. Evaluate — MS MARCO dev/small (binary relevance)
results_dict = {
    "BM25_Baseline": bm25_results, 
    "Dense_BiEncoder": dense_results,
    "Hybrid_RRF": hybrid_results
}

print("\n--- Phase 5a: Evaluation (Custom) ---")
summary = evaluate_all(results_dict, qrels_df)
print("\n=== RESULTS (Custom) ===")
print(summary.to_string(index=False))

print("\n--- Phase 5b: Evaluation (ir_measures / trec_eval-compatible) ---")
summary_ir = evaluate_all_ir_measures(results_dict, qrels_df)
print("\n=== RESULTS (ir_measures — MS MARCO Binary) ===")
print(summary_ir.to_string(index=False))

# 6. Evaluate — TREC-DL 2019 (graded relevance: 0-3)
print("\n--- Phase 6: TREC-DL 2019 Evaluation (Graded Relevance) ---")
bm25_results_dl = retrieve(bm25_retriever, trec_dl_queries)
dense_results_dl = dense_retrieve(bi_model, faiss_idx, passage_ids, trec_dl_queries)
hybrid_results_dl = reciprocal_rank_fusion([bm25_results_dl, dense_results_dl])

trec_dl_dict = {
    "BM25_Baseline": bm25_results_dl, 
    "Dense_BiEncoder": dense_results_dl,
    "Hybrid_RRF": hybrid_results_dl
}
summary_dl = evaluate_trec_dl(trec_dl_dict, trec_dl_qrels)
print("\n=== RESULTS (TREC-DL 2019 — Graded) ===")
print(summary_dl.to_string(index=False))