import os
import pandas as pd
import pyterrier as pt
from src.data_loader import load_corpus, load_queries_and_qrels, load_trec_dl_queries_and_qrels
from src.bm25_retriever import build_index, load_index, get_bm25_retriever, retrieve
from src.dense_retriever import load_biencoder, encode_corpus, load_faiss_index, dense_retrieve
from src.hybrid_retriever import reciprocal_rank_fusion, convex_combination
from src.reranker import load_crossencoder, rerank
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

# 4. Hybrid Retrieval — RRF
print("\n--- Phase 4a: Hybrid Retrieval (RRF) ---")
hybrid_rrf_results = reciprocal_rank_fusion([bm25_results, dense_results])
print(f"Hybrid RRF done. Sample:\n{hybrid_rrf_results.head(3)}")

# 4b. Hybrid Retrieval — Convex Combination
print("\n--- Phase 4b: Hybrid Retrieval (Convex Combination) ---")
hybrid_cc_results = convex_combination(bm25_results, dense_results)
print(f"Hybrid CC done. Sample:\n{hybrid_cc_results.head(3)}")

# 5. Cross-Encoder Re-Ranking (on multiple candidate pools)
print("\n--- Phase 5: Cross-Encoder Re-Ranking ---")
cross_model = load_crossencoder()

# 5a. BM25 + CE
print("  Re-ranking BM25 candidates...")
bm25_ce_results = rerank(cross_model, bm25_results, queries_df, corpus_df)

# 5b. Dense + CE
print("  Re-ranking Dense candidates...")
dense_ce_results = rerank(cross_model, dense_results, queries_df, corpus_df)

# 5c. Hybrid RRF + CE
print("  Re-ranking Hybrid RRF candidates...")
hybrid_rrf_ce_results = rerank(cross_model, hybrid_rrf_results, queries_df, corpus_df)

# 5d. Hybrid CC + CE
print("  Re-ranking Hybrid CC candidates...")
hybrid_cc_ce_results = rerank(cross_model, hybrid_cc_results, queries_df, corpus_df)

print("Re-ranking done.")

# 6. Evaluate — MS MARCO dev/small (binary relevance)
results_dict = {
    "BM25":           bm25_results,
    "Dense":          dense_results,
    "Hybrid_RRF":     hybrid_rrf_results,
    "Hybrid_CC":      hybrid_cc_results,
    "BM25+CE":        bm25_ce_results,
    "Dense+CE":       dense_ce_results,
    "Hybrid_RRF+CE":  hybrid_rrf_ce_results,
    "Hybrid_CC+CE":   hybrid_cc_ce_results,
}

"""print("\n--- Phase 6a: Evaluation (Custom) ---")
summary = evaluate_all(results_dict, qrels_df)
print("\n=== RESULTS (Custom) ===")
print(summary.to_string(index=False))
"""
print("\n--- Phase 6: Evaluation (ir_measures / trec_eval-compatible) ---")
summary_ir = evaluate_all_ir_measures(results_dict, qrels_df)
print("\n=== RESULTS (ir_measures — MS MARCO Binary) ===")
print(summary_ir.to_string(index=False))

# 7. Evaluate — TREC-DL 2019 (graded relevance: 0-3)
print("\n--- Phase 7: TREC-DL 2019 Evaluation (Graded Relevance) ---")
bm25_results_dl = retrieve(bm25_retriever, trec_dl_queries)
dense_results_dl = dense_retrieve(bi_model, faiss_idx, passage_ids, trec_dl_queries)
hybrid_rrf_results_dl = reciprocal_rank_fusion([bm25_results_dl, dense_results_dl])
hybrid_cc_results_dl = convex_combination(bm25_results_dl, dense_results_dl)

bm25_ce_results_dl = rerank(cross_model, bm25_results_dl, trec_dl_queries, corpus_df)
dense_ce_results_dl = rerank(cross_model, dense_results_dl, trec_dl_queries, corpus_df)
hybrid_rrf_ce_results_dl = rerank(cross_model, hybrid_rrf_results_dl, trec_dl_queries, corpus_df)
hybrid_cc_ce_results_dl = rerank(cross_model, hybrid_cc_results_dl, trec_dl_queries, corpus_df)

trec_dl_dict = {
    "BM25":           bm25_results_dl,
    "Dense":          dense_results_dl,
    "Hybrid_RRF":     hybrid_rrf_results_dl,
    "Hybrid_CC":      hybrid_cc_results_dl,
    "BM25+CE":        bm25_ce_results_dl,
    "Dense+CE":       dense_ce_results_dl,
    "Hybrid_RRF+CE":  hybrid_rrf_ce_results_dl,
    "Hybrid_CC+CE":   hybrid_cc_ce_results_dl,
}
summary_dl = evaluate_trec_dl(trec_dl_dict, trec_dl_qrels)
print("\n=== RESULTS (TREC-DL 2019 — Graded) ===")
print(summary_dl.to_string(index=False))