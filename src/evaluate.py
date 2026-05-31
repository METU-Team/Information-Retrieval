"""
Evaluation module.
Computes MRR@10 and nDCG@10 from results and qrels DataFrames.
Do NOT use MAP (per instructor guidance: MS MARCO has sparse binary relevance).
"""
import numpy as np
import pandas as pd
from typing import Callable
import pyterrier as pt

if not pt.started():
    pt.init()


def compute_mrr_at_k(results_df: pd.DataFrame, qrels_df: pd.DataFrame, k: int = 10) -> float:
    relevant = set(zip(qrels_df["qid"], qrels_df["docno"]))
    mrr_scores = []
    for qid, group in results_df.groupby("qid"):
        group = group.sort_values("score", ascending=False).head(k)
        rr = 0.0
        for rank, (_, row) in enumerate(group.iterrows(), start=1):
            if (qid, row["docno"]) in relevant:
                rr = 1.0 / rank
                break
        mrr_scores.append(rr)
    return float(np.mean(mrr_scores))


def compute_ndcg_at_k(results_df: pd.DataFrame, qrels_df: pd.DataFrame, k: int = 10) -> float:
    rel_lookup = {}
    for _, row in qrels_df.iterrows():
        rel_lookup[(row["qid"], row["docno"])] = int(row["label"])

    ndcg_scores = []
    for qid, group in results_df.groupby("qid"):
        group = group.sort_values("score", ascending=False).head(k)
        dcg = sum(
            rel_lookup.get((qid, row["docno"]), 0) / np.log2(rank + 2)
            for rank, (_, row) in enumerate(group.iterrows())
        )
        ideal_rels = sorted(
            [v for (q, _), v in rel_lookup.items() if q == qid], reverse=True
        )[:k]
        idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal_rels))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcg_scores))


def evaluate_all(results_dict: dict, qrels_df: pd.DataFrame) -> pd.DataFrame:
    """
    results_dict: {'bm25': df, 'dense': df, 'hybrid_reranked': df}
    Returns a summary DataFrame with MRR@10 and nDCG@10 for each system.
    (Custom implementation)
    """
    rows = []
    for system_name, results_df in results_dict.items():
        rows.append({
            "System": system_name,
            "MRR@10": round(compute_mrr_at_k(results_df, qrels_df, k=10), 4),
            "nDCG@10": round(compute_ndcg_at_k(results_df, qrels_df, k=10), 4),
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# trec_eval-compatible evaluation via ir_measures
# (ir_measures ships with PyTerrier — no extra install needed)
# ---------------------------------------------------------------------------
def evaluate_all_ir_measures(results_dict: dict, qrels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardized evaluation using ir_measures, which produces results
    identical to trec_eval.  This is an alternative to the custom
    implementations above and can be used for reproducible benchmarking.

    results_dict: {'bm25': df, 'dense': df, ...}
    Returns a summary DataFrame with MRR@10 and nDCG@10 for each system.
    """
    import ir_measures
    from ir_measures import nDCG, RR  # RR = Reciprocal Rank (≡ MRR)

    measures = [RR@10, nDCG@10]

    # Convert qrels DataFrame → list[ir_measures.Qrel]
    qrels = [
        ir_measures.Qrel(str(row["qid"]), str(row["docno"]), int(row["label"]))
        for _, row in qrels_df.iterrows()
    ]

    rows = []
    for system_name, results_df in results_dict.items():
        # Convert results DataFrame → list[ir_measures.ScoredDoc]
        run = [
            ir_measures.ScoredDoc(str(row["qid"]), str(row["docno"]), float(row["score"]))
            for _, row in results_df.iterrows()
        ]
        agg = ir_measures.calc_aggregate(measures, qrels, run)
        rows.append({
            "System": system_name,
            **{str(m): round(v, 4) for m, v in agg.items()},
        })

    df = pd.DataFrame(rows)
    if "RR@10" in df.columns:
        df = df.rename(columns={"RR@10": "MRR@10"})
    return df[["System", "MRR@10", "nDCG@10"]]


def evaluate_trec_dl(results_dict: dict, qrels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluation with TREC-DL graded relevance (0-3 scale).

    Metrics:
      - nDCG@10:    Naturally leverages graded relevance (0/1/2/3).
      - RR(rel=2)@10: MRR where only 'highly relevant' (≥2) counts as a hit.
                       This is the standard TREC-DL convention — a doc must be
                       at least "highly relevant" to be considered a successful
                       retrieval for RR.

    results_dict: {'bm25': df, 'dense': df, ...}
    Returns a summary DataFrame.
    """
    import ir_measures
    from ir_measures import nDCG, RR

    measures = [
        nDCG@10,             # graded: uses actual 0/1/2/3 labels
        RR(rel=2)@10,        # binary at threshold 2: "highly relevant" or better
    ]

    qrels = [
        ir_measures.Qrel(str(row["qid"]), str(row["docno"]), int(row["label"]))
        for _, row in qrels_df.iterrows()
    ]

    rows = []
    for system_name, results_df in results_dict.items():
        run = [
            ir_measures.ScoredDoc(str(row["qid"]), str(row["docno"]), float(row["score"]))
            for _, row in results_df.iterrows()
        ]
        agg = ir_measures.calc_aggregate(measures, qrels, run)
        rows.append({
            "System": system_name,
            **{str(m): round(v, 4) for m, v in agg.items()},
        })

    df = pd.DataFrame(rows)
    if "RR(rel=2)@10" in df.columns:
        df = df.rename(columns={"RR(rel=2)@10": "MRR(rel=2)@10"})
    return df[["System", "MRR(rel=2)@10", "nDCG@10"]]