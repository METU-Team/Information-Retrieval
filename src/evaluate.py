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
    """
    rows = []
    for system_name, results_df in results_dict.items():
        rows.append({
            "System": system_name,
            "MRR@10": round(compute_mrr_at_k(results_df, qrels_df, k=10), 4),
            "nDCG@10": round(compute_ndcg_at_k(results_df, qrels_df, k=10), 4),
        })
    return pd.DataFrame(rows)