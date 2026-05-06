"""
Hybrid retrieval: merges BM25 and Dense retrieval candidate lists.

Implements Reciprocal Rank Fusion (RRF), the standard approach for
combining ranked lists from heterogeneous retrieval systems without
requiring score normalization.

Reference: Cormack, Clarke & Buettcher (2009) — "Reciprocal Rank Fusion
outperforms Condorcet and individual Rank Learning Methods"
"""
import pandas as pd
from src.config import RERANK_TOP_K


def reciprocal_rank_fusion(
    result_dfs: list[pd.DataFrame],
    k: int = 60,
    top_n: int = RERANK_TOP_K,
) -> pd.DataFrame:
    """
    Reciprocal Rank Fusion (RRF) — merges multiple ranked lists.

    For each document d in query q, the RRF score is:
        RRF(d) = Σ  1 / (k + rank_i(d))
    where the sum is over all systems i that retrieved d, and k is a
    smoothing constant (default 60, per the original paper).

    Args:
        result_dfs: list of DataFrames, each with ['qid', 'docno', 'score', 'rank']
        k: RRF smoothing constant (higher = less weight to top ranks)
        top_n: number of results to return per query

    Returns:
        Fused DataFrame with columns ['qid', 'docno', 'score', 'rank']
    """
    fused = {}  # (qid, docno) → cumulative RRF score

    for df in result_dfs:
        for qid, group in df.groupby("qid"):
            ranked = group.sort_values("score", ascending=False).reset_index(drop=True)
            for rank_pos, (_, row) in enumerate(ranked.iterrows()):
                key = (qid, row["docno"])
                rrf_score = 1.0 / (k + rank_pos + 1)  # rank is 1-indexed in formula
                fused[key] = fused.get(key, 0.0) + rrf_score

    # Build output DataFrame
    rows = []
    for (qid, docno), score in fused.items():
        rows.append({"qid": qid, "docno": docno, "score": score})

    result = pd.DataFrame(rows)

    # Keep top_n per query, sorted by fused score
    
    result = (
        result.sort_values(["qid", "score"], ascending=[True, False])
        .groupby("qid")
        .head(top_n)
        .reset_index(drop=True)
    )
    result["rank"] = result.groupby("qid").cumcount()
    return result