"""
Hybrid retrieval: merges BM25 and Dense retrieval candidate lists.

Implements two fusion strategies:

1. Reciprocal Rank Fusion (RRF) — rank-based, no score normalization needed.
   Reference: Cormack, Clarke & Buettcher (2009) — "Reciprocal Rank Fusion
   outperforms Condorcet and individual Rank Learning Methods"

2. Convex Combination (CC) — score-based linear interpolation with min-max
   normalization.  alpha * dense + (1 - alpha) * sparse.
"""
import pandas as pd
from src.config import RERANK_TOP_K , CC_ALPHA


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


# ---------------------------------------------------------------------------
# Convex Combination (weighted linear interpolation)
# ---------------------------------------------------------------------------

def _minmax_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Min-max normalize scores to [0, 1] **per query**.
    """
    out = df.copy()
    for qid, group in out.groupby("qid"):
        lo = group["score"].min()
        hi = group["score"].max()
        rng = hi - lo
        if rng > 0:
            out.loc[group.index, "score"] = (group["score"] - lo) / rng
        else:
            out.loc[group.index, "score"] = 0.0
    return out


def convex_combination(
    sparse_df: pd.DataFrame,
    dense_df: pd.DataFrame,
    alpha: float = CC_ALPHA,
    top_n: int = RERANK_TOP_K,
) -> pd.DataFrame:
    """
    Convex Combination (CC) — score-level fusion.

    final_score(d) = alpha * norm_dense(d) + (1 - alpha) * norm_sparse(d)

    Documents retrieved by only one system get 0 for the missing system's
    normalized score (outer join).

    Args:
        sparse_df: BM25 results DataFrame ['qid', 'docno', 'score', 'rank']
        dense_df:  Dense results DataFrame ['qid', 'docno', 'score', 'rank']
        alpha: weight for the dense component (0 = pure sparse, 1 = pure dense)
        top_n: number of results to return per query

    Returns:
        Fused DataFrame with columns ['qid', 'docno', 'score', 'rank']
    """
    # Normalize each system's scores per query
    sparse_norm = _minmax_normalize(sparse_df)[["qid", "docno", "score"]].rename(
        columns={"score": "score_sparse"}
    )
    dense_norm = _minmax_normalize(dense_df)[["qid", "docno", "score"]].rename(
        columns={"score": "score_dense"}
    )

    # Outer join — docs found by only one system get 0 for the other
    merged = pd.merge(sparse_norm, dense_norm, on=["qid", "docno"], how="outer")
    merged["score_sparse"] = merged["score_sparse"].fillna(0.0)
    merged["score_dense"] = merged["score_dense"].fillna(0.0)

    # Weighted linear combination
    merged["score"] = alpha * merged["score_dense"] + (1 - alpha) * merged["score_sparse"]

    # Keep top_n per query
    result = (
        merged[["qid", "docno", "score"]]
        .sort_values(["qid", "score"], ascending=[True, False])
        .groupby("qid")
        .head(top_n)
        .reset_index(drop=True)
    )
    result["rank"] = result.groupby("qid").cumcount()
    return result