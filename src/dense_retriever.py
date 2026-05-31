"""
Bi-encoder dense retrieval using sentence-transformers + FAISS.

Model: msmarco-distilbert-base-v3
  - Trained on MS MARCO with dot-product similarity
  - Use dot_score / inner product (IndexFlatIP), NOT cosine after L2-norm
"""
import gc
import os
import numpy as np
import faiss
import torch
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from src.config import (
    BIENCODER_MODEL, FAISS_INDEX_PATH, PASSAGE_IDS_PATH,
    EMBEDDING_DIM, FAISS_BATCH_SIZE, DEVICE, DENSE_TOP_K
)

# Number of passages to encode before flushing to FAISS index.
# 50K × 768 × 4 bytes ≈ 150 MB per chunk — safe for most machines.
ENCODE_CHUNK_SIZE = 50_000


def load_biencoder() -> SentenceTransformer:
    model = SentenceTransformer(BIENCODER_MODEL, device=DEVICE)
    return model


def encode_corpus(
    model: SentenceTransformer,
    corpus_df: pd.DataFrame,
    batch_size: int = FAISS_BATCH_SIZE,
    chunk_size: int = ENCODE_CHUNK_SIZE,
    save: bool = True,
) -> None:
    """
    Encode all passages in chunks and build the FAISS index incrementally.

    Instead of materializing all embeddings in RAM at once (which causes OOM
    on large corpora), we encode `chunk_size` passages at a time, add them
    to the FAISS index immediately, and free the chunk memory.

    Peak RAM ≈ model + 1 chunk of embeddings (instead of full corpus).
    """
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    texts = corpus_df["text"].tolist()
    passage_ids = corpus_df["docno"].tolist()
    n = len(texts)

    # Save passage IDs upfront
    np.save(PASSAGE_IDS_PATH, np.array(passage_ids))

    # Create an empty FAISS index
    index = faiss.IndexFlatIP(EMBEDDING_DIM)

    # Encode and add in chunks
    for start in tqdm(range(0, n, chunk_size), desc="Encoding chunks"):
        end = min(start + chunk_size, n)
        chunk_texts = texts[start:end]

        chunk_emb = model.encode(
            chunk_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype("float32")

        index.add(chunk_emb)

        # Free chunk memory
        del chunk_emb
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    if save:
        faiss.write_index(index, FAISS_INDEX_PATH)

    print(f"FAISS index saved: {index.ntotal} vectors")


def load_faiss_index():
    index = faiss.read_index(FAISS_INDEX_PATH)
    passage_ids = np.load(PASSAGE_IDS_PATH, allow_pickle=True).tolist()
    return index, passage_ids


def dense_retrieve(
    model: SentenceTransformer,
    faiss_index,
    passage_ids: list,
    queries_df: pd.DataFrame,
    top_k: int = DENSE_TOP_K,
) -> pd.DataFrame:
    """
    Returns a results DataFrame compatible with PyTerrier: ['qid', 'docno', 'score', 'rank']
    """
    query_embeddings = model.encode(
        queries_df["query"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype("float32")

    scores, indices = faiss_index.search(query_embeddings, top_k)

    rows = []
    for i, qid in enumerate(queries_df["qid"].tolist()):
        for rank, (idx, score) in enumerate(zip(indices[i], scores[i])):
            rows.append({
                "qid": qid,
                "docno": passage_ids[idx],
                "score": float(score),
                "rank": rank,
            })
    return pd.DataFrame(rows)