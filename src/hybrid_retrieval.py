"""
Hybrid retrieval: BM25 + dense embeddings, fused via RRF
(Reciprocal Rank Fusion). Both indices are pre-computed.
"""

import numpy as np
import faiss
from rank_bm25 import BM25Okapi

from src.config import TOP_K_DENSE, TOP_K_BM25, TOP_K_FUSED
from src.embedding import encode_jd
from src.bm25_index import tokenize
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rrf_fuse(dense_ranked, bm25_ranked, k=60):
    """
    dense_ranked, bm25_ranked: lists of (candidate_id, original_index) ordered.
    Returns dict: candidate_id -> rrf_score.
    """
    scores = {}
    for rank, (cid, _) in enumerate(dense_ranked):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, (cid, _) in enumerate(bm25_ranked):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return scores


def hybrid_retrieve(jd_text, jd_embedding, candidates, dense_index, bm25, candidate_ids):
    """
    Returns list of (candidate, dense_sim) for the TOP_K_FUSED candidates.
    dense_sim is preserved for the final scoring formula.
    """
    # ---- Dense side ----
    # jd_embedding is already normalized
    D, I = dense_index.search(jd_embedding, TOP_K_DENSE)
    dense_ranked = []
    dense_sim = {}
    for sim, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        cid = candidate_ids[idx]
        dense_ranked.append((cid, idx))
        dense_sim[idx] = float(sim)

    # ---- BM25 side ----
    q_tokens = tokenize(jd_text)
    bm25_scores = bm25.get_scores(q_tokens)
    top_bm25_idx = np.argsort(bm25_scores)[::-1][:TOP_K_BM25]
    bm25_ranked = [(candidate_ids[i], int(i)) for i in top_bm25_idx]

    # ---- Fuse ----
    rrf = rrf_fuse(dense_ranked, bm25_ranked)
    fused_sorted = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:TOP_K_FUSED]

    # Map back to candidate objects and dense scores
    idx_to_candidate = {c["candidate_id"]: c for c in candidates}
    out = []
    for cid, _ in fused_sorted:
        c = idx_to_candidate[cid]
        idx = candidate_ids.index(cid)  # small dict lookup fine
        out.append((c, dense_sim.get(idx, 0.0)))
    return out
