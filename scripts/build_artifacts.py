"""
Phase 1: pre-compute dense embeddings, BM25 index, candidate IDs.
Allowed to be slow and online (downloads the sentence-transformer model).
"""

import sys
import os
import json
import time

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss
import numpy as np

from src.loader import load_candidates
from src.embedding import encode_candidates
from src.bm25_index import build_bm25, save_bm25


def main():
    t0 = time.time()

    print("Loading candidates...")
    candidates = load_candidates("data/candidates.jsonl")
    print(f"  {len(candidates)} candidates in {time.time()-t0:.1f}s")

    print("Encoding with sentence-transformers...")
    embs = encode_candidates(candidates)
    print(f"  embeddings shape: {embs.shape} in {time.time()-t0:.1f}s")

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    faiss.write_index(index, "artifacts/faiss.index")

    np.save("artifacts/candidate_embeddings.npy", embs)
    ids = [c["candidate_id"] for c in candidates]
    with open("artifacts/candidate_ids.json", "w") as f:
        json.dump(ids, f)

    print("Building BM25 index...")
    bm25, _ = build_bm25(candidates)
    save_bm25(bm25, "artifacts/bm25_index.pkl")
    print(f"  BM25 size: {len(bm25.doc_freqs)}")

    print(f"ALL DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
