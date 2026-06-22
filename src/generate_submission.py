"""
Phase 2: load pre-computed artifacts, run hybrid retrieval,
score, and write submission.csv. Runs within the 5-min limit.
"""

import json
import time

import faiss
import pandas as pd

from src.config import TOP_K_SUBMISSION
from src.loader import load_candidates, load_job_description, extract_jd_query
from src.embedding import encode_jd
from src.bm25_index import load_bm25
from src.hybrid_retrieval import hybrid_retrieve
from src.ranker import rank_fused
from src.reasoning import generate_reasoning

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    t0 = time.time()
    print("[1/5] Loading artifacts...")

    with open("artifacts/candidate_ids.json") as f:
        candidate_ids = json.load(f)

    index = faiss.read_index("artifacts/faiss.index")
    bm25 = load_bm25("artifacts/bm25_index.pkl")

    print(f"[2/5] Loading candidates.jsonl...")
    candidates = load_candidates("data/candidates.jsonl")
    jd_text = load_job_description("data/job_description.txt")
    jd_query = extract_jd_query(jd_text)
    print(f"      Loaded {len(candidates)} candidates in {time.time()-t0:.1f}s")

    print("[3/5] Encoding job description...")
    jd_emb = encode_jd(jd_query)

    print("[4/5] Hybrid retrieval + scoring...")
    fused = hybrid_retrieve(
        jd_query, jd_emb, candidates, index, bm25, candidate_ids
    )
    top100 = rank_fused(fused)
    print(f"      Got {len(top100)} ranked candidates in {time.time()-t0:.1f}s")

    print("[5/5] Writing submission...")
    rows = []
    for rank, (c, sc) in enumerate(top100, start=1):
        rows.append({
            "candidate_id": c["candidate_id"],
            "rank": rank,
            "score": round(sc, 6),
            "reasoning": generate_reasoning(c, sc),
        })

    df = pd.DataFrame(rows, columns=["candidate_id", "rank", "score", "reasoning"])
    df.to_csv("output/submission.csv", index=False)
    print(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
