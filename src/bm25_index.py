"""
BM25 index over candidate text.
Stored to disk in Phase 1, loaded in Phase 2.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
import pickle
from rank_bm25 import BM25Okapi

from src.embedding import candidate_text


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#\.\-]+")


def tokenize(text):
    return _TOKEN_RE.findall(text.lower())


def build_bm25(candidates):
    corpus_tokens = []
    for c in candidates:
        toks = tokenize(candidate_text(c))
        corpus_tokens.append(toks[:8000])
    return BM25Okapi(corpus_tokens), corpus_tokens


def save_bm25(bm25, path):
    with open(path, "wb") as f:
        pickle.dump(bm25, f)


def load_bm25(path):
    with open(path, "rb") as f:
        return pickle.load(f)
