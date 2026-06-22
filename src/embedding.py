import os
import sys

# Make 'src' importable from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import MODEL_NAME

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def candidate_text(c):
    """Concatenate all text-rich fields of a candidate."""
    p = c.get("profile", {})
    parts = [
        p.get("headline", ""),
        p.get("summary", ""),
        p.get("current_title", ""),
        p.get("current_industry", ""),
    ]
    for role in c.get("career_history", []):
        parts.append(role.get("title", ""))
        parts.append(role.get("description", ""))
        parts.append(role.get("industry", ""))
    for skill in c.get("skills", []):
        parts.append(skill.get("name", ""))
    for cert in c.get("certifications", []):
        parts.append(cert.get("name", ""))
    return " ".join(x for x in parts if x)


def encode_jd(jd_text):
    model = get_model()
    emb = model.encode(
        [jd_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return emb


def encode_candidates(candidates, batch_size=128):
    model = get_model()
    texts = [candidate_text(c) for c in candidates]
    embs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embs
