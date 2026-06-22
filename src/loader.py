import gzip
import json

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_candidates(path):
    """Load candidates.jsonl(.gz) into a list of dicts."""
    opener = gzip.open if path.endswith(".gz") else open
    candidates = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def load_job_description(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_jd_query(jd_text):
    """
    Build a focused query string from the JD.
    Long JDs get truncated by the model anyway; we extract the
    most signal-dense sections to maximize embedding quality.
    """
    sections = []
    markers = [
        "What you'd actually be doing",
        "The skills inventory",
        "Things you absolutely need",
        "What we mean by",
        "How to read between the lines",
    ]
    text_lower = jd_text

    for m in markers:
        idx = text_lower.find(m)
        if idx != -1:
            sections.append(jd_text[idx:idx + 1500])

    if not sections:
        return jd_text[:3000]

    return " ".join(sections)[:4000]
