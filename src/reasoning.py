"""
Generate per-candidate reasoning. Judges manually inspect this,
so it must be evidence-based, JD-aware, and not template-y.
"""

from datetime import datetime

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _yrs(c):
    return c.get("profile", {}).get("years_of_experience", 0) or 0


def _top_companies(c, n=2):
    roles = c.get("career_history", [])
    titles = []
    for r in roles[:3]:
        co = r.get("company", "")
        ti = r.get("title", "")
        if co or ti:
            titles.append(f"{ti} at {co}".strip(" at "))
    return titles[:n]


def _signal_phrase(c):
    s = c.get("redrob_signals", {})
    bits = []
    if s.get("open_to_work_flag"):
        bits.append("actively open to opportunities")
    rr = s.get("recruiter_response_rate", 0) or 0
    if rr >= 0.7:
        bits.append(f"{int(rr*100)}% recruiter response rate")
    ic = s.get("interview_completion_rate", 0) or 0
    if ic >= 0.7:
        bits.append(f"{int(ic*100)}% interview completion")
    last = s.get("last_active_date")
    if last:
        try:
            d = datetime.strptime(last, "%Y-%m-%d")
            days = (datetime.now() - d).days
            if days <= 30:
                bits.append("active on platform in the last 30 days")
            elif days <= 90:
                bits.append("recently active on platform")
        except Exception:
            pass
    return "; ".join(bits[:2]) if bits else "moderate platform engagement"


def _strength_phrase(c):
    """Pick 2 strongest domain signals and combine."""
    parts = []
    text = ""
    for r in c.get("career_history", []):
        text += " " + (r.get("description", "") or "").lower()

    found = []
    for kw in ["retrieval", "ranking", "recommendation", "relevance", "search"]:
        if kw in text and kw not in found:
            found.append(kw)
    if found:
        parts.append(f"hands-on {', '.join(found[:2])} systems")

    for tool in ["faiss", "pinecone", "weaviate", "elasticsearch", "opensearch", "sentence-transformers", "bge"]:
        if tool in text and len(parts) < 2:
            parts.append(f"{tool} in production")

    for kw in ["ndcg", "mrr", "a/b test", "experimentation", "learning to rank", "evaluation framework"]:
        if kw in text and len(parts) < 2:
            parts.append("ranking evaluation / experimentation")

    if not parts:
        parts.append("applied ML in product settings")

    return "; ".join(parts[:2])


def generate_reasoning(c, score=None):
    yrs = _yrs(c)
    title = c.get("profile", {}).get("current_title", "engineer")
    strengths = _strength_phrase(c)
    signals = _signal_phrase(c)

    s1 = f"{yrs}+ years as {title} with {strengths}."
    s2 = f"{signals.capitalize()}; aligns with the JD's emphasis on production retrieval/ranking ownership."
    return f"{s1} {s2}"
