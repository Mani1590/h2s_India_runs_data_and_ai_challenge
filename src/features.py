"""
Structured features computed once per candidate (Phase 1),
then loaded and combined with hybrid retrieval in Phase 2.
"""

import re
import numpy as np
import pandas as pd

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    SERVICE_COMPANIES,
    CORE_TOOLS,
    EVAL_KEYWORDS,
    RETRIEVAL_KEYWORDS,
    EXP_IDEAL_MIN,
    EXP_IDEAL_MAX,
)


def _haystack(c):
    """All text fields combined into one lowercase blob."""
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
    for s in c.get("skills", []):
        parts.append(s.get("name", ""))
    return " ".join(parts).lower()


def retrieval_experience(c):
    """Density of retrieval/ranking/recommendation concepts in career text."""
    text = _haystack(c)
    hits = sum(1 for k in RETRIEVAL_KEYWORDS if k in text)
    # Normalize; capped at 5 distinct hits
    return min(hits / 5.0, 1.0)


def evaluation_experience(c):
    """Did the candidate build or use ranking eval systems?"""
    text = _haystack(c)
    hits = sum(1 for k in EVAL_KEYWORDS if k in text)
    return min(hits / 3.0, 1.0)


def tool_match(c):
    """Specific tools named in the JD: FAISS, Pinecone, etc."""
    text = _haystack(c)
    hits = sum(1 for k in CORE_TOOLS if k in text)
    # Also count tools listed in candidate's skills list directly
    skill_names = " ".join(s.get("name", "") for s in c.get("skills", [])).lower()
    hits += sum(1 for k in CORE_TOOLS if k in skill_names)
    return min(hits / 4.0, 1.0)


def product_company_score(c):
    """Fraction of career spent outside service companies."""
    roles = c.get("career_history", [])
    if not roles:
        return 0.0
    bad = 0
    total_months = 0
    for r in roles:
        co = r.get("company", "").lower().strip()
        months = r.get("duration_months", 0) or 0
        total_months += months
        for s in SERVICE_COMPANIES:
            if s in co:
                bad += months
                break
    if total_months == 0:
        return 0.0
    return 1.0 - (bad / total_months)


def experience_score(c):
    """Years-of-experience fit to 5-9 band."""
    yrs = c.get("profile", {}).get("years_of_experience", 0) or 0
    if EXP_IDEAL_MIN <= yrs <= EXP_IDEAL_MAX:
        return 1.0
    if 4 <= yrs < EXP_IDEAL_MIN:
        return 0.75
    if EXP_IDEAL_MAX < yrs <= 12:
        return 0.7
    if 3 <= yrs < 4:
        return 0.5
    if 12 < yrs <= 15:
        return 0.5
    return 0.3


def signal_score(c):
    """
    Behavioral availability signals.
    JD explicitly: 'hasn't logged in for 6 months → not actually available'
    """
    s = c.get("redrob_signals", {})
    if not s:
        return 0.0

    score = 0.0
    score += 0.30 if s.get("open_to_work_flag") else 0.0
    score += float(s.get("recruiter_response_rate", 0)) * 0.25
    score += float(s.get("interview_completion_rate", 0)) * 0.20
    score += min(float(s.get("profile_completeness_score", 0)) / 100.0, 1.0) * 0.10
    score += min(float(s.get("saved_by_recruiters_30d", 0)) / 30.0, 1.0) * 0.10
    score += min(float(s.get("search_appearance_30d", 0)) / 100.0, 1.0) * 0.05

    return min(score, 1.0)


def education_tier_score(c):
    """Map tier_1..tier_4 to a 0-1 score."""
    edu = c.get("education", [])
    if not edu:
        return 0.3
    tier_map = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5, "tier_4": 0.3, "unknown": 0.4}
    best = max(tier_map.get(e.get("tier", "unknown"), 0.4) for e in edu)
    return best


def github_score(c):
    """github_activity_score: -1 (not linked) to 100."""
    v = c.get("redrob_signals", {}).get("github_activity_score", -1)
    if v is None or v < 0:
        return 0.0
    return min(float(v) / 100.0, 1.0)


# ---------- Vectorized feature builder (Phase 1) ----------

def build_feature_frame(candidates):
    rows = []
    for c in candidates:
        rows.append({
            "candidate_id": c["candidate_id"],
            "retrieval_experience": retrieval_experience(c),
            "evaluation_experience": evaluation_experience(c),
            "tool_match": tool_match(c),
            "product_company": product_company_score(c),
            "experience_score": experience_score(c),
            "signal_score": signal_score(c),
            "education_tier": education_tier_score(c),
            "github_score": github_score(c),
        })
    return pd.DataFrame(rows)
