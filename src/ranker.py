"""
Final scoring: combines hybrid-retrieval similarity with
engineered features, then applies honeypot + active penalties.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import (
    retrieval_experience,
    evaluation_experience,
    tool_match,
    product_company_score,
    experience_score,
    signal_score,
    education_tier_score,
    github_score,
)
from src.career import progression_score
from src.honeypot import is_honeypot
from src.active_gate import evaluate as active_evaluate

WEIGHTS = {
    "retrieval":  0.28,
    "semantic":   0.18,
    "product":    0.15,
    "evaluation": 0.10,
    "signal":     0.08,
    "experience": 0.07,
    "tool":       0.06,
    "career":     0.04,
    "education":  0.04,
}


def score_candidate(c, dense_sim):
    _, trust, _ = is_honeypot(c, threshold=0.45)
    active_pen, hard_drop = active_evaluate(c)

    if hard_drop:
        return None

    s = (
        WEIGHTS["retrieval"]  * retrieval_experience(c)
      + WEIGHTS["semantic"]   * max(0.0, dense_sim)
      + WEIGHTS["product"]    * product_company_score(c)
      + WEIGHTS["evaluation"] * evaluation_experience(c)
      + WEIGHTS["signal"]     * signal_score(c)
      + WEIGHTS["experience"] * experience_score(c)
      + WEIGHTS["tool"]       * tool_match(c)
      + WEIGHTS["career"]     * progression_score(c)
      + WEIGHTS["education"]  * education_tier_score(c)
      + 0.02                 * github_score(c)
    )

    s *= active_pen
    s *= trust
    return s


def rank_fused(fused_candidates):
    scored = []
    for c, sim in fused_candidates:
        sc = score_candidate(c, sim)
        if sc is None:
            continue
        scored.append((c, sc))

    scored.sort(key=lambda x: (-x[1], x[0]["candidate_id"]))
    return scored[:100]
