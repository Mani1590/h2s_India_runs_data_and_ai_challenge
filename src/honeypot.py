"""
Honeypot detection. The dataset contains ~80 candidates with
subtly impossible profiles. Returning >10% honeypots in top 100 = DQ.

We compute a 0-1 trust score per candidate. Multi-signal:
- experience vs career duration mismatch
- title seniority vs years of experience
- skill proficiency vs declared duration
- impossible date overlaps
- everything-perfect patterns (likely synthetic)
"""

from datetime import datetime

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _months_between(start, end):
    try:
        s = datetime.strptime(start, "%Y-%m-%d")
        e = datetime.strptime(end, "%Y-%m-%d") if end else datetime.now()
    except Exception:
        return None
    return max(0, (e.year - s.year) * 12 + (e.month - s.month))


def trust_score(c):
    """Higher = more trustworthy. Lower = more likely honeypot."""
    p = c.get("profile", {})
    s = c.get("redrob_signals", {})
    issues = []

    # 1. Experience vs career-duration consistency
    declared_yrs = p.get("years_of_experience", 0) or 0
    total_months = sum(
        (r.get("duration_months") or 0)
        for r in c.get("career_history", [])
    )
    actual_yrs = total_months / 12.0
    yrs_diff = abs(declared_yrs - actual_yrs)
    if yrs_diff > 4:
        issues.append(("years_mismatch", yrs_diff))

    # 2. Date overlap consistency
    intervals = []
    for r in c.get("career_history", []):
        m = _months_between(r.get("start_date", ""), r.get("end_date"))
        if m is not None:
            intervals.append((r.get("start_date"), r.get("end_date"), m))
    declared_total = sum((i[2] or 0) for i in intervals)
    # If declared durations sum to something wildly different from intervals
    if abs(declared_total - total_months) > 12:
        issues.append(("duration_drift", abs(declared_total - total_months)))

    # 3. Skill proficiency vs duration mismatch
    for sk in c.get("skills", []):
        prof = sk.get("proficiency", "")
        dur = sk.get("duration_months", 0) or 0
        if prof in ("expert", "advanced") and dur <= 1:
            issues.append(("skill_duration", sk.get("name")))
        # proficiency but never used in any role
        used_in_roles = any(
            sk.get("name", "").lower() in (r.get("description", "").lower())
            for r in c.get("career_history", [])
        )
        if not used_in_roles and prof == "expert":
            issues.append(("phantom_expert", sk.get("name")))

    # 4. Title seniority vs experience
    yrs = declared_yrs
    titles = " ".join(
        (r.get("title", "") or "") for r in c.get("career_history", [])
    ).lower()
    senior_keywords = ["principal", "staff", "director", "head", "vp", "chief"]
    is_senior = any(k in titles for k in senior_keywords)
    if is_senior and yrs < 4:
        issues.append(("senior_junior", yrs))

    # 5. Everything-perfect pattern (likely synthetic honeypot)
    if s:
        perfect_signals = 0
        if s.get("recruiter_response_rate", 0) >= 0.98:
            perfect_signals += 1
        if s.get("interview_completion_rate", 0) >= 0.98:
            perfect_signals += 1
        if s.get("offer_acceptance_rate", -1) >= 0.95:
            perfect_signals += 1
        if s.get("profile_completeness_score", 0) >= 99:
            perfect_signals += 1
        if perfect_signals >= 3:
            issues.append(("too_perfect", perfect_signals))

    # 6. Excessive skill count vs career duration
    n_skills = len(c.get("skills", []))
    if n_skills > 40 and actual_yrs < 6:
        issues.append(("skill_count", n_skills))

    # Convert issues into a 0-1 trust score
    penalty = 0.0
    weights = {
        "years_mismatch": 0.25,
        "duration_drift": 0.20,
        "skill_duration": 0.15,
        "phantom_expert": 0.20,
        "senior_junior": 0.30,
        "too_perfect": 0.30,
        "skill_count": 0.10,
    }
    for tag, _ in issues:
        penalty += weights.get(tag, 0.10)

    return max(0.0, 1.0 - penalty), issues


def is_honeypot(c, threshold=0.5):
    """Hard filter — drop candidates below this trust threshold."""
    score, issues = trust_score(c)
    return score < threshold, score, issues
