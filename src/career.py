"""
Career-progression analysis. The JD values upward trajectories.
Detects: rapid title inflation vs genuine progression.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Title level rankings (rough industry heuristic)
_TITLE_LEVEL = {
    "intern": 0,
    "trainee": 0,
    "junior": 1,
    "associate": 2,
    "software engineer": 3,
    "engineer": 3,
    "developer": 3,
    "data scientist": 4,
    "machine learning engineer": 4,
    "applied scientist": 4,
    "senior engineer": 5,
    "senior software engineer": 5,
    "senior data scientist": 5,
    "staff engineer": 6,
    "principal engineer": 7,
    "tech lead": 5,
    "lead engineer": 6,
    "engineering manager": 6,
    "director": 7,
}


def _level(title):
    t = (title or "").lower()
    # Longest-keyword match
    best = 3
    for kw, lvl in _TITLE_LEVEL.items():
        if kw in t:
            if lvl > best:
                best = lvl
    return best


def progression_score(c):
    """
    0-1 score reflecting healthy upward progression without
    title-chasing (which the JD explicitly dislikes).
    """
    roles = sorted(
        c.get("career_history", []),
        key=lambda r: r.get("start_date", "") or "",
    )
    if len(roles) < 2:
        return 0.5

    levels = [_level(r.get("title", "")) for r in roles]

    # Penalize level drops (signals instability)
    drops = sum(1 for i in range(1, len(levels)) if levels[i] < levels[i - 1] - 1)

    # Reward monotonic or upward growth
    ups = sum(1 for i in range(1, len(levels)) if levels[i] > levels[i - 1])

    # Penalize title-chasing: many roles with short duration
    short_stints = sum(
        1 for r in roles
        if (r.get("duration_months") or 0) < 14 and not r.get("is_current")
    )

    # Average tenure (in months)
    avg_tenure = sum((r.get("duration_months") or 0) for r in roles) / len(roles)

    score = 0.5
    score += 0.10 * min(ups, 4)              # up to 0.4 reward
    score -= 0.15 * drops                     # drops are bad
    score -= 0.10 * min(short_stints, 3)      # short stints bad
    score += 0.05 if avg_tenure >= 24 else 0  # 2y+ avg tenure bonus
    score += 0.10 if avg_tenure >= 36 else 0  # 3y+ tenure bonus

    return max(0.0, min(1.0, score))
