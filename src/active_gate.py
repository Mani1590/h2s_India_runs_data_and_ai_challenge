"""
Active candidate gating. JD: 'hasn't logged in for 6 months
is, for hiring purposes, not actually available.'

Two outputs:
- inactive_penalty: multiplicative factor in [0.3, 1.0]
- hard_filter: bool, if True candidate is excluded entirely
"""

from datetime import datetime, timedelta
from src.config import LAST_ACTIVE_MAX_DAYS, MIN_RESPONSE_RATE

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except Exception:
        return None


def _days_ago(s):
    d = _parse_date(s)
    if not d:
        return 9999
    return (datetime.now() - d).days


def evaluate(c, today=None):
    if today is None:
        today = datetime.now()

    s = c.get("redrob_signals", {})
    last_active = s.get("last_active_date")
    signup = s.get("signup_date")

    days_since_active = _days_ago(last_active)
    days_since_signup = _days_ago(signup) if signup else 9999
    # Convert: smaller is more recent; if signup recent, treat as fresh-active
    # Actually days_ago returns days since the event, so:
    # last_active = how many days ago they were last active
    # signup = how many days ago they signed up

    response_rate = float(s.get("recruiter_response_rate", 1.0))
    open_to_work = bool(s.get("open_to_work_flag"))

    # ---------- Hard filter ----------
    # Brand-new signup AND no activity AND zero response = clearly junk
    if days_since_active > 365 and days_since_signup < 7:
        return 0.3, True

    # Inactive for >9 months AND low response rate AND not open-to-work
    if (
        days_since_active > LAST_ACTIVE_MAX_DAYS
        and response_rate < MIN_RESPONSE_RATE
        and not open_to_work
    ):
        return 0.4, True

    # ---------- Soft penalty ----------
    penalty = 1.0

    if days_since_active > 180:
        penalty *= 0.7
    elif days_since_active > 90:
        penalty *= 0.85

    if response_rate < 0.10 and days_since_signup > 60:
        penalty *= 0.8

    return max(0.3, penalty), False
