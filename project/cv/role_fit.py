# -*- coding: utf-8 -*-
"""
cv/role_fit.py
Deterministic CV<->role reality-check: cosine content fit + a seniority-gap rule.
Pure ML — no Claude. These numbers are handed to the conversation controller as
facts, so the agent's encouragement/pushback is grounded, never hallucinated.
"""

from sklearn.metrics.pairwise import cosine_similarity

# Role-title keyword -> experience_rank (0-6), the SAME scale as the project's
# experience_rank. Ranks are max-combined, so a longer senior phrase always wins
# over a generic one (e.g. "senior data analyst" -> 4, not the title's base 0).
_ROLE_RANK_RULES = [
    (("intern", "internship", "trainee"), 1),
    (("junior", "entry", "graduate", "associate"), 2),
    (("mid", "senior", "lead", "principal", "staff"), 4),
    (("manager", "head", "vp", "vice president", "director"), 5),
    (("chief", "ceo", "cto", "cfo", "coo", "founder", "c-level", "executive"), 6),
]

# profile.seniority -> rank (same scale).
_SENIORITY_RANK = {
    "intern": 1, "entry": 2, "associate": 3,
    "mid-senior": 4, "director": 5, "executive": 6,
}


def _role_rank(role: str) -> int:
    """Implied seniority of a role title (0 = unknown / generic IC)."""
    r = role.lower()
    best = 0
    for keywords, rank in _ROLE_RANK_RULES:
        if any(k in r for k in keywords):
            best = max(best, rank)
    return best


def _cv_rank(profile: dict) -> int:
    """Candidate's seniority rank from profile.seniority, falling back to years."""
    sen = str(profile.get("seniority", "")).strip().lower()
    if sen in _SENIORITY_RANK:
        return _SENIORITY_RANK[sen]
    years = profile.get("years")
    if isinstance(years, (int, float)) and not isinstance(years, bool):
        if years < 1:
            return 2
        if years < 3:
            return 3
        if years < 7:
            return 4
        if years < 12:
            return 5
        return 6
    return 0


def candidate_role_text(profile: dict) -> str:
    """The role-defining part of the CV — the candidate's held titles + skills, NOT the whole
    document. Comparing this (role-to-role) against a job title is far more discriminating than
    diluting the full CV against a 3-word title (which scores ~uniformly low)."""
    p = profile or {}
    parts = [str(t) for t in p.get("titles_held", []) if t]
    parts += [str(s) for s in p.get("skills", []) if s]
    return " ".join(parts).strip().lower()


def role_fit(profile: dict, cv_text: str, role: str, vectorizer) -> float:
    """Cosine in [0,1] between the candidate's ROLE representation (held titles + skills,
    falling back to the full CV only when those are empty) and the role string."""
    cv_terms = candidate_role_text(profile) or (cv_text or "").lower()
    cv_vec = vectorizer.transform([cv_terms])
    role_vec = vectorizer.transform([str(role).lower()])
    return float(cosine_similarity(cv_vec, role_vec)[0, 0])


def seniority_gap(profile: dict, role: str) -> dict:
    """Compare the role's implied level to the candidate's. Returns role_rank,
    cv_rank, gap (role-cv) and a band: 'stretch' (aiming high), 'fit', 'under'."""
    rr = _role_rank(role)
    cr = _cv_rank(profile)
    gap = rr - cr
    if rr == 0 or cr == 0:        # unknown on either side -> don't flag
        band = "fit"
    elif gap >= 2:
        band = "stretch"
    elif gap <= -2:
        band = "under"
    else:
        band = "fit"
    return {"role_rank": rr, "cv_rank": cr, "gap": gap, "band": band}


def assess(profile: dict, cv_text: str, roles: list, vectorizer) -> list:
    """One {role, role_fit, seniority_gap} fact per non-blank role."""
    out = []
    for role in roles:
        role = str(role).strip()
        if not role:
            continue
        out.append({
            "role": role,
            "role_fit": round(role_fit(profile, cv_text, role, vectorizer), 3),
            "seniority_gap": seniority_gap(profile, role),
        })
    return out
