# -*- coding: utf-8 -*-
"""
cv/scorecard.py
Proactive CV<->role fit scorecard for the agent's opening message.

Measurable ML does the scoring; Claude only supplies vocabulary:
  - match%     = role_fit cosine(CV, role)                       [role_fit.py]
  - seniority  = seniority_gap band (stretch / fit / under)      [role_fit.py]
  - have / gap = DETERMINISTIC set-diff of the CV's skills against the role's
                 typical skills. Claude provides the typical-skill list (vocab
                 only); the overlap and the gap are computed here, not guessed.

Degrades gracefully: with no Claude client the cards still carry match% +
seniority (the ML parts); only the have/gap detail is omitted.
"""

import matching                 # leaf util — safe (no dataset load)
from . import role_fit

# seniority_gap band -> short Hebrew phrase for the scorecard line.
_BAND_HE = {
    "stretch": "קפיצה מעל רמת הניסיון הנוכחית",
    "fit":     "מתאים לרמת הניסיון שלך",
    "under":   "מתחת לרמת הניסיון שלך",
}

_SKILLS_SYSTEM = (
    "Given a single job-role title, list the 4-6 core skills or tools typically required "
    'for it. Return ONLY a JSON object: {"skills": ["sql","python", ...]} — short, lowercase '
    "skill strings. No prose."
)


def _role_skills(role: str, claude_client, model: str) -> list:
    """The role's typical core skills (vocabulary only), via one Claude call. [] if no client."""
    if claude_client is None:
        return []
    raw = matching.reply_json(claude_client, model, f"ROLE: {role}", _SKILLS_SYSTEM, max_tokens=200)
    skills = raw.get("skills") if isinstance(raw, dict) else None
    return [str(s).strip().lower() for s in (skills or []) if str(s).strip()]


def _cv_skill_terms(profile: dict, cv_text: str) -> tuple:
    """The CV's skill evidence: the profile's skill set + the raw CV text (lowercased)."""
    skill_set = {str(s).strip().lower() for s in (profile or {}).get("skills", []) if str(s).strip()}
    return skill_set, (cv_text or "").lower()


def split_have_gap(role_skills: list, profile: dict, cv_text: str,
                   max_have: int = 4, max_gap: int = 3) -> tuple:
    """Deterministically split a role's typical skills into what the CV has vs lacks."""
    skill_set, text = _cv_skill_terms(profile, cv_text)
    have, gap = [], []
    for s in role_skills:
        s_l = s.lower()
        present = (s_l in skill_set) or (s_l in text) or any(s_l in t or t in s_l for t in skill_set)
        (have if present else gap).append(s)
    return have[:max_have], gap[:max_gap]


def build(roles: list, profile: dict, cv_text: str, vectorizer,
          claude_client, model: str) -> list:
    """One scorecard per role. Only the FIRST role gets the have/gap skill detail
    (the agent presents the rest as plain suggestions)."""
    cards = []
    for role in roles:
        role = str(role).strip()
        if not role:
            continue
        match = round(role_fit.role_fit(profile or {}, cv_text or "", role, vectorizer) * 100)
        band = role_fit.seniority_gap(profile or {}, role)["band"]
        cards.append({"role": role, "match": match, "seniority": _BAND_HE.get(band, "")})

    # Rank by the ML match so the most-fitting role leads (and gets the skill detail).
    cards.sort(key=lambda c: c["match"], reverse=True)
    if cards:
        have, gap = split_have_gap(_role_skills(cards[0]["role"], claude_client, model), profile, cv_text)
        cards[0]["have"], cards[0]["gap"] = have, gap
    return cards
