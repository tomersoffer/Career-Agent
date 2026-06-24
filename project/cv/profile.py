# -*- coding: utf-8 -*-
"""
cv/profile.py
Build a structured candidate profile from raw CV text via one Claude call.

Dependency-injection: claude_client and model are passed in — never imported
from agent_runner.
"""

import sys

import matching  # leaf util — safe to import (no dataset load)

# Ensure Hebrew prints correctly on Windows consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_EMPTY_PROFILE = {
    # is_cv defaults True so we FAIL OPEN: when the LLM is unavailable or returns
    # malformed JSON, a genuine CV is never wrongly rejected. Only an explicit
    # is_cv:false from the model triggers the "not a CV" path.
    "is_cv": True,
    "cv_confidence": 0.0,
    "name": "",
    "location": "",
    "skills": [],
    "titles_held": [],
    "seniority": "",
    "years": None,
    "city": "",
    "state": "",
    "domains": [],
    "work_experience": [],
    "projects": [],
}

_SYSTEM_PROMPT = (
    "You are a CV-parsing module. First decide whether the text below is actually a "
    "résumé / CV (a document describing one person's work experience, skills and education). "
    "If it is NOT a CV (e.g. an invoice, an essay, an article, a contract, a random document), "
    'set "is_cv": false and leave the other fields empty — do not invent a profile. '
    "Otherwise extract a structured JSON profile (a candidate 'identity card') from the CV text. "
    "Respond with ONLY a valid JSON object — no prose, no code fences — "
    "with EXACTLY these keys:\n"
    '  "is_cv"        : bool       — true only if the text is a résumé/CV.\n'
    '  "cv_confidence": number     — confidence in the is_cv decision, 0.0–1.0.\n'
    '  "name"         : str        — the candidate\'s full name as written in the CV; "" if absent.\n'
    '  "location"     : str        — human-readable location as on the CV (e.g. "Tel Aviv, Israel"); "" if absent.\n'
    '  "skills"       : list[str]  — technical and soft skills found in the CV (lowercase).\n'
    '  "titles_held"  : list[str]  — job titles the candidate has actually held.\n'
    '  "seniority"    : str        — one of "intern","entry","associate","mid-senior","director","executive",'
    ' or "" if unclear.\n'
    '  "years"        : int|null   — total years of professional experience, or null if not determinable.\n'
    '  "city"         : str        — the candidate\'s current or most recent city (English); "" if absent.\n'
    '  "state"        : str        — 2-letter US state code if the city is in the US, else ""; "" if absent.\n'
    '  "domains"      : list[str]  — broad industry/functional domains (e.g. "data science", "finance").\n'
    '  "work_experience": list[obj] — roles held, MOST RECENT FIRST, each '
    '{"title": str, "company": str, "period": str, "highlights": list[str] (1-3 short bullets)}.\n'
    '  "projects"     : list[obj]  — notable projects, each {"name": str, "description": str (one line)}.\n'
    "Keep values concise and factual — do not invent anything not present in the CV."
)


def _norm_experience(x):
    """Coerce one work-experience entry into a stable {title, company, period, highlights} dict."""
    if not isinstance(x, dict):
        return {"title": str(x).strip(), "company": "", "period": "", "highlights": []}
    return {
        "title":      str(x.get("title", "")).strip(),
        "company":    str(x.get("company", "")).strip(),
        "period":     str(x.get("period", "")).strip(),
        "highlights": [str(h).strip() for h in (x.get("highlights") or []) if str(h).strip()][:3],
    }


def _norm_project(x):
    """Coerce one project entry into a stable {name, description} dict."""
    if not isinstance(x, dict):
        return {"name": str(x).strip(), "description": ""}
    return {"name": str(x.get("name", "")).strip(), "description": str(x.get("description", "")).strip()}


def build_profile(cv_text: str, claude_client, model: str) -> dict:
    """Parse a CV and return a structured profile dict.

    Parameters
    ----------
    cv_text : str
        Plain text extracted from the candidate's CV.
    claude_client :
        An initialised Anthropic client, or None (graceful degrade).
    model : str
        The Claude model ID to use (e.g. ``"claude-sonnet-4-6"``).

    Returns
    -------
    dict
        Keys: skills, titles_held, seniority, years, city, state, domains.
        All keys are always present (empty-valued on failure/degrade).
    """
    if claude_client is None:
        return dict(_EMPTY_PROFILE)

    context = f"CV TEXT:\n{cv_text}"
    raw = matching.reply_json(claude_client, model, context, _SYSTEM_PROMPT, max_tokens=1500)

    # Merge whatever the LLM returned onto our safe defaults so keys are always stable.
    profile = dict(_EMPTY_PROFILE)
    if isinstance(raw, dict):
        # Classification — default True (fail open) if the key is absent/non-bool.
        profile["is_cv"] = bool(raw.get("is_cv", True))
        conf = raw.get("cv_confidence")
        if isinstance(conf, (int, float)) and not isinstance(conf, bool):
            profile["cv_confidence"] = max(0.0, min(1.0, float(conf)))
        profile["name"] = str(raw.get("name", "")).strip()
        profile["location"]    = str(raw.get("location", "")).strip()
        profile["skills"]      = [str(s) for s in raw.get("skills", []) if s]
        profile["titles_held"] = [str(t) for t in raw.get("titles_held", []) if t]
        seniority = str(raw.get("seniority", "")).strip().lower()
        profile["seniority"]   = seniority if seniority in (
            "intern", "entry", "associate", "mid-senior", "director", "executive"
        ) else ""
        yrs = raw.get("years")
        profile["years"]   = int(yrs) if isinstance(yrs, (int, float)) and not isinstance(yrs, bool) else None
        profile["city"]    = str(raw.get("city", "")).strip()
        profile["state"]   = str(raw.get("state", "")).strip().upper()
        profile["domains"] = [str(d) for d in raw.get("domains", []) if d]
        we = raw.get("work_experience")
        profile["work_experience"] = [_norm_experience(x) for x in we][:8] if isinstance(we, list) else []
        pr = raw.get("projects")
        profile["projects"] = [_norm_project(x) for x in pr][:8] if isinstance(pr, list) else []

    return profile
