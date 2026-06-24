# -*- coding: utf-8 -*-
"""
cv/gap.py
Job-specific gap analysis: which of THE SELECTED JOB's requirements the CV already
evidences (covered) and which are genuinely missing (missing).

Why this exists
---------------
The deterministic token matcher in cv/jd_keywords.py (`must_haves`) flags a
requirement as missing unless *every* significant token literally appears in the CV.
That produces noisy gaps: it can't tell that "Power BI" already covers a "BI tools"
requirement, that "PostgreSQL / MySQL" covers "SQL", or that a bullet describing
A/B tests covers an "experimentation" requirement. The agent then asks about
"gaps" that aren't real.

`analyze_gaps` does a single, focused LLM pass that compares the job's stated
requirements against the WHOLE CV semantically (synonyms, abbreviations, evidence
in experience/projects — not just the skills line), so the tailoring interview only
ever targets the candidate's REAL gaps for that specific job.

Dependency-injected: claude_client and model are passed in (never imported from
agent_runner). Falls back to the deterministic matcher when the LLM is disabled or
returns nothing usable, so callers always get a valid {covered, missing} result.
"""

import matching                 # leaf util — the shared Claude JSON helper
from cv import jd_keywords as _jdkw   # deterministic fallback


# System prompt — the model is a precise ATS gap analyst, NOT a writer. It only
# classifies the job's OWN requirements against the CV; it never invents requirements
# and never edits the CV.
_GAP_SYSTEM_PROMPT = """\
אתה אנליסט ATS מדויק. בהינתן דרישות של משרה ספציפית וקורות-חיים של מועמד, סווג כל דרישה
של המשרה לאחת משתי קבוצות, **בלי להמציא דרישות ובלי לכתוב מחדש את הקו"ח**:

- "covered": דרישה שיש לה **עדות אמיתית** בקו"ח — בכל מקום בו (כישורים, ניסיון, פרויקטים,
  השכלה). כלול גם התאמות **שקולות סמנטית**: ראשי-תיבות, מילים נרדפות, וכלי-על מול כלי ספציפי
  (למשל "Power BI"/"Tableau" מכסים "BI tools"; "PostgreSQL"/"MySQL" מכסים "SQL"; תיאור של
  A/B testing מכסה "experimentation").
- "missing": דרישת-מפתח של המשרה שאין לה **שום** עדות בקו"ח.

כללים:
- השתמש **אך ורק** בדרישות שנאמרות בפועל במשרה (שדה הכישורים + דרישות מתוך התיאור). אל תוסיף דרישות משלך.
- נסח כל דרישה בקצרה (1-4 מילים), בלשון המשרה.
- בספק — העדף "covered" (אל תטריד את המשתמש על מה שכנראה כבר קיים). סווג כ-"missing" רק כשהיעדר ברור.
- עד 8 פריטים בכל רשימה, החשובים-ביותר-תחילה.

## פורמט החזרה (חובה!)
החזר ONLY JSON תקין — שום טקסט מחוץ לאובייקט:
{"covered": ["<דרישה>", ...], "missing": ["<דרישה>", ...]}"""


def _norm_list(value, cap):
    """Coerce an LLM field into a clean, deduped, capped list of non-empty strings."""
    if not isinstance(value, list):
        return []
    out, seen = [], set()
    for item in value:
        s = str(item).strip()
        key = s.lower()
        if s and key not in seen:
            seen.add(key)
            out.append(s)
    return out[:cap]


def analyze_gaps(cv_text: str, job: dict, claude_client, model, top_k: int = 8) -> dict:
    """Semantic, job-specific gap analysis between a CV and the selected job.

    Parameters
    ----------
    cv_text : str
        Full plain-text CV.
    job : dict
        The selected job; uses 'title', 'company', 'skills', 'description'.
    claude_client :
        Initialised Anthropic client, or None (LLM-disabled -> deterministic fallback).
    model : str
        Claude model ID.
    top_k : int
        Cap per list.

    Returns
    -------
    dict
        ``{"covered": [...], "missing": [...]}`` — requirement phrases, importance
        order. Always valid; falls back to jd_keywords.must_haves on any failure.
    """
    if not job:
        return {"covered": [], "missing": []}

    # Deterministic baseline — used as the fallback and never worse than before.
    fallback = _jdkw.must_haves(cv_text, job, top_k=top_k)

    if claude_client is None:
        return fallback

    skills = job.get("skills", "")
    if isinstance(skills, list):
        skills = ", ".join(str(s) for s in skills)

    context = "\n".join([
        "=== פרטי המשרה ===",
        f"תפקיד: {job.get('title', '')}",
        f"חברה: {job.get('company', '')}" if job.get("company") else "",
        f"כישורים נדרשים: {skills}" if skills else "",
        f"תיאור המשרה: {job.get('description', '')}" if job.get("description") else "",
        "",
        "=== קורות החיים של המועמד ===",
        cv_text or "(ריק)",
    ])

    raw = matching.reply_json(claude_client, model, context, _GAP_SYSTEM_PROMPT, max_tokens=700)

    if not isinstance(raw, dict) or ("covered" not in raw and "missing" not in raw):
        return fallback   # LLM disabled / unparseable -> deterministic baseline

    return {
        "covered": _norm_list(raw.get("covered"), top_k),
        "missing": _norm_list(raw.get("missing"), top_k),
    }
