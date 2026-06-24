# -*- coding: utf-8 -*-
"""
cv/jd_keywords.py
Deterministic extraction of key required terms from a job posting and
checking which ones a CV already covers.

Pure Python — no Claude calls, no network I/O.
"""

import re

from cv.fit import _tokenize


def extract(job: dict, top_k: int = 12) -> list:
    """Top required terms from the job, ranked by frequency.

    skills field is weighted 3x, title 2x, description 1x.
    Terms are deduped and stopword-filtered (via _tokenize).

    Parameters
    ----------
    job : dict
        Job posting with optional keys 'skills' (str or list),
        'title' (str), 'description' (str).
    top_k : int
        Maximum number of terms to return.

    Returns
    -------
    list[str]
        Most important terms first, length <= top_k.
        Returns [] on None/missing/blank job.
    """
    if not job:
        return []

    # Normalise skills (may be list or comma string)
    skills_raw = job.get("skills", "") or ""
    if isinstance(skills_raw, list):
        skills_raw = ", ".join(str(s) for s in skills_raw)

    title_raw = str(job.get("title", "") or "")
    description_raw = str(job.get("description", "") or "")

    # Tokenise each field separately so we can apply weights per field
    skills_tokens = _tokenize(skills_raw)
    title_tokens = _tokenize(title_raw)
    description_tokens = _tokenize(description_raw)

    # Accumulate weighted scores
    scores: dict = {}

    for token in skills_tokens:
        scores[token] = scores.get(token, 0) + 3

    for token in title_tokens:
        scores[token] = scores.get(token, 0) + 2

    for token in description_tokens:
        scores[token] = scores.get(token, 0) + 1

    if not scores:
        return []

    # Sort: descending score, then ascending alphabetical for ties
    ranked = sorted(scores.keys(), key=lambda t: (-scores[t], t))

    return ranked[:top_k]


def coverage(cv_text: str, job: dict, top_k: int = 12) -> dict:
    """Split the job's top terms into those the CV covers vs those it lacks.

    Parameters
    ----------
    cv_text : str
        Full plain text of the candidate's CV.
    job : dict
        Job posting dict (see extract()).
    top_k : int
        Passed through to extract().

    Returns
    -------
    dict
        {"covered": list[str], "missing": list[str]} — importance order
        preserved. A term counts as covered if it appears in the tokenised
        CV text.
        Returns {"covered": [], "missing": []} on None/blank job.
    """
    terms = extract(job, top_k=top_k)
    if not terms:
        return {"covered": [], "missing": []}

    cv_tokens = _tokenize(cv_text or "")

    covered = [t for t in terms if t in cv_tokens]
    missing = [t for t in terms if t not in cv_tokens]

    return {"covered": covered, "missing": missing}


# ---------------------------------------------------------------------------
# must_haves() — phrase-aware requirement grounding for the tailoring flow
# ---------------------------------------------------------------------------

# Words that mark a JD clause as a hard requirement (English + Hebrew).
_CUE_WORDS = frozenset({
    "required", "require", "requirement", "requirements", "must", "essential",
    "proficient", "proficiency", "strong", "minimum", "mandatory", "expertise",
    "נדרש", "נדרשת", "חובה", "דרישה", "דרישות",
})


def _skill_phrases(job: dict) -> list:
    """The job's skills as WHOLE phrases (list or comma string), blanks dropped."""
    raw = job.get("skills", "") or ""
    if isinstance(raw, list):
        items = [str(s).strip() for s in raw]
    else:
        items = [p.strip() for p in str(raw).split(",")]
    return [p for p in items if p]


def _phrase_covered(phrase: str, cv_tokens: set) -> bool:
    """A phrase counts as covered when ALL of its significant tokens are in the CV.
    A phrase with no significant tokens (pure stopwords) is treated as covered so it
    never pollutes the 'missing' list."""
    ptoks = _tokenize(phrase)
    if not ptoks:
        return True
    return ptoks.issubset(cv_tokens)


def _cue_terms(job: dict) -> list:
    """Single terms drawn from description clauses that contain a cue word.
    Splits the description into clauses on sentence/line/bullet boundaries; keeps the
    significant tokens of any clause mentioning a requirement cue. Deduped, cue words
    themselves excluded."""
    desc = str(job.get("description", "") or "")
    if not desc:
        return []
    out, seen = [], set()
    for clause in re.split(r"[.\n;•·\-–|]+", desc):
        low = clause.lower()
        if not any(cue in low for cue in _CUE_WORDS):
            continue
        for tok in _tokenize(clause):
            if tok in _CUE_WORDS or tok in seen:
                continue
            seen.add(tok)
            out.append(tok)
    return out


def must_haves(cv_text: str, job: dict, top_k: int = 8) -> dict:
    """Phrase-aware requirement targets for the tailoring interview.

    Returns ``{"covered": [...], "missing": [...]}`` — skills phrases first
    (most important), then cue-context description terms; deduped
    case-insensitively; each list capped at ``top_k``. Safe on None/empty job.
    """
    if not job:
        return {"covered": [], "missing": []}

    cv_tokens = _tokenize(cv_text or "")

    targets, seen = [], set()
    def _add(t):
        key = t.strip().lower()
        if key and key not in seen:
            seen.add(key)
            targets.append(t.strip())

    for ph in _skill_phrases(job):     # primary source
        _add(ph)
    for t in _cue_terms(job):          # secondary source
        _add(t)

    covered, missing = [], []
    for t in targets:
        (covered if _phrase_covered(t, cv_tokens) else missing).append(t)

    return {"covered": covered[:top_k], "missing": missing[:top_k]}
