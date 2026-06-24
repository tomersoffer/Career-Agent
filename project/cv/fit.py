# -*- coding: utf-8 -*-
"""
cv/fit.py
Deterministic fit scoring and skill-gap analysis.

Pure ML — no Claude required.
"""

import re

from sklearn.metrics.pairwise import cosine_similarity

from . import role_fit   # leaf util — for the shared candidate_role_text()

# Words so generic they are useless for gap analysis.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "to", "for", "with",
    "on", "at", "by", "is", "are", "be", "as", "it", "its", "we",
    "you", "this", "that", "have", "has", "had", "not", "but", "from",
    "will", "can", "may", "must", "would", "could", "should",
    "experience", "work", "working", "years", "year", "strong",
    "ability", "skills", "skill", "knowledge", "understanding",
    "environment", "team", "teams", "role", "roles", "job", "position",
    "required", "preferred", "plus", "bonus", "including", "using",
    "ability", "excellent", "good", "great", "proven",
})


def _tokenize(text: str) -> set:
    """Lowercase, split on non-alphanumeric separators, drop stopwords & short tokens."""
    tokens = re.split(r"[,\s/|;()\[\]]+", text.lower())
    return {t for t in tokens if len(t) > 1 and t not in _STOPWORDS}


def fit_and_gap(profile: dict, cv_text: str, job: dict, vectorizer) -> dict:
    """Compute a cosine fit score and a skill-gap breakdown for one job.

    Parameters
    ----------
    profile : dict
        Candidate profile from ``cv/profile.py`` (keys: skills, …).
    cv_text : str
        Full plain text of the CV.
    job : dict
        Job posting dict with at least ``"title"``; optionally ``"skills"``
        (comma-separated string or list) and ``"description"`` (str).
    vectorizer :
        Already-fitted ``TfidfVectorizer`` (same vocab as the rest of the app).

    Returns
    -------
    dict
        ``{"fit_score": float, "missing": list[str], "present": list[str]}``

        * ``fit_score`` — cosine similarity in [0, 1] between the CV vector
          and the job-title vector (same scale as ``matching.title_scores``).
        * ``missing``  — job keywords / skills absent from the CV.
        * ``present``  — job keywords / skills found in the CV.
    """
    # ---- 1. Fit score: cosine(candidate ROLE, job title) ----
    # Role-to-role (held titles + skills), NOT the whole CV — the full document dilutes the
    # cosine against a short title and scores ~uniformly low. Fall back to the CV if empty.
    job_title = str(job.get("title", "")).lower() or "unknown"
    role_text = role_fit.candidate_role_text(profile) or (cv_text or "").lower()
    cv_vec    = vectorizer.transform([role_text])
    job_vec   = vectorizer.transform([job_title])
    fit_score = float(cosine_similarity(cv_vec, job_vec)[0, 0])

    # ---- 2. Skill-gap: (job keywords) vs (CV term set + profile skills) ----
    # Build the job keyword set from skills field + description
    job_skills_raw = job.get("skills", "")
    if isinstance(job_skills_raw, list):
        job_skills_raw = ", ".join(str(s) for s in job_skills_raw)
    job_description = str(job.get("description", ""))

    job_keywords = _tokenize(job_skills_raw + " " + job_description)

    # CV terms: tokens from the raw text + normalised profile skills
    cv_terms = _tokenize(cv_text)
    for skill in profile.get("skills", []):
        cv_terms.update(_tokenize(skill))

    present = sorted(kw for kw in job_keywords if kw in cv_terms)
    missing = sorted(kw for kw in job_keywords if kw not in cv_terms)

    return {"fit_score": fit_score, "missing": missing, "present": present}
