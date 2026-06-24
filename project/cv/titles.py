# -*- coding: utf-8 -*-
"""
cv/titles.py
Deterministic title suggestion via TF-IDF cosine similarity.

No Claude required — this is the graded ML core.
"""

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def suggest_titles(
    cv_text: str,
    vectorizer,
    title_matrix,
    titles,
    top_k: int = 8,
) -> list:
    """Return the top-k real job titles most similar to the CV text.

    Parameters
    ----------
    cv_text : str
        Plain text of the candidate's CV.
    vectorizer :
        An already-fitted ``TfidfVectorizer`` (from ``agent_runner``).
    title_matrix :
        Sparse matrix (n_jobs × vocab) — the pre-vectorised title column.
        Rows are aligned to ``titles``.
    titles : array-like of str
        The job title string for each row in ``title_matrix``.
    top_k : int
        How many distinct titles to return (default 8).

    Returns
    -------
    list[str]
        Up to ``top_k`` distinct titles, highest-cosine first.
        Deduplication is case-insensitive; the best-scoring casing is kept.
    """
    cv_vec = vectorizer.transform([cv_text.lower()])
    sims = cosine_similarity(cv_vec, title_matrix).flatten()  # shape: (n_jobs,)

    # Argsort descending
    order = np.argsort(sims)[::-1]

    seen_lower: dict[str, float] = {}   # lower-cased title -> best score
    best_casing: dict[str, str] = {}    # lower-cased title -> original casing with best score

    for idx in order:
        title = str(titles[idx])
        score = float(sims[idx])
        key = title.lower()
        if key not in seen_lower or score > seen_lower[key]:
            seen_lower[key] = score
            best_casing[key] = title
        if len(seen_lower) >= top_k * 5:   # early-stop: gathered enough candidates
            break

    # Sort deduped candidates by score descending and take top_k
    ranked = sorted(seen_lower.keys(), key=lambda k: seen_lower[k], reverse=True)
    return [best_casing[k] for k in ranked[:top_k]]


def clean_titles(raw_titles: list, claude_client, model: str) -> list:
    """Normalise raw scraped job titles into clean, canonical role names via Claude.

    e.g. 'Big Data Engineer (Java+Python+AWS)' -> 'Big Data Engineer';
         'Data Engineer-SQL/ Python/ Data Analysis' -> 'Data Engineer'.
    De-duplicates and preserves relevance order. Returns ``raw_titles`` unchanged when
    there is no Claude client (graceful degrade).
    """
    if claude_client is None or not raw_titles:
        return list(raw_titles)

    import matching  # leaf util — safe

    context = "RAW TITLES:\n" + "\n".join(f"- {t}" for t in raw_titles)
    system = (
        "Normalise each raw job title into a clean, canonical role name: strip parenthetical "
        "tech lists, slashes, and stack/seniority noise (e.g. 'Big Data Engineer (Java+Python+AWS)' "
        "-> 'Big Data Engineer'; 'Data Engineer-SQL/ Python/ Data Analysis' -> 'Data Engineer'). "
        "Title-case, de-duplicate, keep order by relevance. Output ONLY a JSON object: "
        '{"titles": ["Data Engineer", ...]}. No prose.'
    )
    result = matching.reply_json(claude_client, model, context, system, max_tokens=300)
    titles = result.get("titles") if isinstance(result, dict) else None
    return [str(t) for t in titles if t] if titles else list(raw_titles)
