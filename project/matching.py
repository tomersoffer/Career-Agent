# =====================================================================
# SHARED MATCHING CORE - one ranking algorithm used by BOTH searches
# (gold dataset tab + scraped tab). The user prompt can filter by any
# combination of THREE independent dimensions: TITLE, LOCATION, EXPERIENCE.
# Each dimension the user specifies becomes an AND filter; dimensions they
# don't mention impose no constraint. Only the location-scoring strategy
# and the result/reply shaping differ per tab.
# =====================================================================
import json

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---- the single home for every tunable constant (no more drifting copies) ----
TITLE_FLOOR = 0.30      # a title is "relevant" at/above this cosine
LOC_FLOOR = 0.25        # a location is "relevant" at/above this score
EXP_FLOOR = 0.80        # experience within ~1 rank of the requested level (1 - |d|/6)
SYNONYM_WEIGHT = 0.9    # synonym matches count slightly less than the exact role term
W_TITLE, W_LOC, W_EXP = 0.5, 0.3, 0.2   # ranking weights for the hybrid score


def title_scores(intent, vectorizer, title_matrix):
    """Weighted max-over-variants cosine of (role + synonyms) vs a title matrix.
    Each variant is scored as its own phrase (precision) and the best wins (recall);
    the exact role gets full weight, synonyms a small discount.
    Returns (scores[n], role_specified)."""
    variants = [v for v in [intent.get("role_en", "")] + intent.get("synonyms", []) if v]
    if not variants:
        return np.zeros(title_matrix.shape[0]), False
    weights = np.array([1.0] + [SYNONYM_WEIGHT] * (len(variants) - 1))
    sims = cosine_similarity(vectorizer.transform([v.lower() for v in variants]), title_matrix)
    return (sims * weights[:, None]).max(axis=0), True


def experience_scores(ranks, want_rank):
    """Proximity of each job's experience_rank to the requested rank.
    Returns (scores[n], exp_specified)."""
    ranks = np.asarray(ranks, dtype=float)
    if want_rank and want_rank > 0:
        return np.clip(1.0 - np.abs(ranks - want_rank) / 6.0, 0.0, 1.0), True
    return np.zeros(len(ranks)), False


def select(title_score, role_specified,
           loc_score, loc_specified,
           exp_score, exp_specified,
           ranks=None, top_n=5):
    """Apply the per-dimension AND gate over whatever the user specified, then rank the
    survivors by the weighted hybrid. Returns (ordered_positions, hybrid_scores)."""
    n = len(title_score)
    title_ok = (title_score >= TITLE_FLOOR) if role_specified else np.ones(n, dtype=bool)
    loc_ok = (loc_score >= LOC_FLOOR) if loc_specified else np.ones(n, dtype=bool)
    if exp_specified:
        exp_ok = exp_score >= EXP_FLOOR
        if ranks is not None:                       # unknown experience (rank 0) gets the benefit of the doubt
            exp_ok = exp_ok | (np.asarray(ranks) == 0)
    else:
        exp_ok = np.ones(n, dtype=bool)

    keep = np.where(title_ok & loc_ok & exp_ok)[0]
    hybrid = W_TITLE * title_score + W_LOC * loc_score + W_EXP * exp_score
    if len(keep) == 0:
        return np.array([], dtype=int), hybrid
    order = keep[np.argsort(hybrid[keep])[::-1]][:top_n]
    return order, hybrid


def reply(claude_client, model, context, system_prompt, max_tokens=2000):
    """Shared Claude call: same prompt-caching + extraction for both tabs."""
    if claude_client is None:
        return "[LLM DISABLED]\n\n" + context
    msg = claude_client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": context}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


def reply_json(claude_client, model, context, system_prompt, max_tokens=900):
    """Same Claude call as reply(), but the model is asked for a JSON object; we extract and
    parse it. Returns {} when the LLM is disabled or the output isn't valid JSON."""
    if claude_client is None:
        return {}
    txt = reply(claude_client, model, context, system_prompt, max_tokens)
    try:
        start, end = txt.find("{"), txt.rfind("}")
        return json.loads(txt[start:end + 1])
    except Exception:
        return {}
