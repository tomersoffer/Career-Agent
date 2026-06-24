# -*- coding: utf-8 -*-
"""
analytics.py — deterministic, whitelisted aggregation engine over the gold table.

Every function takes (df, params, ctx) and returns a small fact dict that ALWAYS
includes "function" and "n". Numbers are computed in pandas — never by the LLM.
Dependency-injected: the caller passes the dataframe + a Ctx (fitted vectorizer,
title matrix aligned row-for-row with df, valid states, seniority map). Pure: no
module-level dataset load, so unit tests use a tiny synthetic frame.
"""

from collections import namedtuple
import json as _json

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import matching   # shared JSON-mode Claude helpers (leaf util, no dataset load)

Ctx = namedtuple("Ctx", ["vectorizer", "title_matrix", "valid_states", "seniority_rank"])

ROLE_COS_THRESHOLD = 0.30   # a row's title must clear this cosine to the role text to count


def _dump(obj):
    return _json.dumps(obj, ensure_ascii=False)


def _role_mask(df, role, ctx):
    """Boolean mask of rows whose TITLE is cosine-similar to `role` (reuses fitted TF-IDF)."""
    role = (role or "").strip().lower()
    if not role:
        return np.ones(len(df), dtype=bool)
    q = ctx.vectorizer.transform([role])
    sims = cosine_similarity(q, ctx.title_matrix).ravel()
    return sims >= ROLE_COS_THRESHOLD


def _filter(df, params, ctx):
    """Apply the optional, composable filters shared by every function. Returns a sub-frame."""
    mask = _role_mask(df, params.get("role"), ctx)

    state = (params.get("state") or "").strip().upper()
    if state and state in ctx.valid_states:
        mask &= (df["job_state"].values == state)

    exp = (params.get("experience") or "").strip().lower()
    if exp:
        rank = ctx.seniority_rank.get(exp)
        if rank is not None:
            mask &= (df["experience_rank"].values == rank)

    remote = params.get("remote")
    if isinstance(remote, bool):
        mask &= (df["remote_allowed"].values == (1 if remote else 0))

    return df[mask]


def salary_stats(df, params, ctx):
    sub = _filter(df, params, ctx)
    sal = sub["salary"].dropna()
    sal = sal[sal > 0]
    if len(sal) == 0:
        return {"function": "salary_stats", "n": 0,
                "median": None, "p25": None, "p75": None}
    return {
        "function": "salary_stats",
        "n": int(len(sal)),
        "median": int(round(sal.median())),
        "p25": int(round(sal.quantile(0.25))),
        "p75": int(round(sal.quantile(0.75))),
    }


def _value_counts(series, n):
    """Return [[value, count], ...] for the top n non-blank values."""
    vc = series.dropna().astype(str).str.strip()
    vc = vc[vc != ""].value_counts().head(n)
    return [[k, int(v)] for k, v in vc.items()]


def top_skills(df, params, ctx, n=5):
    sub = _filter(df, params, ctx)
    exploded = (sub["skills"].dropna().astype(str)
                .str.split(",").explode().str.strip().str.lower())
    exploded = exploded[exploded != ""]
    vc = exploded.value_counts().head(n)
    return {"function": "top_skills", "n": int(len(sub)),
            "items": [[k, int(v)] for k, v in vc.items()]}


def count_jobs(df, params, ctx):
    sub = _filter(df, params, ctx)
    return {"function": "count_jobs", "n": int(len(sub)), "count": int(len(sub))}


def remote_share(df, params, ctx):
    sub = _filter(df, params, ctx)
    # Count only jobs with a KNOWN remote status (drop NaN) — consistent with how _filter
    # treats `remote`: NaN rows aren't asserted to be on-site, so they don't deflate the share.
    known = sub["remote_allowed"].dropna()
    if len(known) == 0:
        return {"function": "remote_share", "n": 0, "share_pct": None}
    share = float(known.mean())
    return {"function": "remote_share", "n": int(len(known)),
            "share_pct": int(round(share * 100))}


def experience_breakdown(df, params, ctx):
    sub = _filter(df, params, ctx)
    return {"function": "experience_breakdown", "n": int(len(sub)),
            "items": _value_counts(sub["experience_level"], 7)}


def top_locations(df, params, ctx, n=5):
    sub = _filter(df, params, ctx)
    return {"function": "top_locations", "n": int(len(sub)),
            "items": _value_counts(sub["job_state"], n)}


def top_industries(df, params, ctx, n=5):
    sub = _filter(df, params, ctx)
    return {"function": "top_industries", "n": int(len(sub)),
            "items": _value_counts(sub["job_industry"], n)}


# ---------------------------------------------------------------------------
# answer_data_question orchestrator: map -> compute -> narrate
# ---------------------------------------------------------------------------

# The whitelist: function name -> callable. The LLM may ONLY pick from these keys.
FUNCTIONS = {
    "salary_stats": salary_stats,
    "top_skills": top_skills,
    "count_jobs": count_jobs,
    "remote_share": remote_share,
    "experience_breakdown": experience_breakdown,
    "top_locations": top_locations,
    "top_industries": top_industries,
}

_MAP_SYSTEM = (
    "You translate a user's question about a US jobs dataset into ONE analytics call. "
    "Respond with ONLY a JSON object: {\"function\": <name>, \"params\": {...}}. "
    "function MUST be exactly one of: " + ", ".join(FUNCTIONS) + ". "
    "params keys (all optional): \"role\" (job title in English, lowercase), \"state\" (2-letter "
    "US code), \"experience\" (one of intern/entry/associate/mid-senior/director/executive), "
    "\"remote\" (true/false). Omit a param if the user didn't constrain it. No prose."
)

_NARRATE_SYSTEM = (
    "You are a career data assistant. You are given a user question and a COMPUTED FACT (JSON) "
    "from our jobs database. Answer the question in ONE short sentence IN HEBREW, in PLAIN, SIMPLE, "
    "EASY-TO-READ language (everyday words, no flowery filler), using ONLY the "
    "numbers in the fact — never invent figures. Salaries are USD. If n is 0, say honestly that "
    "there were no matching postings. If n is small (under 5), note the answer is based on only n "
    "postings. At most one emoji. No preamble."
)

_FAIL_REPLY = "מצטער, לא הצלחתי לחשב את זה כרגע. נסה לנסח אחרת או לציין תפקיד/מיקום."


def answer_data_question(question, df, ctx, claude_client, model):
    """Map the question to one whitelisted function, compute it in pandas, narrate in Hebrew.
    Never raises; returns a Hebrew string."""
    if claude_client is None:
        return _FAIL_REPLY
    try:
        plan = matching.reply_json(claude_client, model, "QUESTION: " + question,
                                   _MAP_SYSTEM, max_tokens=200)
        fn = FUNCTIONS.get(str(plan.get("function", "")).strip())
        if fn is None:
            return _FAIL_REPLY
        params = plan.get("params") if isinstance(plan.get("params"), dict) else {}
        fact = fn(df, params, ctx)
        context = "QUESTION: " + question + "\n\nCOMPUTED FACT:\n" + _dump(fact)
        reply = matching.reply(claude_client, model, context, _NARRATE_SYSTEM, max_tokens=200)
    except Exception as exc:           # any map/compute/narrate error -> graceful, logged
        print("[analytics] failed:", exc)
        return _FAIL_REPLY
    return reply or _FAIL_REPLY
