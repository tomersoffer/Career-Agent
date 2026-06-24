# -*- coding: utf-8 -*-
"""
cv/state_delta.py
Per-turn conversational state updates: turn a user message into a {set, add, remove}
delta and merge it into a persistent state object. Pure merge + one dependency-injected
LLM extraction helper. Reused by the interview (Flow A) and, later, job-search criteria.

No dataset/network import beyond the leaf `matching` util.
"""

import matching  # leaf util — safe (no dataset load)

# Field -> kind. "list" fields support add/remove; "scalar" fields support set/clear.
PROFILE_SCHEMA = {
    "titles_held": "list",
    "skills":      "list",
    "years":       "scalar",
    "seniority":   "scalar",
    "city":        "scalar",
    "state":       "scalar",
}

# Job-search criteria the user can amend across turns (Flow B).
SEARCH_SCHEMA = {
    "roles":     "list",
    "city":      "scalar",
    "state":     "scalar",
    "seniority": "scalar",
    "years":     "scalar",
    "remote":    "scalar",   # bool; honored by get_recommendations as a hard filter
}

_EMPTY_DELTA = {"set": {}, "add": {}, "remove": {}, "reply": "", "ready": False,
                "skip": False, "blocked": False}


def _norm_list(v):
    """Coerce to a deduped (case-insensitive), order-preserving list of trimmed strings."""
    if v is None:
        return []
    if not isinstance(v, list):
        v = [v]
    seen, out = set(), []
    for x in v:
        s = str(x).strip()
        k = s.lower()
        if s and k not in seen:
            seen.add(k)
            out.append(s)
    return out


def apply_delta(state, delta, schema):
    """Merge a {set, add, remove} delta into state. Pure; never raises.

    set     replaces a field (list coerced; scalar stored as-is, None clears).
    add     unions items into a list field (deduped, order-preserving).
    remove  drops matching items from a list field (case-insensitive), or clears a scalar.
    remove is applied AFTER set/add so "change to X and remove X" nets to removed.
    Fields not in `schema` are ignored.
    """
    out = dict(state) if isinstance(state, dict) else {}
    if not isinstance(delta, dict):
        return out

    for f, v in (delta.get("set") or {}).items():
        if f not in schema:
            continue
        out[f] = _norm_list(v) if schema[f] == "list" else (None if v in ("", None) else v)

    for f, v in (delta.get("add") or {}).items():
        if schema.get(f) != "list":
            continue
        out[f] = _norm_list(_norm_list(out.get(f)) + _norm_list(v))

    for f, v in (delta.get("remove") or {}).items():
        if f not in schema:
            continue
        if schema[f] == "list":
            drop = {s.lower() for s in _norm_list(v)}
            out[f] = [x for x in _norm_list(out.get(f)) if x.lower() not in drop]
        else:
            out[f] = None

    return out


def _coerce_delta(raw, schema):
    """Validate an LLM delta: keep only schema fields, coerce reply/ready. Never raises."""
    if not isinstance(raw, dict):
        return dict(_EMPTY_DELTA)

    def _d(key):
        v = raw.get(key)
        return v if isinstance(v, dict) else {}

    setd = {f: v for f, v in _d("set").items() if f in schema}
    addd = {f: v for f, v in _d("add").items() if schema.get(f) == "list"}
    remd = {f: v for f, v in _d("remove").items() if f in schema}
    return {"set": setd, "add": addd, "remove": remd,
            "reply": str(raw.get("reply", "")), "ready": bool(raw.get("ready", False)),
            "skip": bool(raw.get("skip", False)), "blocked": bool(raw.get("blocked", False))}


def _context(state, history, user_msg):
    lines = ["=== מצב נוכחי (state) ===", str(state),
             "", "=== היסטוריית שיחה ==="]
    for t in (history or [])[-8:]:
        who = "User" if t.get("role") == "user" else "Agent"
        lines.append(f"{who}: {str(t.get('content', ''))[:500]}")
    lines += ["", "=== הודעת המשתמש הנוכחית ===", user_msg or "(תור פתיחה)"]
    return "\n".join(lines)


def extract_delta(state, history, user_msg, schema, system_prompt, claude_client, model):
    """Call the LLM to produce a validated delta (+reply, +ready). Degrades to empty."""
    if claude_client is None or not (user_msg or "").strip():
        return dict(_EMPTY_DELTA)
    raw = matching.reply_json(claude_client, model, _context(state, history, user_msg),
                              system_prompt, max_tokens=500)
    return _coerce_delta(raw, schema)
