# -*- coding: utf-8 -*-
"""
cv/interview.py
Build a structured candidate profile by asking 4 clarifying questions, for users
who have no CV. Each answer is parsed into the SAME profile-field shape that
cv/profile.py produces; on completion we synthesize a small cv_text so the built
profile flows through the existing CV-ranked search pipeline unchanged.

Dependency-injection: claude_client and model are passed in — never imported
from agent_runner. Degrades gracefully when claude_client is None.
"""

import sys

import matching  # leaf util — safe to import (no dataset load)
from cv import state_delta as _sd
from cv import prompts as _prompts

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Ordered interview script. Each entry is one vertical-stepper node + one field group.
STEPS = [
    {"key": "role",       "label_he": "תפקיד",
     "question_he": "איזה תפקיד אתה מחפש? (אפשר לציין כמה)"},
    {"key": "experience", "label_he": "ניסיון",
     "question_he": "כמה שנות ניסיון יש לך, ובאיזו רמה?"},
    {"key": "location",   "label_he": "מיקום",
     "question_he": "באיזה מיקום לחפש? (עיר או מדינה)"},
    {"key": "skills",     "label_he": "כישורים",
     "question_he": "אילו כלים, טכנולוגיות ומתודולוגיות אתה שולט בהם? (למשל Python, SQL, Selenium, Jira)"},
]

_VALID_SENIORITY = ("intern", "entry", "associate", "mid-senior", "director", "executive")

# Appended to every NON-role step: the user often volunteers a role while answering a
# different question (e.g. "add product too" when asked about experience). Capture it so
# it is never lost — the caller merges add_titles into titles_held additively.
_ADD_TITLES_NOTE = (
    ' Also, if the user names, adds, or changes any desired job ROLE/title here — even though '
    'this question is about something else (e.g. "add product too", "also data") — list those '
    'roles in "add_titles": list[str] (English, lowercase, with close synonyms); [] if none.'
)

_SYSTEMS = {
    "role": (
        "Extract the roles the user wants from their answer (Hebrew or English). Respond with ONLY a "
        'JSON object: {"titles_held": list[str]} — EVERY distinct role or domain the user names, in '
        "ENGLISH, lowercase, in the user's order of priority. Capture ALL of them — NEVER drop a role the "
        "user mentioned. For a broad domain, include its common close roles for recall (e.g. 'data' -> "
        "'data analyst','data scientist','data engineer'; 'product' -> 'product manager','product analyst') "
        "— but do NOT narrow a broad domain down to a single arbitrary seniority. [] if none. No prose."
    ),
    "experience": (
        "Extract experience from the user's answer (Hebrew or English). Respond with ONLY a JSON "
        'object: {"years": int|null, "seniority": str, "add_titles": list[str]}. "years" = total years '
        'of professional experience, or null. "seniority" = one of "intern","entry","associate",'
        '"mid-senior","director","executive", or "".' + _ADD_TITLES_NOTE + " No prose."
    ),
    "location": (
        "Extract a location from the user's answer (Hebrew or English). Respond with ONLY a JSON "
        'object: {"city": str, "state": str, "add_titles": list[str]}. "city" = the city in English '
        '("" if none). "state" = the 2-letter US state code if the city is in the US, else "".'
        + _ADD_TITLES_NOTE + " No prose."
    ),
    "skills": (
        "Extract skills from the user's answer (Hebrew or English). Respond with ONLY a JSON "
        'object: {"skills": list[str], "add_titles": list[str]} — "skills" = technical and soft skills, '
        "lowercase, [] if none." + _ADD_TITLES_NOTE + " No prose."
    ),
}


def _empty(step_key):
    return {
        "role":       {"titles_held": []},
        "experience": {"years": None, "seniority": ""},
        "location":   {"city": "", "state": ""},
        "skills":     {"skills": []},
    }[step_key]


def _dedupe_lower(items):
    seen, out = set(), []
    for x in items or []:
        s = str(x).strip().lower()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _fallback(step_key, answer):
    """LLM unavailable / non-JSON: store the raw answer in the natural field."""
    answer = answer.strip()
    if step_key == "role":
        return {"titles_held": [answer.lower()] if answer else []}
    if step_key == "skills":
        parts = [p for p in (s.strip() for s in answer.replace(";", ",").split(",")) if p]
        return {"skills": _dedupe_lower(parts)}
    if step_key == "location":
        return {"city": answer, "state": ""}
    return _empty(step_key)   # experience can't be safely guessed -> leave empty


def _coerce(step_key, raw):
    if step_key == "role":
        return {"titles_held": _dedupe_lower(raw.get("titles_held"))}
    if step_key == "experience":
        yrs = raw.get("years")
        years = int(yrs) if isinstance(yrs, (int, float)) and not isinstance(yrs, bool) else None
        sen = str(raw.get("seniority", "")).strip().lower()
        return {"years": years, "seniority": sen if sen in _VALID_SENIORITY else ""}
    if step_key == "location":
        return {"city": str(raw.get("city", "")).strip(),
                "state": str(raw.get("state", "")).strip().upper()}
    if step_key == "skills":
        return {"skills": _dedupe_lower(raw.get("skills"))[:16]}
    return _empty(step_key)


def parse_answer(step_key, answer, claude_client, model):
    """Parse one interview answer into that step's profile field(s). Never raises.

    Besides the step's own field, captures any role the user volunteers out of step
    in ``add_titles`` (present only when non-empty) so the caller can merge it into
    ``titles_held`` additively — the interview is not rigidly one-field-per-turn.
    """
    answer = (answer or "").strip()
    if not answer:
        return _empty(step_key)
    raw = matching.reply_json(claude_client, model, "ANSWER: " + answer,
                              _SYSTEMS[step_key], max_tokens=300)
    if not isinstance(raw, dict) or not raw:
        return _fallback(step_key, answer)
    fields = _coerce(step_key, raw)
    extra = _dedupe_lower(raw.get("add_titles"))
    if extra:
        fields["add_titles"] = extra
    return fields


# Hebrew seniority labels, mirrored from the frontend's SENIORITY_HE.
_SENIORITY_HE = {"intern": "מתמחה", "entry": "ג'וניור", "associate": "מיומן",
                 "mid-senior": "בכיר", "director": "ניהול", "executive": "הנהלה"}


def captured_label(step_index, profile):
    """A short value to show on the just-finished vertical-stepper node ('—' if empty)."""
    key = STEPS[step_index]["key"]
    if key == "role":
        titles = profile.get("titles_held") or []
        return titles[0] if titles else "—"
    if key == "experience":
        bits = []
        if profile.get("years") is not None:
            bits.append(f"{profile['years']} שנ׳")
        if profile.get("seniority"):
            bits.append(_SENIORITY_HE.get(profile["seniority"], profile["seniority"]))
        return " · ".join(bits) if bits else "—"
    if key == "location":
        return profile.get("city") or profile.get("state") or "—"
    if key == "skills":
        skills = profile.get("skills") or []
        return ", ".join(skills[:3]) if skills else "—"
    return "—"


# Mirrors cv.profile._EMPTY_PROFILE (the source of truth for the profile shape) —
# kept local to keep this a pure leaf module; finalize() drops any unknown keys.
_BLANK_PROFILE = {
    "name": "", "location": "", "skills": [], "titles_held": [], "seniority": "",
    "years": None, "city": "", "state": "", "domains": [],
    "work_experience": [], "projects": [],
}


def finalize(profile_so_far):
    """Return a profile with the FULL key set (same shape as cv.profile), filling gaps."""
    out = dict(_BLANK_PROFILE)
    for k, v in (profile_so_far or {}).items():
        if k in out:
            out[k] = v
    return out


def synthesize_cv_text(profile):
    """Build a small plain-text 'CV' from the gathered answers so the existing
    cv_fit / role_fit / search pipeline has real text to score against."""
    lines = []
    titles = profile.get("titles_held") or []
    if titles:
        lines.append("Target role: " + ", ".join(titles))
    exp = []
    if profile.get("years") is not None:
        exp.append(f"{profile['years']} years")
    if profile.get("seniority"):
        exp.append(profile["seniority"])
    if exp:
        lines.append("Experience: " + " ".join(exp))
    loc = ", ".join(b for b in [profile.get("city"), profile.get("state")] if b)
    if loc:
        lines.append("Location: " + loc)
    skills = profile.get("skills") or []
    if skills:
        lines.append("Skills: " + ", ".join(skills))
    return "\n".join(lines)


def seed_from_query(role_en, synonyms, city, state, years, seniority):
    """Map parsed-query fields into partial profile fields for a pre-filled interview.

    Returns (profile, seeded_keys, captured) where seeded_keys is the subset of
    ["role","experience","location"] whose field(s) are non-empty, and captured is
    {step_index: short_label} for those seeded steps. 'skills' is never seeded here.
    """
    titles = _dedupe_lower(([role_en] if role_en else []) + list(synonyms or []))
    sen = str(seniority or "").strip().lower()
    sen = sen if sen in _VALID_SENIORITY else ""
    yrs = int(years) if isinstance(years, (int, float)) and not isinstance(years, bool) else None

    profile = {
        "titles_held": titles,
        "years": yrs,
        "seniority": sen,
        "city": str(city or "").strip(),
        "state": str(state or "").strip().upper(),
    }

    seeded, captured = [], {}
    # step index -> key is fixed by STEPS order: 0 role, 1 experience, 2 location, 3 skills
    if titles:
        seeded.append("role")
        captured[0] = captured_label(0, profile)
    if yrs is not None or sen:
        seeded.append("experience")
        captured[1] = captured_label(1, profile)
    if profile["city"] or profile["state"]:
        seeded.append("location")
        captured[2] = captured_label(2, profile)
    return profile, seeded, captured


def next_unseeded_step(step_index, seeded_keys):
    """Smallest step index > step_index whose key is NOT in seeded_keys, else len(STEPS)."""
    seeded = set(seeded_keys or [])
    i = step_index + 1
    while i < len(STEPS) and STEPS[i]["key"] in seeded:
        i += 1
    return i


def first_unseeded_step(seeded_keys):
    """First step index whose key is not seeded (or len(STEPS) if all seeded)."""
    return next_unseeded_step(-1, seeded_keys)


# ---------------------------------------------------------------------------
# Conversational interview turn (Flow A) — stateful + amendable.
# Replaces the rigid one-field-per-step walk: any field can be added, changed,
# removed, or skipped at any time; role is the only hard requirement to finish.
# ---------------------------------------------------------------------------

# Which profile field(s) back each ordered stepper key, and the delta label for it.
_FIELD_FOR_STEP = {"role": "titles_held", "experience": "years",
                   "location": "city", "skills": "skills"}


def _field_filled(profile, key):
    if key == "role":       return bool(profile.get("titles_held"))
    if key == "experience": return profile.get("years") is not None or bool(profile.get("seniority"))
    if key == "location":   return bool(profile.get("city") or profile.get("state"))
    if key == "skills":     return bool(profile.get("skills"))
    return False


def _next_field(profile, skipped):
    sk = set(skipped or [])
    for s in STEPS:
        if s["key"] not in sk and not _field_filled(profile, s["key"]):
            return s["key"]
    return None


def _question_for(key):
    return next((s["question_he"] for s in STEPS if s["key"] == key), "")


def _known_summary(profile):
    out = {}
    if profile.get("titles_held"):
        out["תפקידים"] = ", ".join(profile["titles_held"])
    if profile.get("years") is not None:
        out["ניסיון"] = profile["years"]
    if profile.get("seniority"):
        out["רמה"] = profile["seniority"]
    if profile.get("city") or profile.get("state"):
        out["מיקום"] = profile.get("city") or profile.get("state")
    if profile.get("skills"):
        out["כישורים"] = ", ".join(profile["skills"])
    return out


def interview_turn(profile, history, user_msg, claude_client, model, skipped=None, us_only=False):
    """One conversational interview turn: merge any add/change/remove the user states into
    the profile, then ask the next missing field or finish. Builds the FULL profile —
    role, experience, location AND skills — before finishing (a field the user genuinely
    has nothing for can be skipped). Degrades to a deterministic next-question when the LLM
    is off.

    Returns ``{"profile", "reply", "done", "next_field", "skipped", "cv_text"}``.
    """
    profile = dict(profile or {})
    skipped = list(skipped or [])

    if claude_client is None:
        nf = _next_field(profile, skipped)
        done = nf is None
        return {"profile": profile, "reply": _question_for(nf) if nf else "בוא נצא לחפש.",
                "done": done, "next_field": nf, "skipped": skipped,
                "cv_text": synthesize_cv_text(finalize(profile)) if done else None}

    nf_asked = _next_field(profile, skipped)   # the field being answered this turn
    nf_label = _prompts.PROFILE_LABELS.get(_FIELD_FOR_STEP.get(nf_asked, ""), nf_asked) if nf_asked else None
    extra = _prompts.interview_extra_rules(nf_asked, nf_label, _known_summary(profile), us_only=us_only)
    system = _prompts.delta_system(_prompts.PROFILE_LABELS, extra)

    d = _sd.extract_delta(profile, history, user_msg, _sd.PROFILE_SCHEMA, system, claude_client, model)
    profile = _sd.apply_delta(profile, d, _sd.PROFILE_SCHEMA)

    # The user has nothing for the current field -> mark it addressed so we don't loop.
    if d.get("skip") and nf_asked and nf_asked not in skipped:
        skipped = skipped + [nf_asked]

    nf = _next_field(profile, skipped)
    done = nf is None                          # finish ONLY when all four fields are addressed
    # On the finishing turn, don't reuse the model's reply (it may still be asking a
    # question) — confirm the profile is complete and move to search.
    reply = "מעולה — יש לנו פרופיל מלא. נצא לחפש?" if done else (d.get("reply") or _question_for(nf))
    return {"profile": profile, "reply": reply, "done": done, "next_field": nf,
            "skipped": skipped, "cv_text": synthesize_cv_text(finalize(profile)) if done else None}
