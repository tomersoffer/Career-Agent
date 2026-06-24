# -*- coding: utf-8 -*-
"""
cv/converse.py
Approach-C conversation controller. Each turn:
  1. Claude extracts the role(s)/location the user wants (handles Hebrew + English).
  2. Python computes the role-fit + seniority-gap FACTS (cv/role_fit.py).
  3. Claude turns the facts into a warm Hebrew reply + a structured action.

Dependency-injection: claude_client/model/vectorizer are passed in — never
imported from agent_runner. Degrades gracefully when claude_client is None.
"""

import json

import matching                # leaf util — safe (no dataset load)
from . import role_fit

# Honest, non-looping fallbacks. The old degrade reply was a question ("which roles?")
# that invited an answer it then ignored — when the LLM is off it just repeated, ignoring
# the user. These tell the truth instead of looping.
_DEGRADE_REPLY = "השירות אינו זמין כעת (ה-AI מושבת). ודא שמפתח ה-API מוגדר ונסה שוב."
_EMPTY_REPLY = "לא הצלחתי לנסח תשובה כרגע — נסה שוב או נסח מחדש."

_EXTRACT_SYSTEM = (
    "You route a Hebrew/English career chat. Read the conversation and the candidate "
    "profile, then return ONLY a JSON object with EXACTLY these keys:\n"
    '  "roles"    : list[str]  — job titles (in English) the user wants searched now; '
    "[] if none settled yet. If the user merely confirmed (e.g. 'כן','yes','בדיוק') the "
    "role the agent proposed, return THAT role.\n"
    '  "location" : str        — a preferred location the user mentioned, else "".\n'
    "Output JSON only, no prose."
)

_REPLY_SYSTEM = (
    "You are a warm, honest, and PROACTIVE career agent who replies in HEBREW. You are "
    "given the candidate PROFILE, computed ROLE FACTS, and the CONVERSATION. Your job is "
    "to actively DRIVE the candidate through this funnel — never just answer passively:\n"
    "  1) confirm ALL the role(s) in their profile (titles_held may hold several — name each, "
    "and NEVER silently drop or demote a role the user gave to a mere 'want to add?');\n"
    "  2) invite them to add or change roles in free text;\n"
    "  3) once role(s) are settled, MOVE to searching;\n"
    "  4) the results will be ranked against their profile — tell them that.\n"
    "  NOTE: if the profile has NO work_experience (it was built from a short interview, not a real "
    "CV), do NOT claim the user 'has a background/רקע/ניסיון' in a role — speak of the roles they WANT "
    "to target.\n"
    "Use the facts: role_fit is 0-1 (how well the CV content matches the role); "
    "seniority_gap.band is 'stretch' (aiming above the CV), 'fit', or 'under'. Rules:\n"
    "- BE VERY SHORT: at most 1-2 short sentences total. No paragraphs, no lists, no "
    "preamble. The user's professor requires brief answers.\n"
    "- Be direct and in Hebrew, no emoji. ALWAYS end with the clear next step "
    "(e.g. 'נצא לחפש?', 'רוצה להוסיף עוד תפקיד?') so the user knows what to do.\n"
    "- For a 'stretch' role, note the gap in ONE short clause — do not lecture.\n"
    "- Keep momentum: if the user has given even one usable role, lean toward 'search' "
    "rather than chatting further. Don't stall the funnel. ALWAYS DEFER to the user — "
    "never block a role they insist on.\n"
    "- When you return action 'search', END the reply by briefly offering to tailor the CV to "
    "one of the results (e.g. 'רוצה שאתאים את הקו\"ח לאחת מהן? כתוב לי איזו').\n"
    "- Choose an action: 'search' when the user has settled on role(s) to search now; "
    "'tailor' when the user wants to improve/adapt their CV for a SPECIFIC job already shown "
    "(put their reference to it — a rank number like '1' or a title — in tailor_query); "
    "'data_question' when the user asks for a COMPUTED statistic over many jobs (average salary, "
    "counts, most-common skills/industries/locations, remote share) — put their question in "
    "data_query; 'advice' when the user asks for open career guidance (interview prep, CV tips, "
    "skills to learn, salary negotiation) rather than listing jobs; 'request_cv' when an updated "
    "CV would clearly help and the user seems open to it; otherwise 'talk'.\n"
    "Return ONLY JSON with keys: "
    '"reply" (str, Hebrew), "action" '
    '("talk"|"search"|"tailor"|"data_question"|"advice"|"request_cv"), '
    '"roles" (list[str] English titles to search), "data_query" (str), "tailor_query" (str), '
    '"location" (str). No prose.'
)


def _format_history(history: list) -> str:
    lines = []
    for turn in (history or [])[-8:]:
        who = "User" if turn.get("role") == "user" else "Agent"
        lines.append(f"{who}: {turn.get('content', '')}")
    return "\n".join(lines)


def _extract_roles(history, profile, claude_client, model):
    context = (
        "CANDIDATE PROFILE:\n" + json.dumps(profile, ensure_ascii=False) +
        "\n\nCONVERSATION:\n" + _format_history(history)
    )
    raw = matching.reply_json(claude_client, model, context, _EXTRACT_SYSTEM, max_tokens=300)
    roles = [str(r).strip() for r in (raw.get("roles") or []) if str(r).strip()]
    location = str(raw.get("location", "")).strip()
    return roles, location


def respond(history, profile, cv_text, vectorizer, claude_client, model) -> dict:
    """Run one conversation turn. Returns
    {reply: str, action: 'talk'|'search'|'request_cv', roles: list[str], location: str}."""
    if claude_client is None:
        return {"reply": _DEGRADE_REPLY, "action": "talk", "roles": [],
                "tailor_query": "", "data_query": "", "location": ""}

    roles, location = _extract_roles(history, profile, claude_client, model)
    facts = role_fit.assess(profile or {}, cv_text or "", roles, vectorizer)

    context = (
        "CANDIDATE PROFILE:\n" + json.dumps(profile, ensure_ascii=False) +
        "\n\nROLE FACTS (computed):\n" + json.dumps(facts, ensure_ascii=False) +
        (("\n\nPREFERRED LOCATION: " + location) if location else "") +
        "\n\nCONVERSATION:\n" + _format_history(history)
    )
    raw = matching.reply_json(claude_client, model, context, _REPLY_SYSTEM, max_tokens=700)

    action = str(raw.get("action", "talk")).strip().lower()
    if action not in ("talk", "search", "tailor", "data_question", "advice", "request_cv"):
        action = "talk"
    reply = str(raw.get("reply", "")).strip() or _EMPTY_REPLY
    out_roles = [str(r).strip() for r in (raw.get("roles") or roles) if str(r).strip()]
    out_location = str(raw.get("location", location)).strip()
    tailor_query = str(raw.get("tailor_query", "")).strip()
    data_query = str(raw.get("data_query", "")).strip()
    return {"reply": reply, "action": action, "roles": out_roles,
            "tailor_query": tailor_query, "data_query": data_query, "location": out_location}
