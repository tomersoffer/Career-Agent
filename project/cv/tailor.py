# -*- coding: utf-8 -*-
"""
cv/tailor.py
One collaborative-editor turn: user ↔ CV-tailoring assistant (Hebrew output).

Dependency-injected: claude_client and model are passed in — never imported
from agent_runner.

Flow — section-walk (one question per turn), driven by _infer_tailor_stage:
  SECTION_WALK (per CV section) → ask ONE focused question to extract a specific fact
                                  for the current section, no CV edit (proposed_cv=null)
  COMPOSE (all sections gathered, or user says "סיימתי"/"מספיק"/"תודה")
                                → emit the FULL CV from the gathered facts
  AMEND   (a rewrite already exists) → a focused, structure-preserving edit

The agent is the master tailor: it decides what each section needs and only extracts
specific factual information — it never asks the user for strategy/preferences.

Each turn is STAGE-SCOPED: the system prompt is built per stage via
prompts.tailor_system_prompt(stage) — the model sees only the current stage's
instructions — and the per-turn context carries only the grounding that stage needs
(the current section + mode during SECTION_WALK; CV structure only at COMPOSE).
"""

import re as _re
import sys

import matching       # leaf util — safe
from cv import prompts  # single runtime source of truth for the CV agent's rules

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------------------------------------------------------------------------
# System prompt — sourced from cv/prompts.py (single runtime source of truth).
# Per turn we build a STAGE-SCOPED prompt via prompts.tailor_system_prompt(stage)
# (static preamble + only the current stage's instructions). _SYSTEM_PROMPT below
# is the full monolithic fallback; do NOT inline rule text here — edit cv/prompts.py.
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = prompts.TAILOR_SYSTEM_PROMPT


def tailor_turn(
    cv_text: str,
    job: dict,
    history: list,
    user_msg: str,
    claude_client,
    model: str,
    composed: bool = False,
    has_cv: bool = False,
    section_index: int = 0,
) -> dict:
    """Run one collaborative-editor turn for CV tailoring.

    Parameters
    ----------
    cv_text : str
        Current plain-text CV (the textarea content — source of truth).
    job : dict
        The job being targeted; expected keys: ``title``, ``company`` (opt),
        ``description`` (opt), ``skills`` (opt).
    history : list[dict]
        Previous turns as ``[{"role": "user"|"assistant", "content": str}, …]``.
    user_msg : str
        The new user message for this turn.  Empty string = opener (proactive first turn).
    claude_client :
        Initialised Anthropic client, or None (LLM-disabled degrade).
    model : str
        Claude model ID.

    Returns
    -------
    dict
        ``{"reply": str, "proposed_cv": str | None, "done": bool}``
    """
    sections = prompts.TAILOR_SECTIONS
    total = len(sections)

    if claude_client is None:
        return {"reply": "השירות אינו זמין כעת (LLM מושבת). נסה שוב מאוחר יותר.",
                "proposed_cv": None, "done": False,
                "section_done": False, "next_section_index": section_index, "total_sections": total}

    # Post-rewrite: the user signalled they're satisfied -> exit without re-composing.
    if composed and _is_done(user_msg):
        return {"reply": "סיימתי — שמרתי את הגרסה המעודכנת.", "proposed_cv": None, "done": True,
                "section_done": False, "next_section_index": section_index, "total_sections": total}

    # ------------------------------------------------------------------
    # Ground the model with deterministic JD keyword analysis (no hallucination)
    # ------------------------------------------------------------------
    from cv import jd_keywords as jdkw

    mh = jdkw.must_haves(cv_text, job, top_k=8)
    covered = mh.get("covered", [])
    missing = mh.get("missing", [])

    # ------------------------------------------------------------------
    # Job fields
    # ------------------------------------------------------------------
    job_title   = job.get("title", "")
    job_company = job.get("company", "")
    job_desc    = job.get("description", "")
    job_skills  = job.get("skills", "")
    if isinstance(job_skills, list):
        job_skills = ", ".join(str(s) for s in job_skills)

    is_opener = not bool(user_msg) and not bool(history)

    # ------------------------------------------------------------------
    # Deterministic stage selection — Python owns the section pointer:
    # SECTION_WALK (gather per section) -> COMPOSE (full CV) ; AMEND for post-rewrite edits.
    # ------------------------------------------------------------------
    si = max(0, min(int(section_index or 0), total))   # clamp
    stage, section_key = _infer_tailor_stage(history, user_msg, is_opener, sections, si, composed)
    system_prompt = prompts.tailor_system_prompt(stage)

    context_lines = ["=== השלב הנוכחי: " + stage + " ==="]
    if stage == "SECTION_WALK":
        section_label = next((s["label_he"] for s in sections if s["key"] == section_key), section_key)
        context_lines += ["מצב: " + ("enhance" if has_cv else "build"),
                          "סעיף נוכחי: " + section_label]
        # the next section, so the agent can chain straight into it when this one is done
        if si + 1 < len(sections):
            context_lines.append("הסעיף הבא: " + sections[si + 1]["label_he"])
        else:
            context_lines.append("הסעיף הבא: אין — זהו הסעיף האחרון; אחריו ייכתב הקו\"ח המלא.")

    # Job details — every stage (keeps the question / rewrite on-target).
    context_lines += ["", "=== פרטי המשרה ===", f"תפקיד: {job_title}"]
    if job_company:
        context_lines.append(f"חברה: {job_company}")
    if job_skills:
        context_lines.append(f"כישורים נדרשים מהמשרה: {job_skills}")
    if job_desc:
        context_lines.append(f"תיאור המשרה: {job_desc}")

    # CV body — always (the stub in build mode carries role/skills; the full CV in enhance).
    context_lines += ["", "=== קורות חיים נוכחיים ===", cv_text]

    # Covered/missing requirements — JD-aware for the walk and the rewrite.
    context_lines += [
        "",
        "=== דרישות מכוסות (שלב מילה-במילה; אל תטען לכיסוי של דרישה שאינה כאן ואינה בקו\"ח) ===",
        ", ".join(covered) or "אין",
    ]
    if stage == "SECTION_WALK":
        context_lines.append("דרישות חסרות (להכוונת השאלה): " + (", ".join(missing) or "אין"))

    # Conversation history — to acknowledge the last answer / collect confirmed facts.
    if history:
        context_lines.append("")
        context_lines.append("=== היסטוריית שיחה ===")
        for turn in history:
            role_label = "משתמש" if turn.get("role") == "user" else "עוזר"
            context_lines.append(f"{role_label}: {turn.get('content', '')}")

    context_lines += [
        "",
        "=== הודעת המשתמש הנוכחית ===",
        user_msg if user_msg else "(תור פתיחה — פתח בשאלה על הסעיף הנוכחי)",
    ]

    context = "\n".join(context_lines)

    raw = matching.reply_json(claude_client, model, context, system_prompt, max_tokens=2400)

    if isinstance(raw, dict):
        section_done = bool(raw.get("section_done", False))
        next_idx = si + 1 if (stage == "SECTION_WALK" and section_done) else si
        return {
            "reply":              str(raw.get("reply", "")),
            "proposed_cv":        raw.get("proposed_cv") or None,
            "done":               bool(raw.get("done", False)),
            "section_done":       section_done,
            "next_section_index": next_idx,
            "total_sections":     total,
        }

    # Fallback if JSON parsing failed
    return {"reply": str(raw) if raw else "שגיאה בעיבוד התגובה.", "proposed_cv": None,
            "done": False, "section_done": False, "next_section_index": si, "total_sections": total}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Closing signals: the user is ready to stop the interview and get the rewrite.
_DONE_WORDS = prompts.DONE_WORDS


def _is_done(user_msg: str) -> bool:
    """True when the user's message is a closing signal (move to the final rewrite)."""
    txt = (user_msg or "").lower()
    for w in _DONE_WORDS:
        # Use word-boundary matching so short tokens like "די" don't fire inside longer words.
        if _re.search(r'(?<!\w)' + _re.escape(w) + r'(?!\w)', txt):
            return True
    return False


def _infer_tailor_stage(history: list, user_msg: str, is_opener: bool,
                        sections: list, section_index: int, composed: bool = False) -> tuple:
    """Decide the tailoring stage and return (stage, section_key).

    SECTION_WALK — gather the section at ``section_index`` (one short Q, no edit).
    COMPOSE      — all sections gathered (index past the last) OR the user signals done.
    AMEND        — a rewrite already exists (``composed``): a focused edit of the current
                   CV (tailor_turn handles the composed+closing exit before this is reached).
    """
    if composed and not is_opener:
        return ("AMEND", "")
    if _is_done(user_msg):
        return ("COMPOSE", "")
    if 0 <= section_index < len(sections):
        return ("SECTION_WALK", sections[section_index]["key"])
    return ("COMPOSE", "")


def _compute_gap(cv_text: str, job: dict) -> dict:
    """Legacy helper kept for backwards compatibility (prefer jd_keywords now).

    Note: jd_keywords.coverage() is now the primary gap analysis in tailor_turn.
    This helper is retained so any external code that imported it continues to work.
    """
    from cv.fit import _tokenize  # reuse tokeniser
    job_skills_raw = job.get("skills", "")
    if isinstance(job_skills_raw, list):
        job_skills_raw = ", ".join(str(s) for s in job_skills_raw)
    job_description = str(job.get("description", ""))
    job_keywords = _tokenize(job_skills_raw + " " + job_description)
    cv_terms = _tokenize(cv_text)
    present = sorted(kw for kw in job_keywords if kw in cv_terms)
    missing = sorted(kw for kw in job_keywords if kw not in cv_terms)
    return {"fit_score": 0.0, "missing": missing, "present": present}
