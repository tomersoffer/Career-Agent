# -*- coding: utf-8 -*-
# =====================================================================
# LIVE AGENTIC TOOL-CALLING EVALUATION  (model under test = LLM_BACKEND)
# ---------------------------------------------------------------------
# This is the agent_loop.py counterpart to eval_intent_live.py.
#
# eval_intent_live.py grades parse_query() — the OLD single-shot intent JSON
# ({mode,intent,state,...}) that still routes the SCRAPE tab. THIS harness grades
# the LIVE main-tab mechanism: agent_loop.run_agent_turn(), where the model is
# given the real OpenAI tool schemas (TOOLS) and ITSELF decides which tool to
# call, with what arguments, and may chain several before replying.
#
# We grade the agentic decision the app actually branches on:
#   * tool    — which tool the model CALLED first (the routing decision)
#   * mode    — the resulting pipeline mode the loop reports back to app.py
#   * state   — the 2-letter state ARGUMENT the model passed to the tool
#               (this is where the documented company-as-role / Hebrew-location
#                failures resurface — now they live in the tool arguments)
#   * metric  — for get_job_stat, the whitelisted analytics function chosen
#   * no_tool — for OFF-TOPIC prompts, the scope guard must fire: the model must
#               decline WITHOUT calling any tool (tool_trace empty)
#
# Ground truth follows AGENT_SYSTEM (agent_loop.py) verbatim. Smalltalk is graded
# on `mode` only (the model may answer a greeting via the small_talk tool OR
# directly — both surface as mode="chat"), so the grade is robust to either.
#
# REQUIRES LLM_BACKEND=openai — agent_loop uses the OpenAI tools API directly.
#
#   run:  python -X utf8 tests/eval_agent_loop_live.py
# =====================================================================
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_runner as ag
import agent_loop

# Each case: prompt + the fields we assert on (only the keys present are graded).
# `last_q` seeds a prior search so the paging case ("עוד") has something to page.
CASES = [
    # ---- clear job searches -> tool=search_jobs, mode=search ----
    {"q": "אני מחפש משרת data analyst בניו יורק",
     "tool": "search_jobs", "mode": "search", "state": "NY"},
    {"q": "find me nurse jobs in Texas",
     "tool": "search_jobs", "mode": "search", "state": "TX"},
    {"q": "data analyst",                       # bare role, no verb
     "tool": "search_jobs", "mode": "search"},
    # role implied (not named) + explicit find -> search:
    {"q": "אני מהנדס תוכנה עם 5 שנות ניסיון, תמצא לי משרות מתאימות",
     "tool": "search_jobs", "mode": "search"},
    # English search with city -> CA grounding in the tool args:
    {"q": "show me senior product manager roles in San Francisco",
     "tool": "search_jobs", "mode": "search", "state": "CA"},
    # typo'd English search -> still search, Chicago -> IL:
    {"q": "fnid me marketing jobz in chicago",
     "tool": "search_jobs", "mode": "search", "state": "IL"},
    # location-only search (no role) -> search, Boston -> MA:
    {"q": "מה יש לכם בבוסטון?",
     "tool": "search_jobs", "mode": "search", "state": "MA"},

    # ---- computed statistics over many jobs -> tool=get_job_stat, mode=data_question ----
    {"q": "כמה מרוויח data analyst בממוצע בניו יורק?",
     "tool": "get_job_stat", "mode": "data_question", "state": "NY", "metric": "salary_stats"},
    {"q": "how many remote QA jobs are there?",
     "tool": "get_job_stat", "mode": "data_question", "metric": "count_jobs"},
    # "most common skills" is explicitly a COMPUTED statistic:
    {"q": "מהם הכישורים הנפוצים ביותר למשרות data analyst?",
     "tool": "get_job_stat", "mode": "data_question", "metric": "top_skills"},
    # EXTREMUM statistic — graded on tool/mode only (metric is model's call):
    {"q": "איזה תפקיד משלם הכי הרבה?",
     "tool": "get_job_stat", "mode": "data_question"},

    # ---- open career guidance -> tool=give_career_advice, mode=advice ----
    {"q": "איך מתכוננים לראיון עבודה לתפקיד אנליסט נתונים?",
     "tool": "give_career_advice", "mode": "advice"},
    {"q": "אני שוקל לעבור מהוראה לתחום הדאטה, מה דעתך?",
     "tool": "give_career_advice", "mode": "advice"},
    # skills-to-learn -> advice per spec, even though a role is named:
    {"q": "כדאי לי ללמוד Python או SQL בשביל data analyst?",
     "tool": "give_career_advice", "mode": "advice"},
    # comparison / career-paths info -> advice (not a listing):
    {"q": "מה ההבדל בין data analyst ל-data scientist?",
     "tool": "give_career_advice", "mode": "advice"},
    # career-path advice naming two roles -> advice:
    {"q": "אני data analyst אבל רוצה להיות מנהל מוצר, מה כדאי לי לעשות?",
     "tool": "give_career_advice", "mode": "advice"},

    # ---- smalltalk / about-the-system -> mode=chat (tool optional, not asserted) ----
    {"q": "תודה רבה!", "mode": "chat"},
    {"q": "מה אתה יודע לעשות?", "mode": "chat"},
    # NEGATION + thanks trap: dominant act is "thanks" -> chat, NOT a search:
    {"q": "אני כבר לא מחפש עבודה, רק רציתי להגיד תודה רבה!", "mode": "chat"},

    # ---- paging the previous search -> tool=more_results, mode=more ----
    {"q": "עוד", "tool": "more_results", "mode": "more", "last_q": "data analyst"},

    # ---- NEW: profile-builder interview tool (no parse_query equivalent) ----
    # user has no CV and wants to build a profile together -> start_profile_interview:
    {"q": "אין לי קורות חיים, אפשר שנבנה פרופיל ביחד?",
     "tool": "start_profile_interview", "mode": "interview_kickoff"},
    # wants personalized matching but named no concrete role -> interview, not search:
    {"q": "אני רוצה שתמצא לי משרות שמתאימות לי",
     "tool": "start_profile_interview", "mode": "interview_kickoff"},

    # ---- NEW: OFF-TOPIC SCOPE GUARD -> must decline WITHOUT calling a tool ----
    {"q": "מה מזג האוויר היום בתל אביב?", "no_tool": True},
    {"q": "מי ניצח במונדיאל האחרון?", "no_tool": True},
    {"q": "תן לי מתכון לעוגת שוקולד", "no_tool": True},

    # ---- DOCUMENTED FAILURE (location identification), now in the tool args ----
    # "מתכנת בגוגל בקליפורניה" — the model should pass state="CA" to search_jobs.
    # On this Hebrew phrasing gpt-5.4-mini intermittently omits the state, so the
    # geo filter is silently dropped. Kept as a graded regression marker.
    {"q": "מתכנת בגוגל בקליפורניה",
     "tool": "search_jobs", "mode": "search", "state": "CA"},
]

GRADED = ("tool", "mode", "state", "metric", "no_tool")


def actual_fields(result, expected):
    """Extract from a run_agent_turn() result ONLY the fields the case grades.

    `tool`/`state`/`metric` read the FIRST tool the model called (its raw
    arguments, before grounding) — that is the routing decision under test.
    `no_tool` is True when the model called nothing (scope guard / direct reply).
    Shared with the stability harness."""
    trace = result.get("tool_trace") or []
    first = trace[0] if trace else None
    args = (first or {}).get("args") or {}
    out = {}
    if "tool" in expected:
        out["tool"] = first["tool"] if first else ""
    if "mode" in expected:
        out["mode"] = result.get("mode")
    if "state" in expected:
        out["state"] = str(args.get("state", "")).strip().upper()
    if "metric" in expected:
        out["metric"] = args.get("metric")
    if "no_tool" in expected:
        out["no_tool"] = (len(trace) == 0)
    return out


def _trace_str(result):
    """Compact, human-readable tool trace for the per-case printout."""
    trace = result.get("tool_trace") or []
    if not trace:
        return "(no tool called — direct reply)"
    steps = []
    for t in trace:
        a = {k: v for k, v in (t.get("args") or {}).items() if v not in ("", None)}
        steps.append(f"{t['tool']}({json.dumps(a, ensure_ascii=False)})")
    return "  ->  ".join(steps)


# ---------------------------------------------------------------------
# Diagnostic probes (NOT auto-graded) — the agent_loop analogue of the
# document's appendix א.2. Each prints the tool + raw args the model chose,
# for manual inspection of argument quality (company-as-role, foreign cities,
# English-vs-Hebrew location grounding).
# ---------------------------------------------------------------------
PROBES = [
    "Tesla jobs",                                   # company name as a role?
    "data analyst jobs in London",                  # foreign city -> no US state
    "משרות data analyst בתל אביב",                   # foreign city (Hebrew)
    "jobs in Portland",                             # ambiguous city -> OR?
    "I want to work at Google",                     # company, no role/location
    "software engineer at Google in California",    # English -> state=CA (contrast)
    "מתכנת בגוגל בקליפורניה",                        # Hebrew -> state=CA? (contrast)
    "what time is it?",                             # off-topic
]


def run():
    print("MODEL UNDER TEST:", ag.LLM_MODEL, "| backend:", ag.LLM_BACKEND)
    if ag.LLM_BACKEND != "openai":
        print("WARNING: agent_loop uses the OpenAI tools API; set LLM_BACKEND=openai "
              "(and OPENAI_API_KEY) or the loop cannot issue tool calls.")
    print("=" * 96)

    passed, failures = 0, []
    for i, c in enumerate(CASES, 1):
        result = agent_loop.run_agent_turn(c["q"], last_q=c.get("last_q", ""),
                                           offset=c.get("offset", 0))
        expected = {k: c[k] for k in GRADED if k in c}
        actual = actual_fields(result, expected)
        ok = (actual == expected)
        passed += ok
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] #{i:02d}  {c['q']}")
        print(f"        expected : {expected}")
        print(f"        actual   : {actual}")
        print(f"        trace    : {_trace_str(result)}")
        print(f"        reply    : {(result.get('reply') or result.get('intro') or '')[:160]}")
        if not ok:
            failures.append((i, c["q"], expected, actual))
        print("-" * 96)

    n = len(CASES)
    print(f"\nACCURACY: {passed}/{n} = {passed / n:.0%}")
    if failures:
        print(f"\nMISROUTING CASES ({len(failures)}) — the agent chose the wrong tool/args:")
        for i, q, exp, act in failures:
            diff = {k: f"expected {exp[k]!r} but got {act.get(k)!r}"
                    for k in exp if exp[k] != act.get(k)}
            print(f"  #{i} '{q}' -> {diff}")
    else:
        print("\nNo misrouting found in this set.")

    # ---- diagnostic probes (manual inspection of tool-argument quality) ----
    print("\n" + "=" * 96)
    print("DIAGNOSTIC PROBES (tool + raw args the model chose — inspect manually):")
    print("=" * 96)
    for q in PROBES:
        result = agent_loop.run_agent_turn(q)
        print(f"  '{q}'")
        print(f"       {_trace_str(result)}")
        print(f"       reply: {(result.get('reply') or result.get('intro') or '')[:140]}")
        print("-" * 96)

    return passed, n, failures


if __name__ == "__main__":
    run()
