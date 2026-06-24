# -*- coding: utf-8 -*-
# =====================================================================
# LIVE INTENT-CLASSIFICATION EVALUATION  (model under test = LLM_BACKEND)
# ---------------------------------------------------------------------
# Unlike the mocked unit tests (test_intent.py / test_router.py), this harness
# calls the REAL model through parse_query() and grades its routing decisions
# against ground-truth labels derived strictly from PARSE_SYSTEM's own spec.
# It reports per-case PASS/FAIL, overall accuracy, and isolates any case where
# the chat FAILED to identify the request (misrouting) — required evidence for
# the project's verification section.
#
#   run:  python -X utf8 tests/eval_intent_live.py
# =====================================================================
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_runner as ag

# Each case: prompt + the fields we assert on. We grade the high-level ROUTING
# decision the app actually branches on: `mode` and `intent` (and `state` when a
# location is named). Ground truth follows PARSE_SYSTEM verbatim.
CASES = [
    # ---- clear job searches -> mode=search, intent=job_search ----
    {"q": "אני מחפש משרת data analyst בניו יורק",
     "mode": "search", "intent": "job_search", "state": "NY"},
    {"q": "find me nurse jobs in Texas",
     "mode": "search", "intent": "job_search", "state": "TX"},
    {"q": "data analyst",          # bare role, no verb -> "when unsure choose search"
     "mode": "search", "intent": "job_search"},

    # ---- computed statistics over many jobs -> mode=data_question, intent=info ----
    # (info EVEN THOUGH a role/location is named — the hard boundary)
    {"q": "כמה מרוויח data analyst בממוצע בניו יורק?",
     "mode": "data_question", "intent": "info", "state": "NY"},
    {"q": "how many remote QA jobs are there?",
     "mode": "data_question", "intent": "info"},

    # ---- open career guidance -> mode=advice, intent=info ----
    {"q": "איך מתכוננים לראיון עבודה לתפקיד אנליסט נתונים?",
     "mode": "advice", "intent": "info"},
    # career-transition question: names a domain but it's guidance, not a listing
    {"q": "אני שוקל לעבור מהוראה לתחום הדאטה, מה דעתך?",
     "mode": "advice", "intent": "info"},

    # ---- smalltalk / about-the-system -> mode=smalltalk, intent=info ----
    {"q": "תודה רבה!",
     "mode": "smalltalk", "intent": "info"},
    {"q": "מה אתה יודע לעשות?",
     "mode": "smalltalk", "intent": "info"},

    # ---- paging the previous search -> wants_more=True, mode=more ----
    # graded on the behavior-relevant fields: app.py routes mode-first and bypasses
    # `intent` when mode=="more" (app.py:546,679), so intent is not asserted here.
    {"q": "עוד",
     "wants_more": True, "mode": "more"},

    # ---- ADVERSARIAL BOUNDARY PROBES (where small models tend to misroute) ----
    # skills-to-learn -> advice/info per spec ("skills to learn ... NOT asking to list jobs"),
    # even though a role is named:
    {"q": "כדאי לי ללמוד Python או SQL בשביל data analyst?",
     "mode": "advice", "intent": "info"},
    # comparison / career-paths info -> advice/info (not a job listing):
    {"q": "מה ההבדל בין data analyst ל-data scientist?",
     "mode": "advice", "intent": "info"},
    # "most common skills" is explicitly a COMPUTED statistic -> data_question/info per spec:
    {"q": "מהם הכישורים הנפוצים ביותר למשרות data analyst?",
     "mode": "data_question", "intent": "info"},
    # role implied (not named) + explicit find -> search/job_search:
    {"q": "אני מהנדס תוכנה עם 5 שנות ניסיון, תמצא לי משרות מתאימות",
     "mode": "search", "intent": "job_search"},
    # English search with city -> state grounding:
    {"q": "show me senior product manager roles in San Francisco",
     "mode": "search", "intent": "job_search", "state": "CA"},

    # ---- HARD TRAPS (round 2) ----
    # NEGATION + thanks: the dominant act is "thanks" -> smalltalk/info, is_search false,
    # despite the literal substring "מחפש עבודה" (classic negation trap):
    {"q": "אני כבר לא מחפש עבודה, רק רציתי להגיד תודה רבה!",
     "mode": "smalltalk", "intent": "info"},
    # EXTREMUM statistic ("highest paying") -> a computed stat over many jobs = data_question/info:
    {"q": "איזה תפקיד משלם הכי הרבה?",
     "mode": "data_question", "intent": "info"},
    # CAREER-PATH advice naming two roles -> advice/info (not a listing):
    {"q": "אני data analyst אבל רוצה להיות מנהל מוצר, מה כדאי לי לעשות?",
     "mode": "advice", "intent": "info"},
    # typo'd English search -> still search/job_search, city Chicago -> IL:
    {"q": "fnid me marketing jobz in chicago",
     "mode": "search", "intent": "job_search", "state": "IL"},
    # location-only search (no role) -> search/job_search, Boston -> MA:
    {"q": "מה יש לכם בבוסטון?",
     "mode": "search", "intent": "job_search", "state": "MA"},

    # ---- DOCUMENTED FAILURE (location identification) ----
    # "מתכנת בגוגל בקליפורניה" = "programmer at Google in California". CA exists in the data,
    # and the English phrasing resolves to state='CA' correctly — but on this Hebrew phrasing
    # gpt-5.4-mini puts "California" in `city` and returns state='' , so the app silently drops
    # the location filter and searches the whole USA. This is the required "chat fails to
    # identify the request" case; kept as a graded regression marker.
    {"q": "מתכנת בגוגל בקליפורניה",
     "mode": "search", "intent": "job_search", "state": "CA"},
]

GRADED = ("mode", "intent", "state", "wants_more")


def run():
    print("MODEL UNDER TEST:", ag.LLM_MODEL, "| backend:", ag.LLM_BACKEND)
    print("=" * 90)
    passed, failures = 0, []
    for i, c in enumerate(CASES, 1):
        got = ag.parse_query(c["q"])
        expected = {k: c[k] for k in GRADED if k in c}
        actual = {k: got.get(k) for k in expected}
        ok = (actual == expected)
        passed += ok
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] #{i:02d}  {c['q']}")
        print(f"        expected: {expected}")
        print(f"        actual  : {actual}")
        if not ok:
            failures.append((i, c["q"], expected, actual))
        print("-" * 90)

    n = len(CASES)
    print(f"\nACCURACY: {passed}/{n} = {passed / n:.0%}")
    if failures:
        print(f"\nMISROUTING CASES ({len(failures)}) — chat failed to identify the request:")
        for i, q, exp, act in failures:
            diff = {k: f"expected {exp[k]!r} but got {act[k]!r}"
                    for k in exp if exp[k] != act[k]}
            print(f"  #{i} '{q}' -> {diff}")
    else:
        print("\nNo misrouting found in this set.")
    return passed, n, failures


if __name__ == "__main__":
    run()
