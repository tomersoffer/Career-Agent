# -*- coding: utf-8 -*-
"""
Captures end-to-end traces of the NEW agentic tool-calling loop (agent_loop.py),
and re-runs the labeled battery as a TOOL-SELECTION accuracy test (since which
tool the model calls is now the real routing decision). Writes:
    output/traces_agent.json     — rich end-to-end traces (tool_calls + results + reply)
    output/tool_eval.json        — per-case tool-selection accuracy
All values come from live runs against gpt-5.4-mini.
"""
import os, sys, json
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "project"))
sys.path.insert(0, os.path.join(_ROOT, "project", "tests"))

import agent_loop as al
from eval_intent_live import CASES   # reuse the 21 labeled prompts

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUTDIR, exist_ok=True)

# mode (from the labeled battery) -> the tool we expect the agent to call
MODE_TO_TOOL = {"search": "search_jobs", "data_question": "get_job_stat",
                "advice": "give_career_advice", "smalltalk": "small_talk",
                "more": "more_results"}


def slim_trace(tr):
    """Compact a tool_trace entry: keep tool + args, and a small result summary."""
    out = []
    for e in tr:
        res = e.get("result") or {}
        summary = {}
        if "matches" in res:
            summary = {"matches_n": len(res["matches"]),
                       "top": [m.get("title") + " @ " + str(m.get("company")) for m in res["matches"][:3]]}
        elif "fact" in res:
            summary = {"fact": res["fact"]}
        elif "capabilities" in res:
            summary = {"capabilities": res["capabilities"][:80] + "…"}
        elif "on_screen_jobs" in res:
            summary = {"on_screen_jobs": len(res["on_screen_jobs"])}
        else:
            summary = res
        out.append({"tool": e["tool"], "args": e["args"], "result": summary})
    return out


# ---------------- 1) rich end-to-end traces ----------------
RICH = [
    ("search", "אני מחפש משרת data analyst בניו יורק", {}),
    ("data_question", "מה השכר הטיפוסי ל-data analyst בניו יורק?", {}),
    ("advice", "כדאי לי ללמוד Python או SQL בשביל data analyst?", {}),
    ("smalltalk", "מה אתה יודע לעשות?", {}),
    ("more", "עוד", {"last_q": "data analyst בניו יורק", "offset": 3}),
    ("multi_step", "תמצא לי משרות data analyst בטקסס, וגם כמה הן משלמות בממוצע", {}),
    ("failure_probe", "מתכנת בגוגל בקליפורניה", {}),     # the OLD parse_query failure — now resolved
    ("failure_company", "Tesla jobs", {}),               # genuine remaining failure: company-as-role
]
rich = []
for label, prompt, kw in RICH:
    r = al.run_agent_turn(prompt, **kw)
    rich.append({"label": label, "prompt": prompt,
                 "tool_trace": slim_trace(r["tool_trace"]),
                 "matches_n": len(r["matches"]),
                 "mode": r["mode"], "state": r["state"], "reply": r["reply"]})
    print("[rich] %-14s tools=%s" % (label, [t["tool"] for t in r["tool_trace"]]))

with open(os.path.join(OUTDIR, "traces_agent.json"), "w", encoding="utf-8") as f:
    json.dump(rich, f, ensure_ascii=False, indent=2)

# ---------------- 2) tool-selection accuracy over the 21-case battery ----------------
rows, passed = [], 0
for i, c in enumerate(CASES, 1):
    mode = c.get("mode") or ("more" if c.get("wants_more") else "search")
    expected = MODE_TO_TOOL.get(mode, "search_jobs")
    kw = {"last_q": "data analyst בניו יורק", "offset": 3} if mode == "more" else {}
    r = al.run_agent_turn(c["q"], **kw)
    tools = [t["tool"] for t in r["tool_trace"]]
    ok = expected in tools
    passed += ok
    rows.append({"i": i, "q": c["q"], "expected_tool": expected,
                 "tools_called": tools, "ok": ok})
    print("[eval] #%02d %-12s exp=%-18s got=%s %s" % (i, mode, expected, tools, "OK" if ok else "MISS"))

result = {"n": len(CASES), "passed": passed, "rows": rows}
with open(os.path.join(OUTDIR, "tool_eval.json"), "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print("\nTOOL-SELECTION ACCURACY: %d/%d = %.0f%%" % (passed, len(CASES), 100 * passed / len(CASES)))
print("Saved traces_agent.json + tool_eval.json")
