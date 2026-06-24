# -*- coding: utf-8 -*-
# =====================================================================
# FULL-CONVERSATION MODEL COMPARISON  (real session memory, real endpoints)
# ---------------------------------------------------------------------
# Instead of grading standalone prompts, this harness plays the ROLE OF THE
# BROWSER: it holds one session (cv_text / profile / history / lastQuery / offset
# / searchCriteria / lastMatches) and threads it through the SAME Flask endpoints
# the front-end calls (driven via app.test_client(), no network). So every turn
# runs through the REAL routing in app.py — Flow-B refine -> CV-conversation
# (_cv_turn) -> agentic tool loop -> interview-kickoff -> tailoring — with genuine
# cross-turn memory, not isolated calls.
#
# One ~22-turn conversation exercises every aspect of the agent:
#   scope guard (cold + mid-context), intent ID (smalltalk/search/data/advice),
#   no-CV profile interview, CV upload, CV-ranked search, paging, and the
#   CV->job tailoring section-walk + compose.
#
# Each model drives the WHOLE conversation; per turn we score a behavioural
# rubric. The no-CV agentic turns are OpenAI-native (agent_loop uses the OpenAI
# tools API); for Claude we transparently route them through a native Anthropic
# tool loop (run_anthropic_turn) so all three are apples-to-apples.
#
#   run:  python -X utf8 tests/compare_conversation.py [K]    (default K=3)
# =====================================================================
import os, sys, json, time, statistics, io
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests"))

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, "..", "local.env"))
os.environ.setdefault("LLM_BACKEND", "openai")   # agent_loop / app import need an OpenAI key present

import agent_runner as ag
import agent_loop as al
import app as webapp                              # Flask app (loads the KB once, already in ag)

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
ANTH_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
K = int(sys.argv[1]) if len(sys.argv) > 1 else 3

MODELS = [
    {"name": "Claude Sonnet 4.6", "backend": "anthropic", "model": "claude-sonnet-4-6",
     "price_in": 3.00, "price_out": 15.00},
    {"name": "GPT-4.1-mini", "backend": "openai", "model": "gpt-4.1-mini",
     "price_in": 0.40, "price_out": 1.60},
    {"name": "GPT-5.4-mini", "backend": "openai", "model": "gpt-5.4-mini-2026-03-17",
     "price_in": 0.75, "price_out": 4.50},
]

# A realistic short CV (pasted as cv_text — the /api/cv JSON-paste path).
CV_TEXT = """Maya Cohen
Data Analyst | Tel Aviv, Israel

Summary
Data analyst with 3 years of experience turning raw data into dashboards and
decisions. Strong SQL and Python; builds BI reporting for product and growth teams.

Experience
Data Analyst, Wix — 2022-2025
- Built churn and funnel dashboards in Tableau used weekly by the growth team.
- Wrote SQL pipelines over Postgres; automated weekly reporting in Python.
Junior Analyst, Payoneer — 2021-2022
- Ad-hoc analysis and A/B test readouts for the payments product.

Education
B.A. Economics & Statistics, Tel Aviv University, 2021

Skills
SQL, Python, Tableau, Excel, A/B testing, statistics
"""


# =====================================================================
# NATIVE ANTHROPIC TOOL LOOP — faithful mirror of agent_loop.run_agent_turn,
# returning the EXACT same dict shape (so app.py's /api/chat handler is happy),
# but driving Claude's tools API. Reuses al.TOOLS / al.AGENT_SYSTEM / al._exec.
# =====================================================================
def _anthropic_tools():
    out = []
    for t in al.TOOLS:
        f = t["function"]
        out.append({"name": f["name"], "description": f["description"],
                    "input_schema": f.get("parameters", {"type": "object", "properties": {}})})
    return out


def _block_to_dict(b):
    if b.type == "text":
        return {"type": "text", "text": b.text}
    if b.type == "tool_use":
        return {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
    return {"type": b.type}


def run_anthropic_turn(prompt, history=None, last_q="", offset=0, screen_matches=None):
    import anthropic
    client = anthropic.Anthropic(api_key=ANTH_KEY)
    tools = _anthropic_tools()

    messages = []
    for t in (history or [])[-8:]:
        role = "assistant" if t.get("role") in ("agent", "assistant") else "user"
        messages.append({"role": role, "content": str(t.get("content", ""))[:1000]})
    messages.append({"role": "user", "content": prompt})

    st = {"matches": [], "mode": "chat", "state": None, "query": None}
    trace, final = [], ""

    for _hop in range(al.MAX_HOPS):
        resp = client.messages.create(model=ag.LLM_MODEL, max_tokens=1400,
                                      system=al.AGENT_SYSTEM, tools=tools, messages=messages)
        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        text = "".join(b.text for b in resp.content if b.type == "text")
        if not tool_uses:
            final = text
            break
        messages.append({"role": "assistant", "content": [_block_to_dict(b) for b in resp.content]})
        results = []
        for tu in tool_uses:
            args = tu.input if isinstance(tu.input, dict) else {}
            result = al._exec(tu.name, args, st, last_q, offset, screen_matches)
            trace.append({"tool": tu.name, "args": args, "result": result})
            results.append({"type": "tool_result", "tool_use_id": tu.id,
                            "content": json.dumps(result, ensure_ascii=False)})
        messages.append({"role": "user", "content": results})
        if st["mode"] == "interview_kickoff":
            return {"matches": [], "intro": "", "tip": "", "reply": "",
                    "mode": "interview_kickoff", "seed": st.get("seed", {}),
                    "state": None, "query": None, "tool_trace": trace}
    else:
        final = final or "לא הצלחתי להשלים את הבקשה — נסה לנסח אחרת."

    return {"matches": st["matches"], "intro": final, "tip": "", "reply": final,
            "mode": st["mode"], "state": st["state"], "query": st["query"], "tool_trace": trace}


# Patch the agent loop so /api/chat's no-CV path uses the Anthropic loop when the
# model under test is Claude — transparent to app.py.
_orig_run_agent_turn = al.run_agent_turn


def _dispatch_agent_turn(prompt, **kw):
    if ag.LLM_BACKEND == "anthropic":
        return run_anthropic_turn(prompt, **kw)
    return _orig_run_agent_turn(prompt, **kw)


al.run_agent_turn = _dispatch_agent_turn


def select_model(spec):
    ag.LLM_BACKEND = spec["backend"]
    ag.LLM_MODEL = spec["model"]
    if spec["backend"] == "openai":
        from llm_backend import OpenAIClient
        ag.claude_client = OpenAIClient(api_key=OPENAI_KEY)
    else:
        import anthropic
        ag.claude_client = anthropic.Anthropic(api_key=ANTH_KEY)


# =====================================================================
# THE CONVERSATION — a list of turns. Each turn is a function (client, S) that
# POSTs to a real endpoint, updates the session S, and returns a record dict.
# S mirrors the browser's cvState + convo + interview/tailor sub-state.
# =====================================================================
def _new_session():
    return {
        # chat/cv session (mirrors front-end cvState + convo)
        "cv_text": None, "profile": None, "history": [], "roles": [],
        "searchCriteria": None, "lastQuery": "", "offset": 0, "lastMatches": [],
        # interview sub-conversation
        "iv_profile": {}, "iv_history": [], "iv_skipped": [],
        # tailoring sub-conversation
        "tailor_job": None, "tailor_history": [], "section_index": 0, "composed": False,
    }


def _slim_matches_for_body(matches):
    return [{"rank": m.get("rank"), "title": m.get("title"), "company": m.get("company"),
             "location": m.get("location"), "experience_level": m.get("experience_level"),
             "salary_band": m.get("salary_band")} for m in (matches or [])[:5]]


def _post(client, path, body):
    r = client.post(path, data=json.dumps(body), content_type="application/json")
    try:
        return r.get_json()
    except Exception:
        return {"_raw": r.data.decode("utf-8", "replace")}


# ---- endpoint drivers (mirror front-end app.js request/response threading) ----
def chat_turn(client, S, prompt):
    S["history"].append({"role": "user", "content": prompt})
    body = {"prompt": prompt, "lastQuery": S["lastQuery"], "offset": S["offset"],
            "history": S["history"][-8:], "lastMatches": _slim_matches_for_body(S["lastMatches"])}
    if S["cv_text"]:
        body["cv_text"] = S["cv_text"]; body["profile"] = S["profile"]; body["cv_roles"] = S["roles"]
    if S["searchCriteria"]:
        body["searchCriteria"] = S["searchCriteria"]
    data = _post(client, "/api/chat", body)
    # thread response back into session (front-end app.js:397-407)
    if data.get("matches"):
        S["lastMatches"] = data["matches"]
    mode = data.get("mode")
    got = len(data.get("matches") or [])
    if mode == "more":
        S["offset"] += got
    elif mode == "search":
        S["lastQuery"] = data.get("query") or prompt; S["offset"] = got
    agent_text = data.get("intro") or data.get("reply") or ""
    if agent_text:
        S["history"].append({"role": "assistant", "content": agent_text})
    S["history"] = S["history"][-16:]
    if S["cv_text"] and isinstance(data.get("roles"), list):
        S["roles"] = data["roles"]
    if data.get("searchCriteria"):
        S["searchCriteria"] = data["searchCriteria"]
    return data


def profile_build_turn(client, S, user_msg):
    body = {"profile": S["iv_profile"], "user_msg": user_msg,
            "history": S["iv_history"][-8:], "skipped": S["iv_skipped"], "us_only": True}
    if user_msg:
        S["iv_history"].append({"role": "user", "content": user_msg})
    data = _post(client, "/api/profile-build", body)
    S["iv_profile"] = data.get("profile", S["iv_profile"])
    S["iv_skipped"] = data.get("skipped", S["iv_skipped"])
    if data.get("reply"):
        S["iv_history"].append({"role": "assistant", "content": data["reply"]})
    if data.get("done") and data.get("cv_text"):
        # interview finished -> load the synthesized profile into the chat session
        S["cv_text"] = data["cv_text"]; S["profile"] = S["iv_profile"]
        S["roles"] = (S["iv_profile"].get("titles_held") or [])
    return data


def cv_upload_turn(client, S):
    data = _post(client, "/api/cv", {"cv_text": CV_TEXT})
    if data.get("ok"):
        S["cv_text"] = data.get("cv_text") or CV_TEXT
        S["profile"] = data.get("profile")
        S["history"] = []                      # front-end resets the chat on a fresh CV
        S["roles"] = []
        S["searchCriteria"] = None
    return data


def cv_tailor_turn(client, S, user_msg, is_open=False):
    body = {"cv_text": S["cv_text"], "job": S["tailor_job"] or {},
            "history": S["tailor_history"][-40:], "user_msg": user_msg,
            "open": is_open, "composed": S["composed"], "has_cv": True,
            "section_index": S["section_index"]}
    if user_msg:
        S["tailor_history"].append({"role": "user", "content": user_msg})
    data = _post(client, "/api/cv-tailor", body)
    if data.get("reply"):
        S["tailor_history"].append({"role": "assistant", "content": data["reply"]})
    if "next_section_index" in data:
        S["section_index"] = data["next_section_index"]
    if data.get("proposed_cv"):
        S["composed"] = True
    return data


# ---- rubric helpers ----
def _tools(data):
    return [t.get("tool") for t in (data.get("tool_trace") or [])]


# =====================================================================
# THE SCRIPT — ordered (label, flow, fn, rubric). rubric(data, S) -> (ok, got_str)
# =====================================================================
def build_script():
    steps = []

    def add(label, flow, fn, rubric, expected):
        steps.append({"label": label, "flow": flow, "fn": fn, "rubric": rubric, "expected": expected})

    # --- Phase 1: cold open, no CV (agentic loop / anthropic shim) ---
    add("1 scope-guard (cold)", "scope",
        lambda c, S: chat_turn(c, S, "מי ניצח במונדיאל אתמול?"),
        lambda d, S: (len(_tools(d)) == 0 and not d.get("matches"), f"tools={_tools(d)} mode={d.get('mode')}"),
        "decline, NO tool")
    add("2 smalltalk", "intent",
        lambda c, S: chat_turn(c, S, "מה אתה יודע לעשות?"),
        lambda d, S: ("small_talk" in _tools(d) or d.get("mode") == "chat", f"tools={_tools(d)} mode={d.get('mode')}"),
        "small_talk / capabilities")
    add("3 search w/o profile -> kickoff", "intent+kickoff",
        lambda c, S: chat_turn(c, S, "אני רוצה שתעזור לי למצוא עבודה"),
        lambda d, S: (d.get("mode") == "interview", f"mode={d.get('mode')}"),
        "mode=interview (start_profile_interview)")

    # seed the interview sub-state from the kickoff response
    def _seed_iv(c, S):
        d = S.get("_last_kickoff") or {}
        S["iv_profile"] = d.get("profile") or {}
        return d

    # --- Phase 2: build-profile interview (4 fields) ---
    add("4 interview: role", "build-profile",
        lambda c, S: profile_build_turn(c, S, "data analyst, וגם BI analyst"),
        lambda d, S: (bool(S["iv_profile"].get("titles_held")) and d.get("next_field") == "experience",
                      f"titles={S['iv_profile'].get('titles_held')} next={d.get('next_field')}"),
        "role captured -> ask experience")
    add("5 interview: experience", "build-profile",
        lambda c, S: profile_build_turn(c, S, "3 שנים, רמת ג'וניור"),
        lambda d, S: ((S["iv_profile"].get("years") is not None or S["iv_profile"].get("seniority"))
                      and d.get("next_field") == "location",
                      f"years={S['iv_profile'].get('years')} sen={S['iv_profile'].get('seniority')} next={d.get('next_field')}"),
        "exp captured -> ask location")
    add("6 interview: location", "build-profile",
        lambda c, S: profile_build_turn(c, S, "ניו יורק"),
        lambda d, S: (S["iv_profile"].get("state") == "NY" and d.get("next_field") == "skills",
                      f"state={S['iv_profile'].get('state')} next={d.get('next_field')}"),
        "state=NY -> ask skills")
    add("7 interview: skills -> done", "build-profile",
        lambda c, S: profile_build_turn(c, S, "Python, SQL, Tableau"),
        lambda d, S: (d.get("done") and bool(d.get("cv_text")),
                      f"done={d.get('done')} cv_text={bool(d.get('cv_text'))}"),
        "profile complete, cv_text synthesized")

    # --- Phase 3: real search (profile now loaded) + paging + intent boundaries ---
    add("8 real job search", "search",
        lambda c, S: chat_turn(c, S, "מעולה, צא לחפש"),
        lambda d, S: (d.get("mode") == "search" and len(d.get("matches") or []) > 0,
                      f"mode={d.get('mode')} matches={len(d.get('matches') or [])}"),
        "mode=search, NY data-analyst matches")
    add("9 paging (memory)", "paging",
        lambda c, S: chat_turn(c, S, "עוד"),
        lambda d, S: (d.get("mode") == "more", f"mode={d.get('mode')} matches={len(d.get('matches') or [])}"),
        "more_results, offset advances")
    add("10 data question", "intent:data",
        lambda c, S: chat_turn(c, S, "כמה מרוויח data analyst בממוצע בניו יורק?"),
        lambda d, S: (d.get("mode") == "data_question", f"mode={d.get('mode')}"),
        "mode=data_question (computed stat)")
    add("11 career advice", "intent:advice",
        lambda c, S: chat_turn(c, S, "איך כדאי לי להתכונן לראיון לתפקיד הזה?"),
        lambda d, S: (d.get("mode") == "advice", f"mode={d.get('mode')}"),
        "mode=advice")
    add("12 scope-guard (mid-context)", "scope",
        lambda c, S: chat_turn(c, S, "רגע, תן לי מתכון לפסטה ברוטב עגבניות"),
        lambda d, S: (d.get("mode") in ("chat", "talk") and not d.get("matches"),
                      f"mode={d.get('mode')} matches={len(d.get('matches') or [])}"),
        "decline, stay in scope (no search)")

    # --- Phase 4: CV upload flow ---
    add("13 CV upload", "cv-upload",
        lambda c, S: cv_upload_turn(c, S),
        lambda d, S: (d.get("ok") and bool((d.get("profile") or {}).get("titles_held") or d.get("titles")),
                      f"ok={d.get('ok')} titles={(d.get('titles') or [])[:2]}"),
        "profile + titles + scorecard")
    add("14 CV-ranked search", "search:cv",
        lambda c, S: chat_turn(c, S, "כן, חפש לי את התפקידים האלה"),
        lambda d, S: (d.get("mode") == "search" and len(d.get("matches") or []) > 0,
                      f"mode={d.get('mode')} matches={len(d.get('matches') or [])}"),
        "mode=search, matches carry cv_fit%")

    # capture the job to tailor against (top match from turn 14)
    def _grab_job(c, S):
        m = (S["lastMatches"] or [{}])[0]
        S["tailor_job"] = {"title": m.get("title", "data analyst"), "company": m.get("company", ""),
                           "description": "", "skills": ""}
        return {"_job": S["tailor_job"]}

    # --- Phase 5: tailoring handoff + section-walk + compose ---
    add("15 tailor handoff", "tailor",
        lambda c, S: (_grab_job(c, S), chat_turn(c, S, "תתאים לי את הקורות חיים למשרה מספר 1"))[1],
        lambda d, S: (d.get("mode") == "tailor", f"mode={d.get('mode')} q={d.get('tailor_query')!r}"),
        "mode=tailor")
    add("16 tailor: opener", "tailor-walk",
        lambda c, S: cv_tailor_turn(c, S, "", is_open=True),
        lambda d, S: (bool(d.get("reply")), f"reply={bool(d.get('reply'))} sec_idx={S['section_index']}"),
        "asks about section 1")

    SECTION_ANSWERS = [
        "השם שלי מאיה כהן, אימייל maya@example.com, טלפון 050-1234567, לינקדאין /in/mayacohen",
        "תואר ראשון בכלכלה וסטטיסטיקה מאוניברסיטת תל אביב, 2021",
        "אנליסטית דאטה בוויקס 2022-2025: בניתי דשבורדים של churn ו-funnel ב-Tableau, וכתבתי pipelines ב-SQL",
        "פרויקט: אוטומציה של דוחות שבועיים בפייתון שחסכה כ-10 שעות עבודה בשבוע",
        "SQL, Python, Tableau, Excel, A/B testing, סטטיסטיקה",
        "תקציר: אנליסטית דאטה עם 3 שנות ניסיון בהפיכת דאטה לדשבורדים ולהחלטות עסקיות",
    ]
    for j, ans in enumerate(SECTION_ANSWERS):
        idx = 17 + j
        add(f"{idx} tailor: section {j+1}", "tailor-walk",
            (lambda ans: lambda c, S: cv_tailor_turn(c, S, ans))(ans),
            lambda d, S: (bool(d.get("reply")) or bool(d.get("section_done")),
                          f"section_done={d.get('section_done')} next_idx={d.get('next_section_index')}"),
            f"gather section {j+1}, advance")

    add("23 tailor: compose", "tailor-compose",
        lambda c, S: cv_tailor_turn(c, S, "סיימתי, תכתוב את הגרסה הסופית"),
        lambda d, S: (bool(d.get("proposed_cv")), f"proposed_cv={bool(d.get('proposed_cv'))} done={d.get('done')}"),
        "proposed_cv emitted")

    return steps


# =====================================================================
# RUN
# =====================================================================
def run_conversation(client, S, steps, transcript=None):
    """Drive one full conversation; return (passed, total, latencies)."""
    passed, latencies = 0, []
    for st in steps:
        t0 = time.time()
        try:
            data = st["fn"](client, S)
        except Exception as e:
            data = {"_error": f"{type(e).__name__}: {e}"}
        dt = time.time() - t0
        latencies.append(dt)
        # stash kickoff response so the interview can seed from it
        if st["label"].startswith("3 "):
            S["_last_kickoff"] = data
            S["iv_profile"] = data.get("profile") or {}
        try:
            ok, got = st["rubric"](data, S)
        except Exception as e:
            ok, got = False, f"rubric-error {type(e).__name__}: {e}"
        passed += bool(ok)
        if transcript is not None:
            transcript.append({"turn": st["label"], "flow": st["flow"], "ok": bool(ok),
                               "expected": st["expected"], "got": got, "latency": round(dt, 3)})
    return passed, len(steps), latencies


def grade_model(spec, steps, save_transcript=False):
    select_model(spec)
    client = webapp.app.test_client()
    total_pass = total = 0
    latencies, all_fail, transcript = [], [], ([] if save_transcript else None)
    for k in range(K):
        S = _new_session()
        tr = [] if (save_transcript and k == 0) else None
        p, n, lat = run_conversation(client, S, steps, transcript=tr)
        total_pass += p; total += n; latencies += lat
        if tr is not None:
            transcript = tr
    return {"name": spec["name"], "model": spec["model"],
            "adherence": total_pass / total, "passed": total_pass, "total": total,
            "lat_mean": statistics.mean(latencies), "lat_median": statistics.median(latencies),
            "price_in": spec["price_in"], "price_out": spec["price_out"],
            "transcript": transcript}


def run():
    steps = build_script()
    print(f"FULL-CONVERSATION COMPARISON  |  {len(steps)} turns x K={K} runs  |  real endpoints + session memory")
    print("=" * 104)
    results = []
    for spec in MODELS:
        print(f"\n>>> {spec['name']}  ({spec['model']}, {spec['backend']}) ...", flush=True)
        r = grade_model(spec, steps, save_transcript=True)
        results.append(r)
        print(f"    adherence : {r['passed']}/{r['total']} = {r['adherence']:.1%}")
        print(f"    latency   : mean {r['lat_mean']:.3f}s | median {r['lat_median']:.3f}s")
        for t in (r["transcript"] or []):
            mark = "PASS" if t["ok"] else "FAIL"
            print(f"      [{mark}] {t['turn']:32s} ({t['flow']:14s}) {t['latency']:.2f}s  "
                  f"exp:{t['expected']}  got:{t['got']}")

    print("\n" + "=" * 104)
    print("SUMMARY — full-conversation agentic comparison")
    print("-" * 104)
    print(f"{'model':22s} {'conv adherence':>15s} {'price in/out (1M)':>20s} {'resp time':>11s}")
    print("-" * 104)
    for r in results:
        price = f"${r['price_in']:.2f}/${r['price_out']:.2f}"
        print(f"{r['name']:22s} {r['adherence']:>14.0%} {price:>20s} {r['lat_mean']:>9.3f}s")
    print("-" * 104)

    out = os.path.join(_ROOT, "output", "model_comparison_conversation.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in r.items()} for r in results], f, ensure_ascii=False, indent=2)
    print("Saved ->", out)
    return results


if __name__ == "__main__":
    run()
