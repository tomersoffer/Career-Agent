# -*- coding: utf-8 -*-
# =====================================================================
# AGENTIC TOOL-CALLING FLOW COMPARISON  (Claude vs GPT-4.1-mini vs GPT-5.4-mini)
# ---------------------------------------------------------------------
# The single-shot intent battery (tests/compare_models.py) does NOT separate the
# three candidates — they all classify JSON well. The project's real selection
# criterion was AGENTIC TOOL-CALLING: can the model pick the right tool, pass
# valid arguments, CHAIN several tools across hops (search -> stat), and HOLD the
# scope guard (refuse off-topic with NO tool) — consistently, across runs.
#
# This harness grades exactly that. Every model drives the SAME agent
# (agent_loop.TOOLS / AGENT_SYSTEM / _exec):
#   * OpenAI models  -> agent_loop.run_agent_turn   (OpenAI native tools API)
#   * Claude         -> run_anthropic_turn (below)  (Anthropic native tools API)
# so the only variable is the model. Each case is run K times; "Flow adherence"
# is the share of runs whose tool TRACE satisfies the case's rubric.
#
#   run:  python -X utf8 tests/compare_models_agentic.py [K]    (default K=3)
# =====================================================================
import os, sys, json, time, statistics
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests"))

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, "..", "local.env"))
os.environ.setdefault("LLM_BACKEND", "openai")   # agent_loop imports require an OpenAI key

import agent_runner as ag
import agent_loop as al

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

# =====================================================================
# THE FLOW BATTERY — each case carries a rubric over the tool TRACE.
#   need_seq : tools that must appear, IN THIS ORDER, as a subsequence
#              (this is what catches a model that won't CHAIN search->stat)
#   forbid   : tools that must NOT be called (scope guard: [] tools for off-topic)
#   no_tool  : True  -> the model must answer with NO tool call at all
#   kw       : extra run_agent_turn kwargs (e.g. last_q for paging)
# =====================================================================
CASES = [
    # ---- single-tool routing ----
    {"q": "אני מחפש משרת data analyst בניו יורק", "need_seq": ["search_jobs"]},
    {"q": "find me nurse jobs in Texas", "need_seq": ["search_jobs"]},
    {"q": "מה השכר הטיפוסי ל-data analyst בניו יורק?", "need_seq": ["get_job_stat"]},
    {"q": "מהם הכישורים הנפוצים ביותר למשרות data analyst?", "need_seq": ["get_job_stat"]},
    {"q": "כדאי לי ללמוד Python או SQL בשביל data analyst?", "need_seq": ["give_career_advice"]},
    {"q": "מה אתה יודע לעשות?", "need_seq": ["small_talk"]},
    {"q": "עוד", "need_seq": ["more_results"],
     "kw": {"last_q": "data analyst בניו יורק", "offset": 3,
            "history": [{"role": "user", "content": "אני מחפש משרת data analyst בניו יורק"},
                        {"role": "agent", "content": "הנה 3 משרות data analyst בניו יורק שמצאתי עבורך."}]}},

    # ---- MULTI-STEP CHAINS (the real Flow stressors: act -> observe -> act) ----
    # one request that requires TWO tools, in order: find jobs, THEN compute a stat.
    {"q": "תמצא לי משרות data analyst בטקסס, וגם כמה הן משלמות בממוצע",
     "need_seq": ["search_jobs", "get_job_stat"]},
    # stat first, then a concrete listing for the same role.
    {"q": "כמה משרות data analyst יש בקליפורניה, ואז תראה לי כמה מהן",
     "need_seq": ["get_job_stat", "search_jobs"]},
    # search, then page the SAME search in the same turn.
    {"q": "find data analyst jobs in New York, then show me the next few as well",
     "need_seq": ["search_jobs"]},

    # ---- SCOPE GUARD (must HOLD the system prompt: refuse, call NO tool) ----
    {"q": "מה מזג האוויר היום בתל אביב?", "no_tool": True},
    {"q": "מי ניצח במונדיאל האחרון?", "no_tool": True},
    {"q": "תן לי מתכון לעוגת שוקולד", "no_tool": True},
]


# =====================================================================
# ANTHROPIC NATIVE TOOL LOOP — mirror of agent_loop.run_agent_turn, but driving
# Claude's tools API. Reuses agent_loop.TOOLS / AGENT_SYSTEM / _exec verbatim, so
# the agent's behaviour (executors, schemas, system prompt) is identical; only the
# wire format (OpenAI function-calling <-> Anthropic tool_use) differs.
# =====================================================================
def _anthropic_tools():
    """Translate the OpenAI function schemas into Anthropic tool schemas."""
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


def run_anthropic_turn(prompt, model, history=None, last_q="", offset=0, screen_matches=None):
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
        resp = client.messages.create(model=model, max_tokens=1400,
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
            break
    else:
        final = final or ""

    return {"mode": st["mode"], "state": st["state"], "query": st["query"],
            "reply": final, "tool_trace": trace}


def drive(spec, case):
    """Run one case through the right backend; returns the tool name sequence."""
    kw = case.get("kw", {})
    if spec["backend"] == "anthropic":
        r = run_anthropic_turn(case["q"], spec["model"], **kw)
    else:
        ag.LLM_MODEL = spec["model"]
        r = al.run_agent_turn(case["q"], **kw)
    return [t["tool"] for t in r["tool_trace"]]


# =====================================================================
# RUBRIC SCORING
# =====================================================================
def _is_subsequence(need, got):
    it = iter(got)
    return all(any(g == n for g in it) for n in need)


def score(case, tools):
    if case.get("no_tool"):
        return len(tools) == 0          # scope guard: any tool call is a Flow break
    need = case.get("need_seq", [])
    if any(f in tools for f in case.get("forbid", [])):
        return False
    return _is_subsequence(need, tools)


def grade_model(spec):
    if spec["backend"] == "openai":
        ag.LLM_MODEL = spec["model"]
    # warm up the connection so the handshake isn't charged to the latency mean
    try:
        drive(spec, {"q": "data analyst"})
    except Exception:
        pass

    total_pass, latencies, failures = 0, [], []
    for i, c in enumerate(CASES, 1):
        for _ in range(K):
            t0 = time.time()
            try:
                tools = drive(spec, c)
            except Exception as e:
                tools = [f"ERR:{type(e).__name__}"]
            latencies.append(time.time() - t0)
            ok = score(c, tools)
            total_pass += ok
            if not ok:
                failures.append((i, c["q"], _expected_str(c), tools))
    n = len(CASES)
    return {"name": spec["name"], "model": spec["model"],
            "flow": total_pass / (n * K), "passed": total_pass, "total": n * K,
            "lat_mean": statistics.mean(latencies), "lat_median": statistics.median(latencies),
            "price_in": spec["price_in"], "price_out": spec["price_out"],
            "failures": failures}


def _expected_str(c):
    if c.get("no_tool"):
        return "NO tool (decline off-topic)"
    return " -> ".join(c.get("need_seq", []))


def run():
    print(f"AGENTIC TOOL-CALLING FLOW  |  {len(CASES)} cases x K={K} runs  |  "
          f"rubric: ordered tool-trace + scope guard")
    print("=" * 100)
    results = []
    for spec in MODELS:
        print(f"\n>>> {spec['name']}  ({spec['model']}, {spec['backend']}) ...")
        r = grade_model(spec)
        results.append(r)
        print(f"    flow adherence : {r['passed']}/{r['total']} = {r['flow']:.1%}")
        print(f"    latency        : mean {r['lat_mean']:.3f}s | median {r['lat_median']:.3f}s")
        if r["failures"]:
            print(f"    flow breaks    : {len(r['failures'])}/{r['total']}")
            for i, q, exp, got in r["failures"]:
                print(f"        #{i:02d} expected [{exp}] got {got}  |  {q}")

    print("\n" + "=" * 100)
    print("SUMMARY — agentic tool-calling flow")
    print("-" * 100)
    print(f"{'model':22s} {'flow adherence':>15s} {'price in/out (1M)':>20s} {'resp time':>11s}")
    print("-" * 100)
    for r in results:
        price = f"${r['price_in']:.2f}/${r['price_out']:.2f}"
        print(f"{r['name']:22s} {r['flow']:>14.0%} {price:>20s} {r['lat_mean']:>9.3f}s")
    print("-" * 100)

    out = os.path.join(_ROOT, "output", "model_comparison_agentic.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in r.items() if k != "failures"} for r in results],
                  f, ensure_ascii=False, indent=2)
    print("Saved ->", out)
    return results


if __name__ == "__main__":
    run()
