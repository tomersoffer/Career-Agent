# -*- coding: utf-8 -*-
# =====================================================================
# MODEL SELECTION & COMPARISON HARNESS  (Claude vs GPT-4.1-mini vs GPT-5.4-mini)
# ---------------------------------------------------------------------
# Reproduces the project's "בחירת מודל והשוואת חלופות" findings with REAL,
# live calls. Every candidate model is graded on the SAME labeled battery
# (tests/eval_intent_live.py::CASES) through the SAME backend-agnostic path
# (agent_runner.parse_query -> _llm_parse -> client.messages.create), so the
# only thing that changes between columns is the model under test.
#
# Two of the three metrics in the report table are measured here directly:
#   1. עמידה בהנחיות (instruction-adherence / structured-output + routing):
#      mean accuracy on the GRADED fields (mode/intent/state/wants_more)
#      over N cases x K runs — this captures both "did it return valid,
#      schema-correct JSON" and "did it route/extract entities correctly".
#   2. זמן תגובה (response time): mean wall-clock latency per parse call.
# (Cost per 1M tokens is list pricing, not measured — printed for context.)
#
#   run:  python -X utf8 tests/compare_models.py [K]      (default K=5)
# =====================================================================
import os, sys, time, json, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "local.env"))

import agent_runner as ag
from eval_intent_live import CASES, GRADED

K = int(sys.argv[1]) if len(sys.argv) > 1 else 5

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
ANTH_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()

# The three candidates from the report, with their list pricing (USD / 1M tokens).
# Ordered baseline-first to mirror the document's table.
MODELS = [
    {"name": "Claude Sonnet 4.6", "backend": "anthropic", "model": "claude-sonnet-4-6",
     "price_in": 3.00, "price_out": 15.00},
    {"name": "GPT-4.1-mini", "backend": "openai", "model": "gpt-4.1-mini",
     "price_in": 0.40, "price_out": 1.60},
    {"name": "GPT-5.4-mini", "backend": "openai", "model": "gpt-5.4-mini-2026-03-17",
     "price_in": 0.75, "price_out": 4.50},
]


def select_model(spec):
    """Swap ONLY the live client + model id on the already-loaded agent_runner,
    so the heavy KB (df / tfidf / kmeans) is loaded once and shared across columns."""
    ag.LLM_BACKEND = spec["backend"]
    ag.LLM_MODEL = spec["model"]
    if spec["backend"] == "openai":
        from llm_backend import OpenAIClient
        ag.claude_client = OpenAIClient(api_key=OPENAI_KEY)
    else:
        import anthropic
        ag.claude_client = anthropic.Anthropic(api_key=ANTH_KEY)


def grade_model(spec):
    select_model(spec)
    # one warm-up call so connection setup / TLS handshake isn't charged to the latency mean
    try:
        ag.parse_query("data analyst")
    except Exception:
        pass

    total_pass, latencies, failures = 0, [], {}
    for i, c in enumerate(CASES, 1):
        expected = {k: c[k] for k in GRADED if k in c}
        for _ in range(K):
            t0 = time.time()
            got = ag.parse_query(c["q"])
            latencies.append(time.time() - t0)
            actual = {k: got.get(k) for k in expected}
            if actual == expected:
                total_pass += 1
            else:
                failures.setdefault(i, {"q": c["q"], "expected": expected, "sample": actual, "fails": 0})
                failures[i]["fails"] += 1
    n = len(CASES)
    return {
        "name": spec["name"], "model": spec["model"],
        "adherence": total_pass / (n * K),
        "passed": total_pass, "total": n * K,
        "lat_mean": statistics.mean(latencies),
        "lat_median": statistics.median(latencies),
        "price_in": spec["price_in"], "price_out": spec["price_out"],
        "failures": failures,
    }


def run():
    print(f"MODEL COMPARISON  |  {len(CASES)} labeled cases x K={K} runs each  "
          f"|  metric: GRADED={GRADED}")
    print("=" * 96)
    results = []
    for spec in MODELS:
        print(f"\n>>> {spec['name']}  ({spec['model']}, backend={spec['backend']}) ...")
        r = grade_model(spec)
        results.append(r)
        print(f"    adherence : {r['passed']}/{r['total']} = {r['adherence']:.1%}")
        print(f"    latency   : mean {r['lat_mean']:.4f}s | median {r['lat_median']:.4f}s")
        if r["failures"]:
            print(f"    misroutes : {len(r['failures'])} case(s) failed at least once")
            for i, f in sorted(r["failures"].items()):
                diff = {k: f"exp {f['expected'][k]!r} got {f['sample'][k]!r}"
                        for k in f["expected"] if f["expected"][k] != f["sample"][k]}
                print(f"        #{i:02d} ({f['fails']}/{K}) {f['q']}  -> {diff}")

    # ---- final side-by-side table (mirrors the report's "בחירת מודל" table) ----
    print("\n" + "=" * 96)
    print("SUMMARY — model selection comparison")
    print("-" * 96)
    hdr = f"{'model':22s} {'adherence':>11s} {'price in/out (1M)':>20s} {'resp time':>11s}"
    print(hdr)
    print("-" * 96)
    for r in results:
        price = f"${r['price_in']:.2f}/${r['price_out']:.2f}"
        print(f"{r['name']:22s} {r['adherence']:>10.0%} {price:>20s} {r['lat_mean']:>9.3f}s")
    print("-" * 96)

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "output", "model_comparison.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in r.items() if k != "failures"} for r in results],
                  f, ensure_ascii=False, indent=2)
    print("Saved ->", out)
    return results


if __name__ == "__main__":
    run()
