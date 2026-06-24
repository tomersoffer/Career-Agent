# -*- coding: utf-8 -*-
# =====================================================================
# AGENTIC TOOL-CALLING STABILITY EVALUATION  (non-determinism measurement)
# ---------------------------------------------------------------------
# The agent_loop.py counterpart to eval_intent_stability.py.
#
# A single live run is not enough to validate the agentic loop: agent_loop issues
# its tool calls with NO temperature set (agent_loop.py: max_completion_tokens only),
# so the model is non-deterministic and the SAME prompt can pick a different tool
# or different arguments across calls. This harness runs every labeled case K
# times and reports a per-case pass-rate plus the mean accuracy, exposing both
# consistent failures and intermittent (flaky) tool-selection / argument errors —
# in particular the documented Hebrew-location loss ("מתכנת בגוגל בקליפורניה").
#
# REQUIRES LLM_BACKEND=openai — agent_loop uses the OpenAI tools API directly.
#
#   run:  python -X utf8 tests/eval_agent_loop_stability.py [K]   (default K=5)
# =====================================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_runner as ag
import agent_loop
from eval_agent_loop_live import CASES, GRADED, actual_fields

K = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def run():
    print(f"MODEL: {ag.LLM_MODEL} | backend: {ag.LLM_BACKEND} | runs per case: K={K}")
    if ag.LLM_BACKEND != "openai":
        print("WARNING: agent_loop uses the OpenAI tools API; set LLM_BACKEND=openai "
              "(and OPENAI_API_KEY) or the loop cannot issue tool calls.")
    print("=" * 100)

    total_pass = 0
    flaky, consistent_fail = [], []
    for i, c in enumerate(CASES, 1):
        expected = {k: c[k] for k in GRADED if k in c}
        passes, sample_fail = 0, None
        for _ in range(K):
            result = agent_loop.run_agent_turn(c["q"], last_q=c.get("last_q", ""),
                                               offset=c.get("offset", 0))
            actual = actual_fields(result, expected)
            if actual == expected:
                passes += 1
            elif sample_fail is None:
                sample_fail = actual
        total_pass += passes
        tag = "STABLE-PASS" if passes == K else ("STABLE-FAIL" if passes == 0 else "FLAKY")
        print(f"[{tag:11s}] #{i:02d}  {passes}/{K}  {c['q']}")
        if passes < K:
            print(f"               expected    {expected}")
            print(f"               sample-fail {sample_fail}")
        if 0 < passes < K:
            flaky.append((i, c["q"], passes))
        elif passes == 0:
            consistent_fail.append((i, c["q"]))

    n = len(CASES)
    print("\n" + "=" * 100)
    print(f"MEAN ACCURACY over {n} cases x {K} runs: {total_pass}/{n * K} = {total_pass / (n * K):.1%}")
    if consistent_fail:
        print(f"\nCONSISTENT FAILURES (0/{K}) — the agent reliably picks the wrong tool/args:")
        for i, q in consistent_fail:
            print(f"  #{i} {q}")
    if flaky:
        print(f"\nFLAKY (non-deterministic tool-calling — same input, different decisions):")
        for i, q, p in flaky:
            print(f"  #{i} {q}  ({p}/{K} correct)")
    if not consistent_fail and not flaky:
        print("\nAll cases stable-pass across all runs.")


if __name__ == "__main__":
    run()
