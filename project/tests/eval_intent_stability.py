# -*- coding: utf-8 -*-
# =====================================================================
# INTENT-ROUTING STABILITY EVALUATION  (non-determinism measurement)
# ---------------------------------------------------------------------
# A single live run is not enough to validate a routing layer: at the default
# sampling temperature the model is non-deterministic, so the SAME prompt can
# route differently across calls. This harness runs every labeled case K times
# and reports a per-case pass-rate plus the mean accuracy, exposing both
# consistent failures and intermittent (flaky) ones.
#
#   run:  python -X utf8 tests/eval_intent_stability.py [K]   (default K=5)
# =====================================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_runner as ag
from eval_intent_live import CASES, GRADED

K = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def run():
    print(f"MODEL: {ag.LLM_MODEL} | backend: {ag.LLM_BACKEND} | runs per case: K={K}")
    print("=" * 96)
    total_pass = 0
    flaky, consistent_fail = [], []
    for i, c in enumerate(CASES, 1):
        expected = {k: c[k] for k in GRADED if k in c}
        passes, sample_fail = 0, None
        for _ in range(K):
            got = ag.parse_query(c["q"])
            actual = {k: got.get(k) for k in expected}
            if actual == expected:
                passes += 1
            elif sample_fail is None:
                sample_fail = actual
        total_pass += passes
        rate = passes / K
        tag = "STABLE-PASS" if passes == K else ("STABLE-FAIL" if passes == 0 else "FLAKY")
        print(f"[{tag:11s}] #{i:02d}  {passes}/{K}  {c['q']}")
        if passes < K:
            print(f"               expected {expected}")
            print(f"               sample-fail {sample_fail}")
        if 0 < passes < K:
            flaky.append((i, c["q"], passes))
        elif passes == 0:
            consistent_fail.append((i, c["q"]))

    n = len(CASES)
    print("\n" + "=" * 96)
    print(f"MEAN ACCURACY over {n} cases x {K} runs: {total_pass}/{n * K} = {total_pass / (n * K):.1%}")
    if consistent_fail:
        print(f"\nCONSISTENT FAILURES (0/{K}) — chat reliably fails to identify the request:")
        for i, q in consistent_fail:
            print(f"  #{i} {q}")
    if flaky:
        print(f"\nFLAKY (non-deterministic routing — same input, different decisions):")
        for i, q, p in flaky:
            print(f"  #{i} {q}  ({p}/{K} correct)")
    if not consistent_fail and not flaky:
        print("\nAll cases stable-pass across all runs.")


if __name__ == "__main__":
    run()
