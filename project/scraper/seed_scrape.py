# =====================================================================
# SEED SCRAPE - bootstrap a big Israel jobs dataset.
# A keyword-less LinkedIn search dead-ends at ~1 page, so we SWEEP many
# role keywords (location = Israel), each paginating deep, and accumulate
# into the one master CSV, DEDUPED BY LINKEDIN POSTING ID. We keep going
# until TARGET unique jobs are stored, or until a full pass adds almost
# nothing (LinkedIn has no more public Israel jobs to give).
#
# Run:  python -m scraper.seed_scrape          (uses TARGET below)
#       python -m scraper.seed_scrape 4000     (override target)
# =====================================================================
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")   # silence the KMeans/MKL warning

import datetime
import sys

from . import apify_scraper, build_models, config, store

TARGET = 4000          # unique jobs we want stored (dedup by posting id)
PER_KEYWORD = 500      # ceiling per keyword search (LinkedIn decides the real count)
MAX_PASSES = 3         # cycle the keyword list at most this many times
MIN_NEW_PER_PASS = 20  # if a whole pass adds fewer than this, LinkedIn is dry -> stop

KEYWORDS = [
    "software engineer", "software developer", "backend developer", "frontend developer",
    "full stack developer", "data analyst", "data scientist", "data engineer",
    "machine learning engineer", "devops engineer", "qa engineer", "automation engineer",
    "product manager", "project manager", "program manager", "ux designer", "ui designer",
    "graphic designer", "sales", "account executive", "business development",
    "sales development representative", "marketing", "digital marketing", "content writer",
    "social media manager", "finance", "accountant", "bookkeeper", "financial analyst",
    "controller", "human resources", "recruiter", "talent acquisition", "office manager",
    "operations manager", "customer success", "customer support", "technical support",
    "business analyst", "system administrator", "network engineer", "security engineer",
    "cyber security analyst", "embedded engineer", "electrical engineer", "mechanical engineer",
    "hardware engineer", "team lead", "engineering manager", "legal counsel", "nurse",
]


def _log(msg):
    print("[seed] " + msg, flush=True)


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def run(target=TARGET):
    # start fresh so the bootstrap is a single clean, full-schema dataset
    if os.path.exists(config.MASTER_CSV):
        try:
            os.remove(config.MASTER_CSV)
            _log("cleared existing master CSV for a clean bootstrap")
        except Exception as e:
            _log(f"could not clear master CSV ({e}) - continuing, will accumulate")

    _log(f"target = {target} unique jobs (deduped by posting id) | {len(KEYWORDS)} keywords")
    for pass_no in range(1, MAX_PASSES + 1):
        pass_start = store.count()
        _log(f"===== pass {pass_no} (stored so far: {pass_start}) =====")
        for kw in KEYWORDS:
            if store.count() >= target:
                break
            filters = {"title": kw, "location": "Israel", "limit": PER_KEYWORD,
                       "experienceLevel": [], "datePosted": "", "contractType": [], "remote": ""}
            before = store.count()
            try:
                rows = apify_scraper.scrape(filters)
            except Exception as e:
                _log(f"'{kw}': FAILED ({e})")
                continue
            total = store.upsert(rows, _now())
            _log(f"'{kw}': scraped {len(rows):4d} -> +{total - before:4d} new | stored {total}")

        gained = store.count() - pass_start
        _log(f"pass {pass_no} added {gained} new (total {store.count()})")
        if store.count() >= target:
            _log(f"TARGET REACHED: {store.count()} >= {target}")
            break
        if gained < MIN_NEW_PER_PASS:
            _log(f"pass added only {gained} (< {MIN_NEW_PER_PASS}) - LinkedIn is dry, stopping early")
            break

    _log(f"building models on {store.count()} jobs ...")
    info = build_models.build()
    _log(f"models: {info}")
    _log(f"DONE. stored {store.count()} unique jobs.")
    return store.count()


if __name__ == "__main__":
    tgt = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else TARGET
    run(tgt)
