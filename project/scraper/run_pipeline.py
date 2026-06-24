# =====================================================================
# RUN PIPELINE - single entry point for BOTH the 8am Windows Task
# Scheduler job and the UI "Scrape Now" button:
#     scrape (valig, from saved filters) -> upsert master CSV -> rebuild models
# Run standalone:  python -m scraper.run_pipeline
# =====================================================================
import datetime
import json

from . import apify_scraper, build_models, settings_store, store


def _log(msg):
    print("[scrape] " + msg, flush=True)


def run(limit=None):
    filters = dict(settings_store.load_settings()["filters"])
    if limit:
        filters["limit"] = int(limit)
    _log(f"filters: {filters}")

    rows = apify_scraper.scrape(filters)
    _log(f"received {len(rows)} jobs from the actor")

    scraped_at = datetime.datetime.now().isoformat(timespec="seconds")
    total = store.upsert(rows, scraped_at)
    _log(f"upserted -> master CSV now holds {total} jobs (reposts deduped)")
    model_info = build_models.build()
    _log(f"models rebuilt: {model_info}")
    _log("done.")

    return {
        "ok": True,
        "scraped": len(rows),
        "stored_total": total,
        "scraped_at": scraped_at,
        "models": model_info,
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
