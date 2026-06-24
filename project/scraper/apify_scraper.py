# =====================================================================
# APIFY SCRAPER - run the LinkedIn Jobs actor on ONE search URL and
# normalise the raw items into our column schema. Token comes from
# APIFY_TOKEN in local.env (kept server-side, never sent to the browser).
# =====================================================================
import os

from dotenv import load_dotenv

from . import config

# local.env lives at Agent/local.env  (two levels above project/scraper)
load_dotenv(os.path.join(config.BASE, "..", "..", "local.env"))
APIFY_TOKEN = os.environ.get("APIFY_TOKEN", "").strip()


def get_client():
    """Lazily import the SDK so the package imports fine even before
    `pip install apify-client`; only an actual scrape needs it."""
    if not APIFY_TOKEN:
        raise RuntimeError("APIFY_TOKEN is missing - add it to local.env.")
    try:
        from apify_client import ApifyClient
    except ImportError as e:
        raise RuntimeError("apify-client not installed - run: pip install apify-client") from e
    return ApifyClient(APIFY_TOKEN)


def _clean(v):
    return "" if v is None else str(v).strip()


def normalize(items):
    """Map valig's raw items onto our schema (exact field names, with light fallbacks)."""
    rows = []
    for it in items:
        if not isinstance(it, dict):
            continue
        company = it.get("companyName") or it.get("company") or ""
        if isinstance(company, dict):
            company = company.get("name") or company.get("title") or ""
        url = _clean(it.get("url") or it.get("jobUrl") or it.get("link"))
        apply_type = _clean(it.get("applyType"))
        rows.append({
            "job_id": _clean(it.get("id") or it.get("jobId")),
            "title": _clean(it.get("title")),
            "company_name": _clean(company),
            "company_url": _clean(it.get("companyUrl")),
            "location": _clean(it.get("location")),
            "experience_level": _clean(it.get("experienceLevel") or it.get("seniorityLevel")),
            "contract_type": _clean(it.get("contractType")),
            "work_type": _clean(it.get("workType")),
            "sector": _clean(it.get("sector")),
            "salary": _clean(it.get("salary")),
            "recruiter_name": _clean(it.get("recruiterName")),
            "recruiter_url": _clean(it.get("recruiterUrl")),
            "apply_type": apply_type,
            "applications_count": _clean(it.get("applicationsCount")),
            "posted_at": _clean(it.get("postedDate") or it.get("postedAt")),
            "posted_time_ago": _clean(it.get("postedTimeAgo")),
            "job_posting_url": url,
            # EASY_APPLY jobs are applied to on the LinkedIn posting itself
            "application_url": url if apply_type == "EASY_APPLY" else "",
            "description": _clean(it.get("description")),
        })
    return rows


def build_run_input(filters):
    """Map the field-editor filter labels onto valig's structured actor input.
    Only well-set fields are included so a minimal scrape stays robust."""
    f = filters or {}
    ri = {
        "location": (f.get("location") or "Israel").strip() or "Israel",
        "limit": max(1, min(1000, int(f.get("limit") or config.DEFAULT_LIMIT))),
    }
    title = (f.get("title") or "").strip()
    if title:
        ri["title"] = title
    exp = [config.EXP_CODE[x] for x in f.get("experienceLevel", []) if x in config.EXP_CODE]
    if exp:
        ri["experienceLevel"] = exp
    ct = [config.CONTRACT_CODE[x] for x in f.get("contractType", []) if x in config.CONTRACT_CODE]
    if ct:
        ri["contractType"] = ct
    if f.get("datePosted") in config.DATE_CODE:
        ri["datePosted"] = config.DATE_CODE[f["datePosted"]]
    if f.get("remote") in config.REMOTE_CODE:
        ri["remote"] = config.REMOTE_CODE[f["remote"]]
    return ri


def scrape(filters):
    """Run the valig actor with the given filters and return normalised job rows."""
    client = get_client()
    run_input = build_run_input(filters)
    print(f"[scrape] calling actor {config.ACTOR_ID} with {run_input}", flush=True)
    run = client.actor(config.ACTOR_ID).call(run_input=run_input)
    if run is None:
        raise RuntimeError("Actor run returned no result (check the filters).")
    # apify-client 3.x returns a typed Run object (attribute access, not dict)
    print(f"[scrape] actor {run.status}; dataset={run.default_dataset_id}", flush=True)
    items = client.dataset(run.default_dataset_id).list_items().items
    print(f"[scrape] pulled {len(items)} raw items from the dataset", flush=True)
    return normalize(items)
