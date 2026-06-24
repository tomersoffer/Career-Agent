# =====================================================================
# SEARCH - query WITHIN the scraped Israeli jobs using the scrape-side
# models. Reuses agent_runner.parse_query (LLM intent) + claude_client,
# which are already loaded by the Flask app, then ranks with a hybrid
# cosine on TITLE + LOCATION + EXPERIENCE.
# =====================================================================
import math
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import matching      # shared ranking core (project root) - same algorithm as the gold tab
from . import config

_cache = {"mtime": None}


def _s(v):
    """NaN/None-safe string read from a pandas cell."""
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    s = str(v).strip()
    return "" if s.lower() == "nan" else s

SYSTEM_PROMPT_SCRAPE = (
    "You are an ACTION-ORIENTED job-application agent for the ISRAELI market (not a passive advisor): your "
    "goal is to get the user applying to the right live LinkedIn roles FAST. For each scraped job, write a "
    "SHORT, energetic blurb on why it fits and why to apply NOW. Each job already has a card under your blurb "
    "(title, company, location, experience, contract, applicants, apply button), so you MAY name the company "
    "but DO NOT repeat the location, experience, contract, salary or applicants count, and DO NOT output links. "
    "Everything IN HEBREW. "
    "Return ONLY a JSON object, no text around it, exactly: "
    "{\"intro\": <one short, action-focused opener>, "
    "\"jobs\": {\"1\": <1-2 short sentences for rank #1>, \"2\": <...>, ... one entry per provided rank}, "
    "\"tip\": <one short, concrete next-step tip (e.g. which to apply to first)>}. "
    "Keep every field tight and skimmable; at most one emoji per field. "
    "RISK / OPPORTUNITY profiling — apply per job when relevant: "
    "• ANOMALY: if a job has is_anomaly == 1, START that job's blurb with '⚠️ התראת סיכון: ' and explain in "
    "one sentence that the engine flagged it as a statistical OUTLIER — its profile sits far from its "
    "market-cluster centroid (an unusual combination of role/industry/seniority/location), which may signal a "
    "suspicious or mistaken listing OR a rare opportunity, so verify the details before applying. "
    "• COMPETITION: if a job's applicants count is high (≈100+), add '🔥 תחרותי — הגישו מהר'; if it is low "
    "(≈ under 40), add '⭐ פחות תחרותי — הזדמנות טובה'. "
    "• LOCATION: if a job has in_requested_location == false, add '📍 שים לב: המשרה מחוץ למיקום שביקשת' so the "
    "user knows it is outside their target area."
)


def _ready():
    return os.path.exists(config.MASTER_CSV) and os.path.exists(config.TITLE_VEC_PATH)


def _load():
    if not _ready():
        return None
    mtime = max(os.path.getmtime(config.MASTER_CSV), os.path.getmtime(config.TITLE_VEC_PATH))
    if _cache.get("mtime") == mtime:
        return _cache
    df = pd.read_csv(config.MASTER_CSV)
    title_vec = joblib.load(config.TITLE_VEC_PATH)
    loc_vec = joblib.load(config.LOC_VEC_PATH)
    _cache.update(
        mtime=mtime, df=df, title_vec=title_vec, loc_vec=loc_vec,
        title_m=title_vec.transform(df["title"].fillna("").astype(str).str.lower()),
        loc_m=loc_vec.transform(df["location"].fillna("").astype(str).str.lower()),
        ranks=df["experience_level"].map(config.exp_rank).to_numpy().astype(float),
    )
    return _cache


def search(prompt, top_n=5, offset=0, intent=None):
    c = _load()
    if c is None or len(c["df"]) == 0:
        return [], {"empty": True}

    import agent_runner   # already imported (and loaded) by the Flask app
    intent = intent or agent_runner.parse_query(prompt)
    df = c["df"]
    n = len(df)

    # the SAME 3-dimension search as the gold tab (matching.py). TITLE + EXPERIENCE come
    # straight from the shared core; LOCATION here is a free-text cosine on the city.
    title_score, role_specified = matching.title_scores(intent, c["title_vec"], c["title_m"])
    city = (intent.get("city") or "").strip()
    loc_specified = bool(city)
    loc_outside_israel = False
    if loc_specified:
        loc_score = cosine_similarity(c["loc_vec"].transform([city.lower()]), c["loc_m"]).ravel()
        # all scraped jobs are in Israel; if the requested place doesn't match any of them,
        # DON'T filter on location - still answer with Israeli jobs and flag a warning.
        if float(loc_score.max()) < matching.LOC_FLOOR:
            loc_outside_israel = True
            loc_specified = False
            loc_score = np.zeros(n)
            intent["loc_outside_israel"] = city
    elif (intent.get("state") or "").strip():
        # a US state was parsed with no Israeli city -> the user asked for a non-Israeli location
        loc_outside_israel = True
        intent["loc_outside_israel"] = (intent.get("state") or "").strip()
        loc_score = np.zeros(n)
    else:
        loc_score = np.zeros(n)
    exp_score, exp_specified = matching.experience_scores(c["ranks"], intent.get("experience_rank", 0))

    order, hybrid = matching.select(title_score, role_specified, loc_score, loc_specified,
                                    exp_score, exp_specified, ranks=c["ranks"], top_n=offset + top_n)
    order = order[offset:]                       # skip already-shown matches ("more options" / pagination)
    if len(order) == 0:
        return [], intent

    want_city = "" if loc_outside_israel else (intent.get("city") or "").strip().lower()
    results = []
    for rank, pos in enumerate(order, start=offset + 1):
        row = df.iloc[pos]
        cid = _s(row.get("cluster_id"))
        ia = _s(row.get("is_anomaly"))
        loc = _s(row.get("location"))
        results.append({
            "rank": rank,
            "title": _s(row.get("title")),
            "company": _s(row.get("company_name")),
            "company_url": _s(row.get("company_url")),
            "location": loc,
            "loc_match": (not want_city) or (want_city in loc.lower()),   # in the requested city/area?
            "experience_level": _s(row.get("experience_level")),
            "contract_type": _s(row.get("contract_type")),
            "work_type": _s(row.get("work_type")),
            "sector": _s(row.get("sector")),
            "salary": _s(row.get("salary")),
            "applications_count": _s(row.get("applications_count")),
            "posted_time_ago": _s(row.get("posted_time_ago")),
            "description": _s(row.get("description")),
            "cluster_id": int(float(cid)) if cid not in ("", "nan") else -1,
            "is_anomaly": int(float(ia)) if ia not in ("", "nan") else 0,
            "score": float(hybrid[pos]),
            "title_score": float(title_score[pos]),
            "loc_score": float(loc_score[pos]),
            "exp_score": float(exp_score[pos]),
            "job_posting_url": _s(row.get("job_posting_url")),
            "application_url": _s(row.get("application_url")),
        })
    return results, intent


def _build_context(prompt, matches, intent):
    lines = [f"USER PROMPT: {prompt}",
             f"PARSED: role='{(intent or {}).get('role_en', '')}' "
             f"city='{(intent or {}).get('city', '')}' exp_rank={(intent or {}).get('experience_rank', 0)}"]
    if (intent or {}).get("more_options"):
        lines.append("NOTE: the user asked for MORE options of the SAME previous search - these are the "
                     "NEXT-best matches after the ones already shown. Open with something like "
                     "'הנה עוד אפשרויות עבורך' instead of a generic greeting.")
    lines += ["", "TOP SCRAPED MATCHES (live LinkedIn / Israel):"]
    for m in matches:
        desc = (m.get("description") or "")[:500]
        lines.append(
            f"\n#{m['rank']} {m['title']} @ {m['company']} ({m['location']})\n"
            f"   experience_level={m.get('experience_level','')} | contract={m.get('contract_type','')} | "
            f"sector={m.get('sector','')} | salary={m.get('salary','')} | "
            f"applicants={m.get('applications_count','')} | posted={m.get('posted_time_ago','')} | "
            f"segment={m['cluster_id']} | is_anomaly={m.get('is_anomaly', 0)} | "
            f"in_requested_location={m.get('loc_match', True)}\n"
            f"   description: {desc}\n"
            f"   url: {m['job_posting_url']}"
        )
    return "\n".join(lines)


def generate_reply(prompt, matches, intent):
    """Return {'intro','jobs':{rank:blurb},'tip'} for the interleaved blurb+card layout."""
    import agent_runner
    context = _build_context(prompt, matches, intent)
    return matching.reply_json(agent_runner.claude_client, agent_runner.LLM_MODEL,
                               context, SYSTEM_PROMPT_SCRAPE, max_tokens=900)


def no_match_message(intent):
    if isinstance(intent, dict) and intent.get("empty"):
        return ("עדיין לא נסרקו משרות. הגדירו כתובת LinkedIn (כל המשרות בישראל) "
                "ולחצו על \"סרוק עכשיו\".")
    role = (intent or {}).get("role_en") or "התפקיד שביקשת"
    return (f"לא נמצאו משרות סרוקות שתואמות ל\"{role}\". נסו לנסח מחדש, "
            "או להרחיב את מאגר הסריקה.")
