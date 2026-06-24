# =====================================================================
# SMART CAREER NAVIGATOR - CONVERSATIONAL AGENT (Natural-Language Interface)
# ---------------------------------------------------------------------
# 100% natural-language: the user types one free prompt (e.g.
# "I am looking for a Senior Data Analyst position in NY").
# In-memory RAG: a hard Geo filter (state parsed from the prompt) + a
# dual-component Cosine match on JOB TITLE and EXPERIENCE LEVEL — then
# Claude (Anthropic) writes a tailored Hebrew advisor reply.
# (The Jaccard skill index was deprecated in favour of this NL pipeline.)
# =====================================================================

import os
import re
import sys
import json
import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv

import matching   # shared ranking core (title/location/experience gate + reply)

# ensure Hebrew output prints correctly on a Windows (cp1252) console
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "output")

# load local.env early so the backend/model selection below can read it
load_dotenv(os.path.join(BASE, "..", "local.env"))

# LLM_BACKEND switches the live model between Anthropic (default) and OpenAI, so we can
# A/B test gpt-4.1-mini against Claude on the real flows. OPENAI_MODEL / ANTHROPIC_MODEL
# override the per-backend default. See the client-init section below.
LLM_BACKEND = os.environ.get("LLM_BACKEND", "anthropic").strip().lower()
if LLM_BACKEND == "openai":
    LLM_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini-2026-03-17")
else:
    LLM_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")   # 3.5 retired -> 404

# ---- experience-level vocabulary (experience_rank: 0..6) ----------------------------
# 0 Not specified | 1 Internship | 2 Entry level | 3 Associate
# 4 Mid-Senior level | 5 Director | 6 Executive
# The LLM returns a seniority label; this maps it to the dataset's numeric rank.
SENIORITY_RANK = {"intern": 1, "entry": 2, "associate": 3, "mid-senior": 4,
                  "senior": 4, "director": 5, "executive": 6}

# retrieval thresholds/weights now live in matching.py (shared with the scraped tab)

# =====================================================================
# SECTION 1 - LOAD LOCAL KB & MODELS
# load the gold master, the pre-trained pipeline components, and Claude
# =====================================================================

COLS = ["job_id", "title", "company_name", "location", "job_state",
        "experience_level", "salary_band", "experience_rank", "is_anomaly",
        "cluster_id", "job_posting_url", "application_url",
        "skills", "salary", "remote_allowed", "job_industry"]   # + analytics columns

# prefer the slim + gzipped gold (~7 MB, deploy-friendly); fall back to the raw CSV locally.
GOLD_PATH = os.path.join(OUT_DIR, "gold_linkedin_with_clusters.csv.gz")
if not os.path.exists(GOLD_PATH):
    GOLD_PATH = os.path.join(OUT_DIR, "gold_linkedin_with_clusters.csv")
# Load only columns that actually exist in the file (analytics columns may be absent
# if the gold file hasn't been regenerated yet — they'll be added by the data pipeline).
_available = set(pd.read_csv(GOLD_PATH, nrows=0).columns)
_load_cols = [c for c in COLS if c in _available]
df = pd.read_csv(GOLD_PATH, usecols=_load_cols, low_memory=False)   # .gz auto-detected by pandas
print("Local knowledge base loaded.")
print("Jobs:", len(df), "| anomalies flagged:", int(df["is_anomaly"].sum()))
print("------------------------------")

vectorizer = joblib.load(os.path.join(OUT_DIR, "tfidf_vectorizer.joblib"))
kmeans = joblib.load(os.path.join(OUT_DIR, "kmeans_model.joblib"))
scaler = joblib.load(os.path.join(OUT_DIR, "feature_scaler.joblib"))
print("Loaded models: tfidf_vectorizer, kmeans_model, feature_scaler")

# vectorise the TITLE and EXPERIENCE-LEVEL columns once (short strings -> fast),
# reusing the fitted vocabulary so the prompt lives in the same vector space.
title_matrix = vectorizer.transform(df["title"].fillna("").astype(str).str.lower())
print("Title matrix:", title_matrix.shape)

# flat list of job titles aligned row-for-row with title_matrix.
# cv.titles.suggest_titles() uses this to map cosine-nearest rows back to real title strings.
TITLES = df["title"].fillna("").astype(str).tolist()

# the CLOSED set of states that actually appear in the data - we ground the LLM's
# location output against this so it can never resolve to a state with zero jobs.
VALID_STATES = sorted(s for s in df["job_state"].dropna().unique() if isinstance(s, str) and s.strip())
print("Valid states in data:", len(VALID_STATES))
print("------------------------------")

import analytics   # deterministic data-question engine (leaf module)

# Shared analytics context: the fitted vectorizer + the title matrix (row-aligned with df)
# + the closed state set + the seniority map. Passed into analytics.* on every data-question.
ANALYTICS_CTX = analytics.Ctx(
    vectorizer=vectorizer,
    title_matrix=title_matrix,
    valid_states=VALID_STATES,
    seniority_rank=SENIORITY_RANK,
)

# initialise the LLM client (key from local.env). `claude_client` keeps its name for the
# call sites, but with LLM_BACKEND=openai it's an OpenAI-backed stand-in (llm_backend.py)
# exposing the same .messages.create() surface — so nothing downstream changes.
claude_client = None
if LLM_BACKEND == "openai":
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
    if OPENAI_API_KEY:
        try:
            from llm_backend import OpenAIClient
            claude_client = OpenAIClient(api_key=OPENAI_API_KEY)
            print("OpenAI client initialised (model:", LLM_MODEL + ")")
        except Exception as e:
            print("OpenAI client init failed:", e)
    else:
        print("No OPENAI_API_KEY found - running LOCAL retrieval only.")
else:
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if ANTHROPIC_API_KEY:
        try:
            import anthropic
            claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            print("Claude client initialised (model:", LLM_MODEL + ")")
        except Exception as e:
            print("Claude client init failed:", e)
    else:
        print("No ANTHROPIC_API_KEY found - running LOCAL retrieval only.")
print("------------------------------")

# =====================================================================
# SECTION 2 - CONVERSATIONAL IN-MEMORY RAG PIPELINE
# parse the prompt -> Geo filter -> dual Cosine (Title + Experience)
# =====================================================================

# ---------------------------------------------------------------------
# 2.1  QUERY UNDERSTANDING (LLM-FIRST) - Claude reads the free prompt
#      BEFORE any ML runs and returns structured intent: role + English
#      synonyms, the city the user named, the inferred state (grounded to
#      states that EXIST in the data), proximity-ranked nearby states for
#      a geographic fallback, plus years/seniority.
# ---------------------------------------------------------------------

PARSE_SYSTEM = (
    "You are a query-understanding module for a US job-search engine. The user's message may be "
    "in Hebrew or English, and they usually name a CITY rather than a state - infer the US state "
    "from the city. Respond with ONLY a compact JSON object (no prose, no code fences) with "
    "EXACTLY these keys:\n"
    '  "role_en"       - desired job role/title in ENGLISH, lowercase (e.g. "data analyst"); "" if none.\n'
    '  "role_synonyms" - up to 4 common alternative English titles for that role, each a SPECIFIC, '
    'recognizable job title (e.g. for "data analyst": ["business intelligence analyst","reporting '
    'analyst","bi analyst"]). Do NOT include vague generic words on their own such as "specialist", '
    '"professional", "expert", "coordinator", "associate". Use [] if none, or if the role is not a '
    "plausible real-world job.\n"
    '  "city"          - the city/place the user named, written in English; "" if none.\n'
    '  "state"         - the 2-letter US state code for that city/location. It MUST be one of these '
    "codes that exist in our data: " + ", ".join(VALID_STATES) + '. Use "" if no location was given.\n'
    '  "nearby_states" - 3 to 5 OTHER codes from that same list, ranked by GEOGRAPHIC proximity to '
    "the chosen state (closest first); [] if no location.\n"
    '  "years"         - integer years of experience mentioned, or null.\n'
    '  "seniority"     - one of "intern","entry","associate","mid-senior","director","executive", or "".\n'
    '  "mode"          - one of: "search" (find/list jobs by role/title/skill/location/'
    "experience), \"data_question\" (asks for a COMPUTED statistic over many jobs — averages, "
    "counts, most-common skills/industries/locations, remote share — e.g. \"what's the average "
    "data analyst salary in NY\", \"how many remote QA jobs are there\", \"most common skills for "
    "a PM\"), \"advice\" (open career guidance: interviews, CVs, skills to learn, salary "
    "negotiation, career paths — NOT asking to list jobs), \"smalltalk\" (greeting/thanks/what can "
    "you do), \"more\" (asks for MORE of the same previous search). When unsure between search and "
    "data_question, choose \"search\".\n"
    '  "is_search"     - true if the user is asking to FIND/SEARCH for jobs, or names a role/title/skill/'
    "location/experience to search by; false for a greeting, small talk, thanks, or a question about you or "
    "the system (e.g. \"hi\", \"שלום\", \"מה אתה יכול לעשות?\", \"תודה\").\n"
    '  "wants_more"    - true ONLY if the user is asking for MORE / OTHER / additional / next options of the '
    "SAME previous search, WITHOUT naming a new role or location (e.g. \"more\", \"show me more\", \"what "
    "else\", \"next\", \"עוד\", \"עוד אפשרויות\", \"אפשרויות נוספות\", \"תן עוד\"); otherwise false.\n"
    '  "intent"        - one of "job_search" or "info". "job_search" = the user wants to FIND or APPLY to jobs '
    "for themselves, or names a role/title/skill/location they want to WORK in (e.g. \"data analyst in NY\", "
    "\"אני מחפש עבודה\", \"find me marketing roles\"). \"info\" = an informational/domain question (salary, "
    "market), a question about you or the system, a greeting, thanks, or small talk - EVEN IF a role is named "
    "(e.g. \"how much does a data analyst make?\", \"מה אתה יודע לעשות?\", \"תודה\")."
)


def _llm_parse(user_prompt):
    """Ask Claude to extract structured intent. Returns a dict (may raise on failure)."""
    msg = claude_client.messages.create(
        model=LLM_MODEL,
        max_tokens=400,
        system=[{"type": "text", "text": PARSE_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_prompt}],
    )
    raw = "".join(b.text for b in msg.content if b.type == "text")
    start, end = raw.find("{"), raw.rfind("}")
    return json.loads(raw[start:end + 1])


def _years_to_rank(years):
    if years is None:
        return 0
    if years < 2:
        return 2     # Entry level
    if years < 5:
        return 3     # Associate
    if years < 10:
        return 4     # Mid-Senior level
    return 5         # Director


def parse_query(user_prompt):
    """LLM-first intent extraction. Locations are grounded to states present in the data."""
    role_en, synonyms, city, state, nearby, years, seniority = "", [], "", "", [], None, ""
    is_search, wants_more, mode = None, False, ""
    intent = None

    if claude_client is not None:
        try:
            d = _llm_parse(user_prompt)
            role_en = (d.get("role_en") or "").strip().lower()
            synonyms = [str(s).strip().lower() for s in (d.get("role_synonyms") or []) if str(s).strip()]
            city = (d.get("city") or "").strip()
            state = (d.get("state") or "").strip().upper()
            nearby = [str(s).strip().upper() for s in (d.get("nearby_states") or [])]
            yrs = d.get("years")
            years = int(yrs) if isinstance(yrs, (int, float)) and not isinstance(yrs, bool) else None
            seniority = (d.get("seniority") or "").strip().lower()
            is_search = d.get("is_search")
            wants_more = bool(d.get("wants_more"))
            intent = (d.get("intent") or "").strip().lower() or None
            mode = str(d.get("mode") or "").strip().lower()
        except Exception as e:
            print("LLM parse failed:", e)
    else:
        print("No Claude client - query understanding disabled (set ANTHROPIC_API_KEY).")

    # GROUND locations against the states that actually exist in the dataset
    if state not in VALID_STATES:
        state = ""
    nearby = [s for s in nearby if s in VALID_STATES and s != state]

    rank = SENIORITY_RANK.get(seniority, 0) or _years_to_rank(years)
    # the text we actually feed the TF-IDF cosine: role + its synonyms (recall boost).
    # if the LLM is unavailable we degrade to the raw prompt so the app still does something.
    query_text = " ".join([role_en] + synonyms).strip() or user_prompt.lower()
    # is this a search request, or just chat? fall back to a heuristic if the LLM didn't say.
    if is_search is None:
        is_search = bool(role_en or city or state)

    # Normalize mode: validate against whitelist; fall back from legacy signals.
    VALID_MODES = ("search", "data_question", "advice", "smalltalk", "more")
    if mode not in VALID_MODES:
        # fall back from the legacy signals: searchy -> search, else smalltalk
        mode = "search" if is_search else "smalltalk"
    # `wants_more` only pages a prior SEARCH — it must not hijack an advice/data question
    # that happens to contain "more" (e.g. "give me more advice on that").
    if wants_more and mode == "search":
        mode = "more"

    return {
        "role_en": role_en,
        "synonyms": synonyms,
        "query_text": query_text,
        "city": city,
        "state": state,
        "nearby_states": nearby,
        "years": years,
        "seniority": seniority,
        "experience_rank": rank,
        "nationwide": state == "",
        "used_state": None,
        "is_geo_fallback": False,
        "no_match": False,
        "is_search": bool(is_search),
        "wants_more": bool(wants_more),
        # job_search => route to the seeded interview; info => spoken answer.
        # Default from is_search when the LLM didn't classify.
        "intent": intent if intent in ("job_search", "info")
                  else ("job_search" if is_search else "info"),
        "mode": mode,
    }


# ---------------------------------------------------------------------
# 2.1b  CV-DRIVEN INTENT - builds the SAME intent dict shape that
#        parse_query() returns, using a structured CV profile + the user's
#        explicitly-picked title chips. Picked titles WIN over CV inference.
#        Called by /api/chat and /api/scrape-search when the browser posts
#        cv_text / profile / picked_titles alongside its regular prompt.
# ---------------------------------------------------------------------

def intent_from_cv(profile: dict, picked_titles: list, typed_query: str = "") -> dict:
    """
    Build a parse_query-compatible intent dict from a CV profile + picked titles.

    Parameters
    ----------
    profile       : dict returned by cv.profile.build_profile()
                    expected keys: skills, titles_held, seniority, years, city, state, domains
    picked_titles : list[str] — title chips the user selected; wins over CV inference.
                    Falls back to profile['titles_held'] when empty.
    typed_query   : str — optional free text typed alongside the chips (merged into query_text).

    Returns
    -------
    dict with EXACTLY the same keys as parse_query():
      role_en, synonyms, query_text, city, state, nearby_states, years, seniority,
      experience_rank, nationwide, used_state, is_geo_fallback, no_match,
      is_search, wants_more
    """
    # ---- TITLES: picked chips win; fall back to CV-inferred titles_held ----
    titles_source = picked_titles if picked_titles else (profile.get("titles_held") or [])
    # role_en: first picked/inferred title; synonyms: the rest
    role_en = titles_source[0].strip().lower() if titles_source else ""
    synonyms = [t.strip().lower() for t in titles_source[1:] if t.strip()]
    # query_text: join all picked titles + optional typed query (recall boost)
    parts = [t.strip().lower() for t in titles_source if t.strip()]
    if typed_query.strip():
        parts.append(typed_query.strip().lower())
    query_text = " ".join(parts) or typed_query.lower()

    # ---- EXPERIENCE: prefer explicit seniority label; fall back to years ----
    seniority = (profile.get("seniority") or "").strip().lower()
    years_raw = profile.get("years")
    try:
        years = int(years_raw) if years_raw is not None else None
    except (ValueError, TypeError):
        years = None
    experience_rank = SENIORITY_RANK.get(seniority, 0) or _years_to_rank(years)

    # ---- LOCATION: ground profile state to the closed VALID_STATES set ----
    raw_state = (profile.get("state") or "").strip().upper()
    state = raw_state if raw_state in VALID_STATES else ""

    # nearby_states: not inferred from CV (no LLM call here); left empty.
    # get_recommendations() handles nationwide fallback when state == "".
    nearby_states: list = []

    return {
        "role_en": role_en,
        "synonyms": synonyms,
        "query_text": query_text,
        "city": (profile.get("city") or "").strip(),
        "state": state,
        "nearby_states": nearby_states,
        "years": years,
        "seniority": seniority,
        "experience_rank": experience_rank,
        "nationwide": state == "",
        "used_state": None,
        "is_geo_fallback": False,
        "no_match": False,
        "is_search": bool(role_en or state),
        "wants_more": False,
    }


# ---------------------------------------------------------------------
# 2.1c  AMENDABLE SEARCH CRITERIA (Flow B) — persist the search levers and
#        let the user amend them across turns. criteria <-> intent round-trip,
#        plus a delta-based refine helper built on cv/state_delta.
# ---------------------------------------------------------------------

def intent_from_criteria(criteria):
    """Build a parse_query-shaped intent dict from amendable search criteria.

    criteria keys: roles (list), city, state, seniority, years, remote.
    Returns every key get_recommendations() reads, plus "remote".
    """
    criteria = criteria or {}
    roles = [str(r).strip().lower() for r in (criteria.get("roles") or []) if str(r).strip()]
    role_en = roles[0] if roles else ""
    synonyms = roles[1:]
    seniority = (criteria.get("seniority") or "").strip().lower()
    yrs = criteria.get("years")
    years = int(yrs) if isinstance(yrs, (int, float)) and not isinstance(yrs, bool) else None
    rank = SENIORITY_RANK.get(seniority, 0) or _years_to_rank(years)
    state = (criteria.get("state") or "").strip().upper()
    state = state if state in VALID_STATES else ""
    return {
        "role_en": role_en, "synonyms": synonyms,
        "query_text": " ".join(roles) or role_en,
        "city": (criteria.get("city") or "").strip(), "state": state, "nearby_states": [],
        "years": years, "seniority": seniority, "experience_rank": rank,
        "nationwide": state == "", "used_state": None, "is_geo_fallback": False,
        "no_match": False, "is_search": bool(role_en or state), "wants_more": False,
        "remote": bool(criteria.get("remote")),
    }


def criteria_from_intent(intent):
    """Extract the amendable search levers from a parse_query/intent_from_cv intent."""
    intent = intent or {}
    roles = [r for r in ([intent.get("role_en")] + list(intent.get("synonyms") or [])) if r]
    return {"roles": roles, "city": intent.get("city", "") or "",
            "state": intent.get("state", "") or "", "seniority": intent.get("seniority", "") or "",
            "years": intent.get("years"), "remote": bool(intent.get("remote", False))}


def refine_search_criteria(criteria, history, user_msg, claude_client, model):
    """Amend persisted search criteria from one user message. `changed` is False when the
    message requested no criteria change (empty delta) — the caller then falls through to
    normal handling (advice / paging / new search)."""
    from cv import state_delta, prompts          # local import: cv is a sibling package
    system = prompts.delta_system(prompts.SEARCH_LABELS, prompts.search_extra_rules())
    d = state_delta.extract_delta(criteria or {}, history, user_msg,
                                  state_delta.SEARCH_SCHEMA, system, claude_client, model)
    changed = bool(d["set"] or d["add"] or d["remove"])
    merged = state_delta.apply_delta(criteria or {}, d, state_delta.SEARCH_SCHEMA)
    # `blocked` = the user asked for a non-US location (US-only dataset); the location is
    # left unchanged and the reply explains it, while other changes still apply.
    return {"criteria": merged, "reply": d["reply"], "ready": d["ready"],
            "changed": changed, "blocked": bool(d.get("blocked", False))}


# ---------------------------------------------------------------------
# 2.2  RETRIEVAL - the SHARED 3-dimension search (matching.py): TITLE,
#      LOCATION and EXPERIENCE are independent, combinable filters.
#      LOCATION here is scored: the user's state = 1.0, proximity-ranked
#      nearby states get a lower score (geographic fallback, as a score).
# ---------------------------------------------------------------------

def get_recommendations(user_prompt, top_n=3, offset=0, intent=None):
    intent = intent or parse_query(user_prompt)
    state = intent["state"]

    # ---- optional REMOTE pre-filter (hard filter when the user asked for remote) ----
    base = df.index
    if intent.get("remote"):
        base = base[df.loc[base, "remote_allowed"].astype(bool)]

    # ---- LOCATION as a score: home state 1.0, nearby states lower (closest highest) ----
    if state:
        loc_of = {state: 1.0}
        for i, s in enumerate(intent["nearby_states"]):
            loc_of[s] = max(0.40, 0.62 - 0.04 * i)
        idx = base[df.loc[base, "job_state"].isin([state] + intent["nearby_states"])]
        loc_score = df.loc[idx, "job_state"].map(loc_of).fillna(0.0).to_numpy()
        loc_specified = True
    else:
        idx = base
        loc_score = np.zeros(len(idx))
        loc_specified = False
    if len(idx) == 0:
        intent["no_match"] = True
        return [], intent

    # ---- the three independent dimension scores ----
    title_score, role_specified = matching.title_scores(intent, vectorizer, title_matrix[idx])
    ranks = df.loc[idx, "experience_rank"].to_numpy()
    exp_score, exp_specified = matching.experience_scores(ranks, intent["experience_rank"])

    # ---- shared per-dimension AND gate + hybrid ranking ----
    order, hybrid = matching.select(title_score, role_specified, loc_score, loc_specified,
                                    exp_score, exp_specified, ranks=ranks, top_n=offset + top_n)
    order = order[offset:]                       # skip already-shown matches ("more options" / pagination)
    if len(order) == 0:
        intent["no_match"] = True
        return [], intent

    results, home = [], 0
    for rank, pos in enumerate(order, start=offset + 1):
        i = idx[pos]
        row = df.loc[i]
        if row["job_state"] == state:
            home += 1
        results.append({
            "rank": rank,
            "title": row["title"],
            "company": row["company_name"],
            "location": row["location"],
            "loc_match": (not state) or (row["job_state"] == state),   # in the user's requested state?
            "salary_band": int(row["salary_band"]),
            "experience_rank": int(row["experience_rank"]),
            "experience_level": row["experience_level"],
            "remote_allowed": int(row["remote_allowed"]),
            "is_anomaly": int(row["is_anomaly"]),
            "cluster_id": int(row["cluster_id"]),
            "score": float(hybrid[pos]),
            "title_score": float(title_score[pos]),
            "exp_score": float(exp_score[pos]),
            "job_posting_url": row["job_posting_url"],
            "application_url": row["application_url"],
        })
    intent["used_state"] = df.loc[idx[order[0]], "job_state"] if state else None
    intent["is_geo_fallback"] = bool(state) and home == 0   # all results came from nearby states
    return results, intent


def explain_no_match(intent):
    """A short, honest Hebrew message for when nothing clears the relevance floor anywhere."""
    role = intent.get("role_en") or "התפקיד שביקשת"
    if intent.get("state"):
        loc = f"ב{intent['city']}" if intent.get("city") else f"במדינה {intent['state']}"
        loc += " או במדינות הסמוכות"
    else:
        loc = "בארה\"ב"
    return (f"לא מצאתי משרות שתואמות מספיק טוב ל\"{role}\" {loc}. "
            "נסו לנסח מחדש את התפקיד (למשל \"data analyst\" או \"software engineer\") "
            "או לציין מיקום אחר.")

# =====================================================================
# SECTION 3 - CLAUDE SONNET INTEGRATION (Hebrew executive career agent)
# =====================================================================

SYSTEM_PROMPT = (
    "You are an ACTION-ORIENTED job-application agent (not a passive advisor): your goal is to get the user "
    "applying to the right roles FAST. For each job you receive, write a SHORT, energetic blurb on why it "
    "fits and why it's worth applying NOW. Each job already has a card under your blurb (title, company, "
    "location, salary band, experience level, apply button), so you MAY name the company but DO NOT repeat "
    "the location, salary or experience level, and DO NOT output links. Everything IN HEBREW. "
    "Return ONLY a JSON object, no text around it, exactly: "
    "{\"intro\": <one short, action-focused opener>, "
    "\"jobs\": {\"1\": <1-2 short sentences for rank #1>, \"2\": <...>, ... one entry per provided rank}, "
    "\"tip\": <one short, concrete next-step tip (e.g. which to apply to first)>}. "
    "Keep every field tight and skimmable. Do NOT use decorative emoji (the only symbol allowed is the "
    "'⚠️' risk marker described below). Write every field in PLAIN, SIMPLE, "
    "EASY-TO-READ Hebrew — everyday words, short and direct, NO flowery filler. "
    "RISK PROFILING — this is the agent's data-driven value-add; apply per job when relevant: "
    "• ANOMALY: if a job has is_anomaly == 1, START that job's blurb with '⚠️ התראת סיכון: ' and explain in "
    "one sentence that the engine flagged it as a statistical OUTLIER — it sits far from its market-cluster "
    "centroid (an unusual feature combination, e.g. a salary extreme for the required experience), which may "
    "signal a suspicious or mistaken listing OR a rare genuine opportunity, so verify the details before applying. "
    "• LOCATION: if a job has in_requested_location == false, add '📍 שים לב: המשרה מחוץ למיקום שביקשת' to that "
    "job's blurb. If the engine NOTE says the results are a geographic fallback, say so kindly in the intro."
)

SALARY_LABELS = {0: "Not listed", 1: "<$40K", 2: "$40-60K", 3: "$60-80K", 4: "$80-100K",
                 5: "$100-150K", 6: "$150-200K", 7: "$200K+"}


def build_context(user_prompt, matches, intent=None):
    lines = [f"USER PROMPT: {user_prompt}"]
    if intent:
        asked = intent.get("city") or intent.get("state") or "NONE - searched the whole USA"
        lines.append(
            "PARSED INTENT (how the engine understood the request): "
            f"role='{intent.get('role_en', '')}' synonyms={intent.get('synonyms', [])} | "
            f"asked_location={asked} | results_state={intent.get('used_state') or 'all USA'} | "
            f"target_experience_rank={intent.get('experience_rank', 0)} "
            "(0=any,1=intern,2=entry,3=associate,4=mid-senior,5=director,6=executive)"
        )
        if intent.get("is_geo_fallback"):
            place = intent.get("city") or intent.get("state")
            lines.append(
                f"NOTE: there were no strong matches in {place} ({intent.get('state')}), so these are "
                f"the geographically CLOSEST available roles, in {intent.get('used_state')}. You MUST "
                "tell the user this clearly and kindly at the start of your reply."
            )
        elif intent.get("nationwide"):
            lines.append("NOTE: no location was specified, so results span the entire USA - "
                         "mention this gently and invite the user to add a city or state.")
        if intent.get("more_options"):
            lines.append("NOTE: the user asked for MORE options of the SAME previous search - these are the "
                         "NEXT-best matches after the ones already shown. Open with something like "
                         "'הנה עוד אפשרויות עבורך' instead of a generic greeting.")
    lines += ["", "TOP MATCHING JOBS (local facts):"]
    for m in matches:
        lines.append(
            f"\n#{m['rank']} {m['title']} @ {m['company']} ({m['location']})\n"
            f"   hybrid_score={m['score']:.3f} (title={m['title_score']:.3f}, "
            f"experience={m['exp_score']:.3f}) | cluster={m['cluster_id']} | "
            f"is_anomaly={m['is_anomaly']} | in_requested_location={m.get('loc_match', True)}\n"
            f"   experience_level={m['experience_level']} | salary_band={m['salary_band']} "
            f"({SALARY_LABELS[m['salary_band']]})\n"
            f"   job_posting_url: {m['job_posting_url']}\n"
            f"   application_url: {m['application_url']}"
        )
    return "\n".join(lines)


def generate_reply(user_prompt, matches, intent=None):
    """Return {'intro', 'jobs': {rank: blurb}, 'tip'} for the interleaved blurb+card layout."""
    context = build_context(user_prompt, matches, intent)
    return matching.reply_json(claude_client, LLM_MODEL, context, SYSTEM_PROMPT, max_tokens=1000)


CHAT_SYSTEM = (
    "You are 'סוכן המשרות שלך', a direct, action-oriented JOB-SEARCH agent. The user is just CHATTING "
    "(a greeting, small talk, thanks, or a question about what you do) — they have NOT asked to search yet. "
    "Reply IN HEBREW, VERY SHORT — 1.5 sentences MAX, plain everyday words, straight to the point. "
    "Do NOT use emoji. NO flowery openers, NO filler, NO cheerfulness for its own sake. "
    "Use the CONVERSATION history you are given: greet AT MOST once, on the very first message only — NEVER "
    "greet again, and NEVER re-introduce yourself ('אני כאן לעזור...') on later turns; just continue the flow. "
    "UNDERSTAND THE STEP-BY-STEP FLOW and guide the user ONE step at a time: to match jobs you first need their "
    "profile, built EITHER by uploading a CV with the '+ קו״ח' button, OR — if they have no CV — by building one "
    "together through a few short questions IN THIS ORDER: role, then experience, then location, then skills. "
    "When the user AGREES to start (e.g. 'כן', 'בוא נתחיל', 'יאללה') — do NOT greet again; reply with the next "
    "concrete step, exactly: 'כדי להתחיל תעלה את הקו\"ח שלך או במידה ואין לך נבנה לך פרופיל משתמש ביחד'. "
    "When the user says they have NO CV (or don't want to upload one), do NOT just list the options — say you'll "
    "build a profile together and immediately ask the FIRST question: which job/role are they looking for "
    "(e.g. 'נבנה פרופיל יחד — איזה תפקיד אתה מחפש?'). Once they answer one step, move to the next. "
    "You MAY answer general, factual career questions (e.g. typical salary ranges, what a role does, market "
    "demand) briefly and honestly when asked. Do NOT invent, list or describe any specific jobs. If they ask "
    "what you can do, explain briefly that you read their "
    "CV, find matching jobs, rank them to their profile, and send them straight to apply"
    "{tab_hint}. If the message is off-topic (not about jobs or careers), politely steer back to job searching."
)

_TAB_HINT = {
    "dataset": " — here you search a global dataset of ~119K US jobs",
    "scrape":  " — here you search live LinkedIn jobs scraped in Israel",
}


def chat_reply(prompt, tab="dataset", history=None):
    """Free-form, domain-guided conversational reply (no retrieval, no job cards).

    Stateful: the recent CONVERSATION is passed to the model so it does not
    re-greet every turn and can resolve follow-ups like 'כן' against what it
    just asked.
    """
    if claude_client is None:
        return "כדי להתחיל תעלה את הקו\"ח שלך עם '+ קו\"ח', או במידה ואין לך נבנה לך פרופיל משתמש ביחד."
    system = CHAT_SYSTEM.replace("{tab_hint}", _TAB_HINT.get(tab, ""))
    lines = []
    for t in (history or [])[-8:]:
        who = "User" if t.get("role") == "user" else "Agent"
        lines.append(f"{who}: {str(t.get('content', ''))[:500]}")
    lines.append(f"User: {prompt}")
    context = "CONVERSATION:\n" + "\n".join(lines)
    return matching.reply(claude_client, LLM_MODEL, context, system, max_tokens=300)


ADVICE_SYSTEM = (
    "You are 'סוכן המשרות שלך', a professional, expert CAREER ADVISOR replying in HEBREW. The user is "
    "asking for guidance (interviews, CVs, which skills to learn, salary expectations, career "
    "paths) — possibly about a specific job already shown to them. Use the CONVERSATION, the JOBS "
    "ON SCREEN, and the CANDIDATE PROFILE (when given) to make your answer concrete and personal. "
    "Rules: stay strictly within job/career topics; if asked something off-topic, politely steer "
    "back. BE VERY VERY SHORT — 1.5 sentences MAX, no lists, no preamble (the user's professor requires "
    "brevity). Write in PLAIN, SIMPLE, EASY-TO-READ Hebrew: everyday words, straight to the point, NO flowery "
    "openers or filler. Do NOT invent specific job listings or salary figures; speak in general guidance. "
    "Do NOT use emoji."
)


def _advice_context(prompt, history, matches, profile):
    lines = [f"USER MESSAGE: {prompt}"]
    if profile:
        lines.append("CANDIDATE PROFILE: " + json.dumps(profile, ensure_ascii=False))
    if matches:
        lines.append("JOBS ON SCREEN (the user may be referring to these by rank/title):")
        # cap to a handful of jobs and bound each client-supplied field (cost / prompt-abuse)
        for m in matches[:5]:
            def _f(key):
                return str(m.get(key, ""))[:200]
            lines.append(
                f"  #{_f('rank')} {_f('title')} @ {_f('company')} "
                f"({_f('location')}) | exp={_f('experience_level')} | "
                f"salary_band={_f('salary_band')}"
            )
    if history:
        lines.append("CONVERSATION:")
        for t in history[-8:]:
            who = "User" if t.get("role") == "user" else "Agent"
            lines.append(f"  {who}: {str(t.get('content', ''))[:500]}")
    return "\n".join(lines)


def advice_reply(prompt, history=None, matches=None, profile=None):
    """Grounded career-advice reply (no retrieval, no cards). Returns a Hebrew string."""
    if claude_client is None:
        return ("בשמחה אעזור! ספר/י לי קצת על הרקע שלך או על המשרה שמעניינת אותך, "
                "ואכוון אותך — אפשר גם להעלות קו\"ח כדי שאדייק.")
    context = _advice_context(prompt, history or [], matches or [], profile)
    return matching.reply(claude_client, LLM_MODEL, context, ADVICE_SYSTEM, max_tokens=300) \
        or "לא הצלחתי לנסח תשובה כרגע — נסה שוב."


def run_agent(user_prompt, top_n=3):
    print("==============================================")
    print("USER PROMPT:", user_prompt)
    matches, intent = get_recommendations(user_prompt, top_n)
    print("Parsed: role='%s' syn=%s | city=%s state=%s -> used=%s%s | exp_rank=%s | matches=%d" % (
        intent["role_en"], intent["synonyms"], intent["city"] or "-",
        intent["state"] or "none", intent["used_state"] or "all-USA",
        " (GEO-FALLBACK)" if intent["is_geo_fallback"] else "",
        intent["experience_rank"], len(matches)))
    print("==============================================")
    if not matches:
        print(explain_no_match(intent))
        return
    advice = generate_reply(user_prompt, matches, intent) or {}
    print(advice.get("intro", ""))
    for m in matches:
        print(f"\n#{m['rank']} {m['title']} @ {m.get('company', '')}")
        print("  ", advice.get("jobs", {}).get(str(m["rank"]), ""))
    print("\n" + advice.get("tip", ""))
    print("------------------------------")

# =====================================================================
# INTERACTIVE MODE - type one natural-language prompt, repeat.
# `python agent_runner.py --demo` runs a fixed example.
# =====================================================================
if __name__ == "__main__":
    if "--demo" in sys.argv:
        run_agent("I am looking for a Senior Data Analyst position in NY")
    else:
        print("==============================================")
        print(" SMART CAREER NAVIGATOR - type your request (empty line to quit)")
        print(" e.g. 'I am looking for a Senior Data Analyst position in NY'")
        print("==============================================")
        while True:
            try:
                prompt = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt:
                break
            run_agent(prompt)
    print("AGENT RUN COMPLETE.")
