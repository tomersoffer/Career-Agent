# =====================================================================
# AGENTIC TOOL-CALLING LOOP  (OpenAI native function-calling)
# ---------------------------------------------------------------------
# Unlike parse_query() — which makes ONE classification call and lets Python code
# decide what to run — this module gives the MODEL the tools and lets IT decide
# which to call, with what arguments, and whether to chain several before
# answering. The model issues real tool_calls; we execute them against the same
# deterministic engine (get_recommendations / analytics / advice) and feed the
# results back, looping until the model produces a final Hebrew reply.
#
# Tools (5): search_jobs, get_job_stat, more_results, give_career_advice, small_talk.
# Requires LLM_BACKEND=openai (uses the OpenAI tools API directly).
# =====================================================================
import os
import json

import agent_runner as ag
import analytics
import matching

MAX_HOPS = 4   # bounded act->observe->act loop (multi-step agency, with a stop)

# ---- the OpenAI tool schemas the model may call ----
_STAT_METRICS = list(analytics.FUNCTIONS)   # salary_stats, top_skills, count_jobs, ...

TOOLS = [
    {"type": "function", "function": {
        "name": "search_jobs",
        "description": "Search the US jobs dataset for roles matching a job title, optionally "
                       "filtered by US state, seniority and remote. Returns ranked job matches "
                       "(with anomaly flags). Use when the user wants to FIND/APPLY to jobs.",
        "parameters": {"type": "object", "properties": {
            "role": {"type": "string", "description": "Job title in English, lowercase, e.g. 'data analyst'."},
            "state": {"type": "string", "description": "2-letter US state code (e.g. 'NY'). Omit if no location was given."},
            "seniority": {"type": "string",
                          "enum": ["intern", "entry", "associate", "mid-senior", "director", "executive"],
                          "description": "Optional seniority level."},
            "remote": {"type": "boolean", "description": "Optional: restrict to remote jobs."},
        }, "required": ["role"]},
    }},
    {"type": "function", "function": {
        "name": "get_job_stat",
        "description": "Compute a STATISTIC over the jobs dataset (salary, counts, top skills/"
                       "locations/industries, remote share, experience breakdown). Use for "
                       "data questions like 'average salary', 'how many', 'most common skills'.",
        "parameters": {"type": "object", "properties": {
            "metric": {"type": "string", "enum": _STAT_METRICS,
                       "description": "Which statistic to compute."},
            "role": {"type": "string", "description": "Optional job title filter (English, lowercase)."},
            "state": {"type": "string", "description": "Optional 2-letter US state code."},
            "experience": {"type": "string",
                           "enum": ["intern", "entry", "associate", "mid-senior", "director", "executive"],
                           "description": "Optional seniority filter."},
            "remote": {"type": "boolean", "description": "Optional remote filter."},
        }, "required": ["metric"]},
    }},
    {"type": "function", "function": {
        "name": "more_results",
        "description": "Return the NEXT page of jobs for the user's PREVIOUS search (paging). "
                       "Use only when the user asks for 'more'/'עוד' of the same search.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "give_career_advice",
        "description": "Get grounding context (jobs currently on screen, conversation) to answer "
                       "open career guidance — interviews, CVs, skills to learn, salary negotiation, "
                       "career paths. Use for advice, NOT for listing jobs.",
        "parameters": {"type": "object", "properties": {
            "topic": {"type": "string", "description": "Short summary of what the user is asking about."},
        }, "required": ["topic"]},
    }},
    {"type": "function", "function": {
        "name": "small_talk",
        "description": "Handle greetings, thanks, or questions about what the agent can do. "
                       "Returns the agent's capabilities so you can reply briefly.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "start_profile_interview",
        "description": "Launch the guided profile-builder interview, which asks the user ONE "
                       "question at a time (role, then experience, then location, then skills). "
                       "Call this when the user has NO CV and wants to build a profile together, "
                       "or wants personalized job matching but has not given a concrete role yet. "
                       "Pass any details the user already mentioned so those steps are pre-filled. "
                       "Do NOT ask the user for several profile fields yourself — call this instead.",
        "parameters": {"type": "object", "properties": {
            "role": {"type": "string", "description": "Target role in English, if the user named one."},
            "city": {"type": "string", "description": "City, if mentioned."},
            "state": {"type": "string", "description": "2-letter US state code, if mentioned."},
            "seniority": {"type": "string",
                          "enum": ["intern", "entry", "associate", "mid-senior", "director", "executive"],
                          "description": "Seniority, if mentioned."},
            "years": {"type": "integer", "description": "Years of experience, if mentioned."},
        }},
    }},
]

AGENT_SYSTEM = (
    "You are 'סוכן המשרות שלך', an action-oriented career agent for a US jobs dataset. You ALWAYS "
    "reply IN HEBREW, very short (1-2 sentences per point), plain everyday words, no emoji except "
    "the '⚠️' risk marker, no flowery filler.\n"
    "SCOPE — you handle ONLY jobs and careers (job search, the jobs dataset, career advice, CVs, "
    "interviews, skills, salaries, the candidate's profile). If the user asks about ANYTHING else "
    "(date/time, recipes, cooking, news, sports/World-Cup, shopping, general knowledge, trivia, "
    "etc.) you MUST NOT answer it and MUST NOT call a tool — reply with ONE short Hebrew sentence "
    "that politely declines and steers back, e.g. 'אני עוזר רק בענייני עבודה וקריירה — רוצה שנחפש "
    "משרות או שנבנה פרופיל יחד?'. Never provide the off-topic answer even if you know it.\n"
    "TOOLS — decide which to call and with what arguments; you may chain several before answering "
    "(e.g. search, see results, then refine):\n"
    "(1) search_jobs — to FIND jobs by role/location/etc.; never invent listings.\n"
    "(2) get_job_stat — for statistics/numbers; never invent figures, use only what it returns.\n"
    "(3) give_career_advice — for open career guidance (interviews, CVs, skills, salary).\n"
    "(4) small_talk — for greetings/thanks/'what can you do'.\n"
    "(5) start_profile_interview — when the user has NO CV and wants to build a profile together, "
    "or wants personalized matching but hasn't named a concrete role yet. This launches a guided "
    "interview that asks ONE question at a time, so NEVER ask the user for several profile fields "
    "yourself (role, experience, location, skills) in one message — call this tool instead. If the "
    "user already names a concrete role/location to search right now, prefer search_jobs.\n"
    "If a returned job has is_anomaly=1, start that job's blurb with '⚠️ התראת סיכון:' and note it's "
    "a statistical outlier to verify. When you have what you need, write the final Hebrew reply with "
    "NO further tool calls. If a search returns nothing, say so honestly and suggest refining the "
    "role or location."
)


# ---------------------------------------------------------------------
# tool executors — each returns a small JSON-able fact for the model,
# and records side effects (matches/state/mode) in `state`.
# ---------------------------------------------------------------------
def _intent_from(role, state, seniority, remote):
    """Build a parse_query-shaped intent from the model's tool arguments."""
    st = (state or "").strip().upper()
    st = st if st in ag.VALID_STATES else ""
    rank = ag.SENIORITY_RANK.get((seniority or "").strip().lower(), 0)
    return {
        "role_en": (role or "").strip().lower(), "synonyms": [],
        "query_text": (role or "").strip().lower(),
        "city": "", "state": st, "nearby_states": [], "years": None,
        "seniority": (seniority or "").strip().lower(), "experience_rank": rank,
        "nationwide": st == "", "used_state": None, "is_geo_fallback": False,
        "no_match": False, "is_search": True, "wants_more": False,
        "intent": "job_search", "mode": "search",
        "remote": bool(remote) if isinstance(remote, bool) else None,
    }


def _slim(matches):
    return [{"rank": m["rank"], "title": m["title"], "company": str(m.get("company", "")),
             "location": m["location"], "salary_band": m["salary_band"],
             "experience_level": m["experience_level"], "is_anomaly": m["is_anomaly"],
             "in_requested_location": m.get("loc_match", True)} for m in matches]


def _exec(name, args, st, last_q, offset, screen):
    if name == "search_jobs":
        intent = _intent_from(args.get("role"), args.get("state"),
                              args.get("seniority"), args.get("remote"))
        matches, intent = ag.get_recommendations(args.get("role", ""), top_n=3, intent=intent)
        st["matches"], st["mode"] = matches, "search"
        st["state"] = intent.get("state") or None
        st["query"] = args.get("role", "")
        if not matches:
            return {"matches": [], "note": "no jobs cleared the relevance threshold"}
        return {"matches": _slim(matches)}

    if name == "more_results":
        if not last_q:
            return {"matches": [], "note": "no previous search to page"}
        matches, _ = ag.get_recommendations(last_q, top_n=3, offset=offset or 3)
        st["matches"], st["mode"], st["query"] = matches, "more", last_q
        return {"matches": _slim(matches), "note": "next page of the previous search"}

    if name == "get_job_stat":
        metric = str(args.get("metric", "")).strip()
        fn = analytics.FUNCTIONS.get(metric)
        if fn is None:
            return {"error": "unknown metric"}
        params = {k: args[k] for k in ("role", "state", "experience", "remote") if k in args}
        fact = fn(ag.df, params, ag.ANALYTICS_CTX)
        st["mode"] = "data_question"
        st["state"] = (args.get("state") or "").strip().upper() or None
        return {"fact": fact}

    if name == "give_career_advice":
        st["mode"] = "advice"
        return {"on_screen_jobs": [{"title": m.get("title"), "company": m.get("company"),
                                    "experience_level": m.get("experience_level")} for m in (screen or [])],
                "instruction": "Answer the career question briefly in Hebrew, grounded in the "
                               "conversation and these on-screen jobs. Do not invent specific jobs."}

    if name == "start_profile_interview":
        # hand off to the deterministic interview (cv/interview.py) which asks ONE
        # question per turn; app.py turns this signal into the interview-kickoff response.
        st["mode"] = "interview_kickoff"
        yrs = args.get("years")
        st["seed"] = {
            "role_en": (args.get("role") or "").strip().lower(), "synonyms": [],
            "city": (args.get("city") or "").strip(),
            "state": (args.get("state") or "").strip().upper(),
            "years": int(yrs) if isinstance(yrs, (int, float)) and not isinstance(yrs, bool) else None,
            "seniority": (args.get("seniority") or "").strip().lower(),
        }
        return {"status": "interview_started"}

    if name == "small_talk":
        st["mode"] = "chat"
        return {"capabilities": "קורא קו\"ח, מתאים משרות לפרופיל, מדרג אותן ושולח להגיש; מחפש "
                                "במאגר של כ-119 אלף משרות בארה\"ב.",
                "instruction": "Reply in short Hebrew; if relevant, guide the user to upload a CV "
                               "or build a short profile together."}

    return {"error": "unknown tool"}


def run_agent_turn(prompt, history=None, last_q="", offset=0, screen_matches=None):
    """Drive the multi-step tool-calling loop. Returns the same response shape the
    old /api/chat dispatch produced, plus a `tool_trace` for inspection."""
    if ag.claude_client is None:
        return {"matches": [], "intro": "השירות אינו זמין כרגע.", "tip": "", "reply": "",
                "mode": "chat", "state": None, "query": None, "tool_trace": []}
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "").strip())

    messages = [{"role": "system", "content": AGENT_SYSTEM}]
    for t in (history or [])[-8:]:
        role = "assistant" if t.get("role") in ("agent", "assistant") else "user"
        messages.append({"role": role, "content": str(t.get("content", ""))[:1000]})
    messages.append({"role": "user", "content": prompt})

    st = {"matches": [], "mode": "chat", "state": None, "query": None}
    trace, final = [], ""

    for _hop in range(MAX_HOPS):
        resp = client.chat.completions.create(
            model=ag.LLM_MODEL, messages=messages, tools=TOOLS,
            tool_choice="auto", max_completion_tokens=1400,
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            final = msg.content or ""
            break
        messages.append({
            "role": "assistant", "content": msg.content or None,
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                           for tc in msg.tool_calls],
        })
        for tc in msg.tool_calls:
            try:
                a = json.loads(tc.function.arguments or "{}")
            except Exception:
                a = {}
            result = _exec(tc.function.name, a, st, last_q, offset, screen_matches)
            trace.append({"tool": tc.function.name, "args": a, "result": result})
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(result, ensure_ascii=False)})
        if st["mode"] == "interview_kickoff":
            # don't ask the model for a reply — the guided interview owns the next turn
            return {"matches": [], "intro": "", "tip": "", "reply": "",
                    "mode": "interview_kickoff", "seed": st.get("seed", {}),
                    "state": None, "query": None, "tool_trace": trace}
    else:
        final = final or "לא הצלחתי להשלים את הבקשה — נסה לנסח אחרת."

    return {"matches": st["matches"], "intro": final, "tip": "", "reply": final,
            "mode": st["mode"], "state": st["state"], "query": st["query"],
            "tool_trace": trace}
