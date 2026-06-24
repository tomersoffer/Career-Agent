# =====================================================================
# SMART CAREER NAVIGATOR - WEB BACKEND (Flask)
# ---------------------------------------------------------------------
# Serves the chat front-end and exposes the In-Memory RAG + Claude agent
# over a tiny JSON API. Importing agent_runner loads the dataset + models
# ONCE at startup (~1-2 min), then every request reuses them.
# Run:  python app.py   ->   open http://127.0.0.1:5000
# =====================================================================

import os
import math
import logging
import threading
import traceback
from flask import Flask, request, jsonify, send_from_directory

# CV upload size cap (5 MB) — reject larger files with a Hebrew error.
CV_MAX_BYTES = 5 * 1024 * 1024
CV_ALLOWED_EXTS = {".pdf", ".docx"}

# quiet the per-request access logs (the 3s status polling spams the terminal);
# our own "[scrape]" lines and the Apify actor logs carry the real signal.
logging.getLogger("werkzeug").setLevel(logging.WARNING)

import agent_runner as ag   # loads gold CSV + joblib models + Claude client at import
import analytics
import agent_loop           # agentic tool-calling loop (the model issues real tool calls)
from cv import converse as cv_converse

# scrape-side package (light imports; the Apify SDK is only loaded on an actual scrape)
from scraper import settings_store, store
from scraper import config as scfg
from scraper import run_pipeline
from scraper import search as scrape_search

FRONTEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
app = Flask(__name__, static_folder=FRONTEND, static_url_path="")
# reject oversized request bodies during streaming (before we read them into memory);
# slightly above the CV cap so our handler can still return the friendly Hebrew message.
app.config["MAX_CONTENT_LENGTH"] = CV_MAX_BYTES + 1024 * 1024

# words meaning "show me more of the same search" — deterministic fallback for the
# CV-driven path, which skips the LLM query-parse that normally detects this.
_MORE_WORDS = ("show me more", "what else", "more", "next",
               "עוד אפשרויות", "אפשרויות נוספות", "תן עוד", "עוד")


def _wants_more(prompt):
    p = (prompt or "").strip().lower()
    return any(w in p for w in _MORE_WORDS)


def _attach_cv_fit(matches, cv_text, profile):
    """Attach the deterministic TF-IDF CV-to-title fit % to each match (spec §fit score).
    No-op when no CV text was posted. Failures degrade to cv_fit=None, never raise."""
    cv_text = (cv_text or "").strip()
    if not cv_text or not matches:
        return
    try:
        from cv import fit as cv_fit
    except Exception:
        return
    for m in matches:
        try:
            fg = cv_fit.fit_and_gap(profile or {}, cv_text,
                                    {"title": m.get("title", ""), "skills": m.get("skills", "")},
                                    ag.vectorizer)
            m["cv_fit"] = round(float(fg["fit_score"]) * 100)
        except Exception:
            m["cv_fit"] = None


_CV_NUDGE = "💡 רוצה שאדרג את המשרות לפי קורות החיים שלך? העלה/י קובץ עם הכפתור + קו״ח."


def _scorecard_line(card, detailed):
    """One Hebrew markdown bullet for a role scorecard."""
    line = f"- **{card['role']}** — התאמה {card['match']}%"
    if card.get("seniority"):
        line += f" · {card['seniority']}"
    if detailed:
        if card.get("have"):
            line += " · ✓ " + ", ".join(card["have"])
        if card.get("gap"):
            line += " · חסר: " + ", ".join(card["gap"])
    return line


def _cv_greeting(profile, cards):
    """Proactive Hebrew greeting (instant, no extra LLM call): name + the top detected
    roles, with a fit scorecard on the best one, then a clear next-step prompt."""
    name = (profile.get("name") or "").strip()
    hi = f"נעים להכיר, {name}! 👋" if name else "נעים להכיר! 👋"
    if not cards:
        return f"{hi} קראתי את קורות החיים שלך. אילו תפקידים תרצה שאחפש עבורך?"
    lines = [f"{hi} קראתי את קורות החיים שלך — מצאתי שאתה מתאים ל:", ""]
    lines += [_scorecard_line(c, detailed=(i == 0)) for i, c in enumerate(cards)]
    lines += ["", "שאחפש את אלה, או שתרצה להוסיף/לשנות תפקידים?"]
    return "\n".join(lines)


def _finish_cards(matches):
    """Shared post-retrieval field cleaning for job cards."""
    for m in matches:
        m["company"] = clean(m.get("company"))
        m["job_posting_url"] = clean(m.get("job_posting_url"))
        m["application_url"] = clean(m.get("application_url"))


def _short_intro(intent):
    """A SHORT one-line header for a result list (the lecturer requires brief answers)."""
    if isinstance(intent, dict):
        if intent.get("is_geo_fallback"):
            return "לא מצאתי בדיוק שם — הנה הקרובות ביותר 👇"
        if intent.get("nationwide"):
            return "הנה התאמות מרחבי ארה״ב 👇"
    return "אלה המשרות הכי רלוונטיות שמצאתי 👇"


def _mixed_search(profile, roles, retriever, top_n):
    """Return a MIX across the matched roles: every requested role gets at least one slot
    (coverage), then fill the rest, then order by score. Fixes (a) a 4th role vanishing when
    top_n<len(roles) and (b) ranks that contradicted the score. Returns (matches, intent)."""
    roles = roles[:5]
    target = min(max(top_n, len(roles)), 6)         # guarantee a slot per role (capped)
    per = max(2, target)

    buckets = []
    for role in roles:
        r_intent = ag.intent_from_cv(profile, [role], typed_query="")
        ms, _ = retriever(role, top_n=per, intent=r_intent)
        buckets.append(ms or [])

    seen, merged = set(), []

    def _add(m):
        key = (m.get("title"), m.get("company"), m.get("location"))
        if key in seen:
            return False
        seen.add(key)
        merged.append(m)
        return True

    # Pass 1 — coverage: one result per role (so every requested role appears).
    for ms in buckets:
        for m in ms:
            if _add(m):
                break
    # Pass 2 — fill the remaining slots round-robin.
    depth = 1
    while len(merged) < target and any(depth < len(b) for b in buckets):
        for b in buckets:
            if depth < len(b) and _add(b[depth]) and len(merged) >= target:
                break
        depth += 1

    # Order by score so the rank numbers never contradict the displayed score.
    merged.sort(key=lambda m: m.get("score", 0), reverse=True)
    merged = merged[:target]
    for idx, m in enumerate(merged, 1):
        m["rank"] = idx
    return merged, ag.intent_from_cv(profile, roles, typed_query="")


def _cv_turn(data, prompt, retriever, top_n):
    """One CV-conversation turn (Approach C), shared by both tabs.

    `retriever(query, top_n=, offset=, intent=) -> (matches, intent)` is the tab's
    search fn (ag.get_recommendations or scrape_search.search).

    Replies are kept SHORT (short intro + cards, no per-job blurbs/tip) per the
    lecturer's brevity requirement. Returns a jsonify-ready dict, or None when there
    is no CV context (the caller then falls back to the normal parse_query path).
    """
    cv_profile_data = data.get("profile")
    # Bound the CV text and conversation history at the route boundary (cost / context /
    # replay-DoS) — same posture as /api/cv-tailor. The controller also slices internally.
    cv_text_in = (data.get("cv_text") or "")[:20000]
    if not (cv_profile_data and cv_text_in):
        return None

    last_q = (data.get("lastQuery") or "").strip()
    offset = int(data.get("offset") or 0)
    history = [{"role": t.get("role", "user"), "content": str(t.get("content", ""))[:2000]}
               for t in (data.get("history") or [])[-8:]]
    cv_roles = data.get("cv_roles") or []

    # Paging ("עוד") reuses the stored CV roles — no extra LLM conversation call.
    if last_q and _wants_more(prompt):
        page_intent = ag.intent_from_cv(cv_profile_data, cv_roles, typed_query="")
        matches, intent = retriever(last_q, top_n=top_n, offset=offset, intent=page_intent)
        intent["more_options"] = True
        _finish_cards(matches)
        _attach_cv_fit(matches, cv_text_in, cv_profile_data)
        intro = "הנה עוד התאמות 👇" if matches else "אין עוד אפשרויות לחיפוש הזה."
        return {"matches": matches, "intro": intro, "tip": "", "reply": intro,
                "mode": "more", "query": last_q, "state": intent.get("state") or None,
                "roles": cv_roles, "searchCriteria": ag.criteria_from_intent(intent)}

    # Fresh conversational turn.
    conv = cv_converse.respond(history, cv_profile_data, cv_text_in,
                               ag.vectorizer, ag.claude_client, ag.LLM_MODEL)
    if conv["action"] == "tailor":   # hand off to the docked CV-coaching workspace
        return {"matches": [], "intro": conv["reply"], "tip": "", "reply": conv["reply"],
                "mode": "tailor", "tailor_query": conv.get("tailor_query", ""),
                "state": None, "roles": conv["roles"]}

    if conv["action"] == "data_question":
        q = conv.get("data_query") or prompt
        reply = analytics.answer_data_question(q, ag.df, ag.ANALYTICS_CTX,
                                               ag.claude_client, ag.LLM_MODEL)
        return {"matches": [], "intro": reply, "tip": "", "reply": reply,
                "mode": "data_question", "state": None, "roles": conv["roles"]}

    if conv["action"] == "advice":
        reply = ag.advice_reply(prompt, history=history, matches=(data.get("lastMatches") or []),
                                profile=cv_profile_data)
        return {"matches": [], "intro": reply, "tip": "", "reply": reply,
                "mode": "advice", "state": None, "roles": conv["roles"]}

    if conv["action"] != "search":
        return {"matches": [], "intro": conv["reply"], "tip": "", "reply": conv["reply"],
                "mode": "chat", "state": None, "roles": conv["roles"]}

    roles = conv["roles"]
    if len(roles) >= 2:                       # several roles -> return a MIX, not N of one title
        matches, intent = _mixed_search(cv_profile_data, roles, retriever, top_n)
    else:
        intent = ag.intent_from_cv(cv_profile_data, roles, typed_query="")
        matches, intent = retriever(prompt, top_n=top_n, intent=intent)
    _finish_cards(matches)
    _attach_cv_fit(matches, cv_text_in, cv_profile_data)
    return {"matches": matches, "intro": conv["reply"], "tip": "", "reply": conv["reply"],
            "mode": "search", "query": prompt, "state": intent.get("state") or None,
            "roles": roles, "searchCriteria": ag.criteria_from_intent(intent)}


def clean(v):
    """Convert pandas NaN / 'nan' to None so the JSON is valid for the browser."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, str) and v.strip().lower() == "nan":
        return None
    return v


@app.route("/")
def index():
    return send_from_directory(FRONTEND, "index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "ok": True,
        "llm": ag.claude_client is not None,
        "jobs": int(len(ag.df)),
    })


# =====================================================================
# CV ROUTES - upload, profile extraction, title suggestion, tailoring
# =====================================================================

@app.route("/api/cv", methods=["POST"])
def cv_upload():
    """
    Upload a CV (PDF / DOCX) or paste raw text.

    Multipart form:  field "file"  (.pdf or .docx, ≤ 5 MB)
    JSON fallback:   {"cv_text": "<raw pasted text>"}

    Success response (HTTP 200):
        {
          "ok": true,
          "cv_text": "<extracted plain text>",
          "profile": {
              "skills": [...],
              "titles_held": [...],
              "seniority": "mid-senior",
              "years": 7,
              "city": "Tel Aviv",
              "state": "CA",
              "domains": [...]
          },
          "titles": ["data analyst", "business intelligence analyst", ...]   // top-8
        }

    Error response (HTTP 200 so the browser can read the Hebrew body):
        {"ok": false, "error": "<Hebrew message>", "allow_paste": true}
    """
    # lazy imports so app startup doesn't fail if the cv package is mid-edit
    from cv import extract as cv_extract
    from cv import profile as cv_profile
    from cv import titles as cv_titles
    from cv import scorecard as cv_scorecard

    cv_text = ""
    filename = ""

    # --- file upload path ---
    if "file" in request.files:
        f = request.files["file"]
        filename = f.filename or ""
        ext = os.path.splitext(filename)[1].lower()
        if ext not in CV_ALLOWED_EXTS:
            return jsonify({"ok": False,
                            "error": "קובץ לא נתמך — יש להעלות קובץ PDF או DOCX בלבד.",
                            "allow_paste": True})
        file_bytes = f.read()
        if len(file_bytes) > CV_MAX_BYTES:
            return jsonify({"ok": False,
                            "error": "הקובץ גדול מדי (מעל 5 MB) — יש לצמצם או להדביק טקסט ישירות.",
                            "allow_paste": True})
        try:
            cv_text = cv_extract.extract_text(file_bytes, filename)
        except cv_extract.ExtractError as exc:
            return jsonify({"ok": False,
                            "error": f"לא ניתן לקרוא את הקובץ: {exc} — "
                                     "ניתן להדביק את טקסט קורות החיים ידנית.",
                            "allow_paste": True})
    else:
        # --- JSON paste fallback ---
        data = request.get_json(force=True, silent=True) or {}
        cv_text = (data.get("cv_text") or "").strip()
        if not cv_text:
            return jsonify({"ok": False,
                            "error": "לא התקבל קובץ ולא טקסט — יש להעלות PDF/DOCX או להדביק טקסט.",
                            "allow_paste": True})

    try:
        profile = cv_profile.build_profile(cv_text, ag.claude_client, ag.LLM_MODEL)
        # The document was readable but isn't a CV — stop before the expensive
        # titles/scorecard work and let the client offer the interview instead.
        if not profile.get("is_cv", True):
            return jsonify({"ok": False, "not_cv": True,
                            "error": "הקובץ לא נראה כמו קורות חיים."})
        titles = cv_titles.suggest_titles(cv_text, ag.vectorizer, ag.title_matrix, ag.TITLES, top_k=8)
        # Clean the top suggestions into canonical role names (e.g. drop "(Java+Python+AWS)").
        top_titles = cv_titles.clean_titles(titles[:3], ag.claude_client, ag.LLM_MODEL)
        # Proactive scorecard for those roles (detail on #1).
        cards = cv_scorecard.build(top_titles, profile, cv_text, ag.vectorizer,
                                   ag.claude_client, ag.LLM_MODEL)
    except Exception as exc:
        return jsonify({"ok": False,
                        "error": f"שגיאה בעיבוד קורות החיים: {exc}",
                        "allow_paste": True})

    return jsonify({"ok": True, "cv_text": cv_text, "profile": profile,
                    "titles": titles, "scorecard": cards,
                    "greeting": _cv_greeting(profile, cards)})


@app.route("/api/cv-tailor", methods=["POST"])
def cv_tailor():
    """
    One collaborative-editor turn: AI tailors the CV for a specific job.

    Request body (JSON):
        {
          "cv_text":  "<current CV text from the textarea>",
          "job":      {"title": "...", "company": "...", "description": "...", ...},
          "history":  [{"role": "user"|"assistant", "content": "..."}, ...],
          "user_msg": "<the user's latest chat message>"
        }

    Success response (HTTP 200):
        {
          "reply":       "<Hebrew chat reply>",
          "proposed_cv": "<full revised CV text, or null if no rewrite was suggested>"
        }
    """
    # lazy import
    from cv import tailor as cv_tailor_mod

    data = request.get_json(force=True, silent=True) or {}
    # bound the payload sent to Claude (cost / context-limit / replay DoS)
    cv_text = (data.get("cv_text") or "").strip()[:20000]
    job = data.get("job") or {}
    # The section-walk can run ~6 sections x a couple of turns, and COMPOSE must see EVERY
    # gathered fact (name, dates, etc.). Keep a generous window so early sections survive.
    history = (data.get("history") or [])[-40:]
    user_msg = (data.get("user_msg") or "").strip()[:4000]
    is_open = bool(data.get("open"))                 # proactive opening turn (no user msg yet)
    composed = bool(data.get("composed")) and not is_open   # a full rewrite already exists
    has_cv = bool(data.get("has_cv"))                # real CV uploaded -> enhance; else build
    try:
        section_index = max(0, int(data.get("section_index") or 0))
    except (TypeError, ValueError):
        section_index = 0
    n_sections = len(cv_tailor_mod.prompts.TAILOR_SECTIONS)
    is_compose_turn = section_index >= n_sections    # end-of-walk auto-compose (no user msg)

    if not cv_text:
        return jsonify({"error": "חסר טקסט קורות חיים"}), 400
    if not user_msg and not is_open and not is_compose_turn:
        return jsonify({"error": "חסרה הודעת משתמש"}), 400

    try:
        result = cv_tailor_mod.tailor_turn(
            cv_text, job, history, user_msg, ag.claude_client, ag.LLM_MODEL,
            composed=composed, has_cv=has_cv, section_index=section_index,
        )
    except Exception as exc:
        traceback.print_exc()                        # detail to server log, not the client
        return jsonify({"error": "שגיאה בעיבוד הבקשה — נסה שוב."}), 500

    return jsonify(result)   # {reply, proposed_cv}


@app.route("/api/profile-build", methods=["POST"])
def profile_build():
    """
    Drive the no-CV interview as a stateful, amendable CONVERSATION.

    Request (JSON): {"history": [...], "profile": <partial profile>, "user_msg": str,
                     "skipped": [field,...]}
      - A START call is {"profile": {}, "user_msg": ""} -> returns the opener question.
      - Otherwise the message is merged into the profile (add/change/remove any field),
        and we return the agent reply + the next missing field (or done + cv_text).

    Response (JSON):
      {"profile", "reply", "question" (alias of reply), "done": bool,
       "next_field": str|null, "skipped": [...], "cv_text": str|null}
    """
    from cv import interview

    data = request.get_json(force=True, silent=True) or {}
    profile = data.get("profile")
    profile = profile if isinstance(profile, dict) else {}
    # accept user_msg (new) or answer (legacy clients)
    user_msg = (data.get("user_msg") or data.get("answer") or "").strip()[:400]
    history = [{"role": t.get("role", "user"), "content": str(t.get("content", ""))[:2000]}
               for t in (data.get("history") or [])[-8:]]
    skipped = data.get("skipped")
    skipped = [str(k) for k in skipped] if isinstance(skipped, list) else []
    # the global dataset is US-only; the client sends us_only for that tab so the
    # location step can tell the user it only has US jobs.
    us_only = bool(data.get("us_only"))

    # START: empty profile + no message -> hand back the opener question.
    if not profile and not user_msg:
        return jsonify({"profile": {}, "reply": interview.STEPS[0]["question_he"],
                        "question": interview.STEPS[0]["question_he"],
                        "done": False, "next_field": "role", "skipped": [], "cv_text": None})

    try:
        out = interview.interview_turn(profile, history, user_msg,
                                       ag.claude_client, ag.LLM_MODEL,
                                       skipped=skipped, us_only=us_only)
        # ground the US state code to the dataset's valid set
        st = (out["profile"].get("state") or "").strip().upper()
        out["profile"]["state"] = st if st in ag.VALID_STATES else ""
        out.setdefault("question", out.get("reply", ""))   # back-compat alias
        return jsonify(out)
    except Exception:
        traceback.print_exc()                        # detail to server log, not the client
        return jsonify({"error": "שגיאה בעיבוד הבקשה — נסה שוב."}), 500


def _interview_kickoff(intent):
    """Build the mode:"interview" response that starts a pre-filled interview from a
    job-search opener. `intent` is the dict returned by ag.parse_query()."""
    from cv import interview

    profile, seeded, captured = interview.seed_from_query(
        intent.get("role_en", ""), intent.get("synonyms", []),
        intent.get("city", ""), intent.get("state", ""),
        intent.get("years"), intent.get("seniority", ""),
    )
    # short Hebrew lead-in (no extra LLM call)
    role = (profile.get("titles_held") or [None])[0]
    loc = profile.get("city") or profile.get("state")
    if role or loc:
        bits = "".join(filter(None, [f" — {role}" if role else "", f" ב{loc}" if loc else ""]))
        reply = f"הבנתי{bits}. עוד כמה פרטים ונצא לחפש."
    else:
        reply = "בוא נבנה פרופיל קצר ונצא לחפש."

    step = interview.first_unseeded_step(seeded)
    if step >= len(interview.STEPS):                       # everything provided at once
        profile = interview.finalize(profile)
        st = (profile.get("state") or "").strip().upper()
        profile["state"] = st if st in ag.VALID_STATES else ""
        return {"mode": "interview", "done": True, "profile": profile,
                "seeded": seeded, "captured": captured, "next_step": None,
                "question": "", "cv_text": interview.synthesize_cv_text(profile),
                "reply": reply}
    return {"mode": "interview", "done": False, "profile": profile,
            "seeded": seeded, "captured": captured, "next_step": step,
            "question": interview.STEPS[step]["question_he"], "cv_text": None,
            "reply": reply}


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    prompt = (data.get("prompt") or "").strip()

    if not prompt:
        return jsonify({"matches": [], "reply": "", "state": None})

    try:
        last_q = (data.get("lastQuery") or "").strip()
        offset = int(data.get("offset") or 0)
        # recent history (used by the refine fast-path and downstream handlers)
        history = (data.get("history") or [])[-8:]

        # --- Flow B: refine an existing search when the client echoes criteria ---
        crit = data.get("searchCriteria")
        if isinstance(crit, dict) and crit:
            ref = ag.refine_search_criteria(crit, history, prompt, ag.claude_client, ag.LLM_MODEL)
            if ref["changed"]:
                rintent = ag.intent_from_criteria(ref["criteria"])
                matches, rintent = ag.get_recommendations(prompt, top_n=3, intent=rintent)
                _finish_cards(matches)
                if data.get("cv_text"):
                    _attach_cv_fit(matches, (data.get("cv_text") or "")[:20000], data.get("profile"))
                intro = (ref["reply"] or (_short_intro(rintent) if matches
                                          else ag.explain_no_match(rintent)))
                return jsonify({"matches": matches, "intro": intro, "tip": "", "reply": intro,
                                "mode": "search", "query": prompt,
                                "searchCriteria": ref["criteria"],
                                "state": rintent.get("state") or None})
            if ref.get("blocked"):
                # non-US location request: explain, keep the criteria (incl. remote) unchanged,
                # do NOT fall through (which would reseed criteria and drop fields).
                reply = ref["reply"] or "כרגע יש לי משרות בארה\"ב בלבד — תן עיר/מדינה בארה\"ב, או שאחפש בכל ארה\"ב."
                return jsonify({"matches": [], "intro": reply, "tip": "", "reply": reply,
                                "mode": "chat", "searchCriteria": ref["criteria"], "state": None})

        # --- CV conversation path (Approach C) ---
        cv_resp = _cv_turn(data, prompt, ag.get_recommendations, top_n=3)
        if cv_resp is not None:
            return jsonify(cv_resp)

        # --- agentic tool-calling path (no CV loaded) ---
        # The MODEL drives this turn: given the 5 tools, it decides which to call (search /
        # stat / advice / paging / small-talk), with what args, and may chain several before
        # replying. Replaces the old parse_query()+mode-switch dispatch.
        result = agent_loop.run_agent_turn(
            prompt, history=history, last_q=last_q, offset=offset,
            screen_matches=(data.get("lastMatches") or [])[:5])
        # the agent asked to build a profile -> hand off to the guided interview
        # (cv/interview.py), which asks ONE question per turn.
        if result.get("mode") == "interview_kickoff":
            return jsonify(_interview_kickoff(result["seed"]))
        _finish_cards(result["matches"])
        if data.get("cv_text"):
            _attach_cv_fit(result["matches"], (data.get("cv_text") or "")[:20000], data.get("profile"))
        if result["matches"] and not data.get("cv_text"):
            result["tip"] = _CV_NUDGE
        return jsonify(result)
    except Exception:
        traceback.print_exc()                    # detail to the server log, not the client
        return jsonify({"error": "שגיאה בעיבוד הבקשה — נסה שוב."}), 500


# =====================================================================
# SCRAPE TAB - filter config (field editor), manual scrape, search-within
# =====================================================================
_scrape_state = {"running": False, "last": None}


def _scrape_config_payload():
    s = settings_store.load_settings()
    return {
        "filters": s["filters"],
        "options": {
            "experienceLevel": scfg.EXP_LEVELS,
            "contractType": scfg.CONTRACT_TYPES,
            "remote": scfg.REMOTE_OPTIONS,
            "datePosted": scfg.DATE_OPTIONS,
        },
        "jobs_stored": store.count(),
        "last_scraped": store.last_scraped(),
        "running": _scrape_state["running"],
        "last_result": _scrape_state["last"],
    }


@app.route("/api/scrape-config", methods=["GET", "POST"])
def scrape_config():
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
        if "filters" in data:
            settings_store.save_filters(data.get("filters") or {})
        return jsonify({"ok": True, **_scrape_config_payload()})
    return jsonify(_scrape_config_payload())


@app.route("/api/scrape-now", methods=["POST"])
def scrape_now():
    data = request.get_json(force=True, silent=True) or {}
    limit = data.get("limit")
    # Live scraping can't run on a read-only serverless host (no writable FS / background
    # workers / long-running jobs). Search over the pre-scraped jobs still works.
    if os.environ.get("VERCEL"):
        return jsonify({"ok": False, "error": "סריקה חיה אינה זמינה בגרסה המתארחת (Vercel) — "
                        "אפשר לחפש במשרות שכבר נסרקו. להרצת סריקה חדשה יש להריץ מקומית או בשרת קבוע."})
    if _scrape_state["running"]:
        return jsonify({"ok": False, "error": "סריקה כבר רצה כעת."}), 409

    def worker():
        _scrape_state["running"] = True
        print("\n[scrape] ===== Scrape Now triggered =====", flush=True)
        try:
            res = run_pipeline.run(limit)
            _scrape_state["last"] = res
            print("[scrape] result:", res, flush=True)
        except Exception as e:
            traceback.print_exc()
            _scrape_state["last"] = {"ok": False, "error": str(e)}
            print("[scrape] FAILED:", e, flush=True)
        finally:
            _scrape_state["running"] = False

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True, "started": True})


@app.route("/api/scrape-search", methods=["POST"])
def scrape_search_route():
    data = request.get_json(force=True, silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"matches": [], "reply": ""})
    try:
        last_q = (data.get("lastQuery") or "").strip()
        offset = int(data.get("offset") or 0)

        # --- CV conversation path (Approach C) ---
        cv_resp = _cv_turn(data, prompt, scrape_search.search, top_n=5)
        if cv_resp is not None:
            return jsonify(cv_resp)

        # --- normal free-text path (no CV loaded) ---
        intent = ag.parse_query(prompt)
        mode = intent.get("mode", "search")

        # No CV loaded: a non-paging job_search opener -> start a seeded interview instead
        # of scraping/searching blind (job_search => collect a short profile, then search).
        if not (mode == "more" and last_q) and intent.get("intent") == "job_search":
            return jsonify(_interview_kickoff(intent))

        # bound client-supplied context at the route boundary (cost / replay-DoS)
        screen = (data.get("lastMatches") or [])[:5]
        history = (data.get("history") or [])[-8:]

        if mode == "smalltalk":
            reply = ag.chat_reply(prompt, tab="scrape", history=history)
            return jsonify({"matches": [], "intro": reply, "tip": "", "reply": reply,
                            "mode": "chat", "state": None})
        # NOTE: data_question is intentionally NOT offered on the scrape tab — the live
        # Israel set lacks reliable salary/industry coverage. Treat it as advice instead.
        if mode in ("advice", "data_question"):
            reply = ag.advice_reply(prompt, history=history, matches=screen, profile=None)
            return jsonify({"matches": [], "intro": reply, "tip": "", "reply": reply,
                            "mode": "advice", "state": None})

        more = mode == "more" and bool(last_q)

        if more:
            matches, intent = scrape_search.search(last_q, top_n=5, offset=offset)
            intent["more_options"] = True
            query = last_q
        else:
            matches, intent = scrape_search.search(prompt, top_n=5, intent=intent)
            query = prompt

        # SHORT answers (lecturer requirement): a one-line header + cards, no per-job blurbs.
        _finish_cards(matches)
        if matches:
            intro = "הנה עוד התאמות 👇" if more else _short_intro(intent)
        elif more:
            intro = "אין עוד אפשרויות לחיפוש הזה."
        else:
            intro = scrape_search.no_match_message(intent)
        # scraped jobs are Israel-only: if the user asked elsewhere, say so briefly but still answer
        if matches and isinstance(intent, dict) and intent.get("loc_outside_israel"):
            intro = f"📍 אין משרות ב\"{intent['loc_outside_israel']}\" — הנה הטובות בישראל 👇"
        tip = _CV_NUDGE if (matches and not data.get("cv_text")) else ""
        return jsonify({"matches": matches, "intro": intro, "tip": tip, "reply": intro,
                        "mode": "more" if more else "search", "query": query})
    except Exception:
        traceback.print_exc()                    # detail to the server log, not the client
        return jsonify({"error": "שגיאה בעיבוד הבקשה — נסה שוב."}), 500


if __name__ == "__main__":
    print("=" * 50)
    print(" Smart Career Navigator  ->  http://127.0.0.1:5000")
    print("=" * 50)
    # debug=False: no auto-reloader (it would re-import and re-load the 894MB twice)
    app.run(host="127.0.0.1", port=5000, debug=False)
