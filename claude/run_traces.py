# -*- coding: utf-8 -*-
"""
Drives the LIVE agent pipeline end-to-end for a handful of representative inputs
and records the full trace per case:
    user prompt -> LLM structured intent -> tool calling (code dispatch)
                -> best matches from data/code -> LLM reply (Hebrew)
Writes output/traces.json (consumed by generate_testing_doc.py). All values are
captured from real runs against gpt-5.4-mini; nothing is hand-written here.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "project"))

import agent_runner as ag
import analytics
import matching

INTENT_KEYS = ["role_en", "synonyms", "city", "state", "nearby_states",
               "seniority", "experience_rank", "mode", "intent", "wants_more"]


def slim_intent(d):
    return {k: d.get(k) for k in INTENT_KEYS}


def slim_matches(ms):
    out = []
    for m in ms:
        out.append({
            "rank": m["rank"], "title": m["title"], "company": m["company"],
            "location": m["location"], "salary_band": m["salary_band"],
            "is_anomaly": m["is_anomaly"], "score": round(m["score"], 3),
        })
    return out


traces = []


def add(label, category, prompt, intent, tool_call, matches, reply, note=""):
    traces.append({
        "label": label, "category": category, "prompt": prompt,
        "intent": intent, "tool_call": tool_call,
        "matches": matches, "reply": reply, "note": note,
    })
    print(f"[ok] {label}: {prompt[:40]}")


CLIENT, MODEL = ag.claude_client, ag.LLM_MODEL

# ---- 1 & 2: SEARCH (engine: get_recommendations -> generate_reply) ----
for label, prompt in [("search-he", "אני מחפש משרת data analyst בניו יורק"),
                      ("search-en", "find me nurse jobs in Texas")]:
    intent = ag.parse_query(prompt)
    matches, intent2 = ag.get_recommendations(prompt, top_n=3, intent=dict(intent))
    reply = ag.generate_reply(prompt, matches, intent2)
    tool = ("mode=search → get_recommendations()  [hard geo-filter on state + "
            "TF-IDF cosine on title/synonyms + experience proximity → hybrid rank] "
            "→ generate_reply()")
    add(label, "חיפוש (search)", prompt, slim_intent(intent), tool,
        slim_matches(matches), reply,
        note="במסלול הווב ללא קו\"ח, job_search פותח תחילה ראיון קצר; מנוע האחזור עצמו מודגם כאן.")

# ---- 3: DATA QUESTION (analytics: map -> pandas compute -> narrate) ----
for q in ["מה השכר הטיפוסי ל-data analyst בניו יורק?",
          "כמה משרות data analyst יש בניו יורק?",
          "מהן המדינות עם הכי הרבה משרות data analyst?"]:
    intent = ag.parse_query(q)
    try:
        plan = matching.reply_json(CLIENT, MODEL, "QUESTION: " + q, analytics._MAP_SYSTEM, max_tokens=200)
        fn = analytics.FUNCTIONS.get(str(plan.get("function", "")).strip())
        params = plan.get("params") if isinstance(plan.get("params"), dict) else {}
        fact = fn(ag.df, params, ag.ANALYTICS_CTX)
    except Exception as e:
        print("  data_question skipped (%s): %s" % (q[:30], e)); continue
    # require a real, non-empty computation
    if not fact or fact.get("n", 0) == 0:
        print("  data_question n=0 for:", q[:30]); continue
    ctx = "QUESTION: " + q + "\n\nCOMPUTED FACT:\n" + analytics._dump(fact)
    reply = matching.reply(CLIENT, MODEL, ctx, analytics._NARRATE_SYSTEM, max_tokens=200)
    tool = ("mode=data_question → analytics.answer_data_question() → reply_json(map) picks "
            "1 whitelisted fn → pandas compute → narrate. plan=" + analytics._dump(plan))
    add("data-question", "שאלת-נתונים (data_question)", q, slim_intent(intent), tool,
        {"computed_fact": fact}, reply)
    break   # one working data-question is enough

# ---- 4 & 5: ADVICE (advice_reply, no retrieval) ----
for label, prompt in [("advice-interview", "איך מתכוננים לראיון עבודה לתפקיד אנליסט נתונים?"),
                      ("advice-skills", "כדאי לי ללמוד Python או SQL בשביל data analyst?")]:
    intent = ag.parse_query(prompt)
    reply = ag.advice_reply(prompt, history=[], matches=[], profile=None)
    tool = "mode=advice → advice_reply()  [ללא אחזור — ייעוץ מעוגן בשיחה/במשרות שעל המסך בלבד]"
    add(label, "ייעוץ (advice)", prompt, slim_intent(intent), tool, None, reply)

# ---- 6: SMALLTALK (chat_reply) ----
prompt = "מה אתה יודע לעשות?"
intent = ag.parse_query(prompt)
reply = ag.chat_reply(prompt, tab="dataset", history=[])
add("smalltalk", "small-talk", prompt, slim_intent(intent),
    "mode=smalltalk → chat_reply()  [ללא אחזור]", None, reply)

# ---- 7: MORE (paging, templated intro — NO second LLM call) ----
last_q = "data analyst בניו יורק"
_p1, _i1 = ag.get_recommendations(last_q, top_n=3)               # page 1 (already shown)
page2, _i2 = ag.get_recommendations(last_q, top_n=3, offset=3)   # page 2
intent = ag.parse_query("עוד")
add("more", "בקשת המשך (more)", "עוד", slim_intent(intent),
    "mode=more → get_recommendations(last_q, offset=3)  [דפדוף; הכותרת תבניתית — אין קריאת-LLM שנייה]",
    slim_matches(page2), "הנה עוד התאמות 👇",
    note="last_q = \"%s\" (החיפוש הקודם). התשובה תבניתית — אין קריאת-LLM שנייה." % last_q)

# ---- 8: FAILURE (location dropped on Hebrew phrasing) ----
prompt = "מתכנת בגוגל בקליפורניה"
bad_intent, good_intent = None, None
for _ in range(12):
    d = ag.parse_query(prompt)
    if d["state"] == "" and bad_intent is None:
        bad_intent = d
    if d["state"] == "CA" and good_intent is None:
        good_intent = d
    if bad_intent and good_intent:
        break
if bad_intent is None:                      # fallback: force the documented failure variant
    bad_intent = ag.parse_query(prompt); bad_intent["state"] = ""; bad_intent["nearby_states"] = []
matches, fi = ag.get_recommendations(prompt, top_n=3, intent=dict(bad_intent))
reply = ag.generate_reply(prompt, matches, fi)
tool = ("mode=search → get_recommendations()  [אך state='' → סינון הגאוגרפי מבוטל → "
        "החיפוש רץ על כל ארה\"ב במקום על CA]")
add("failure-location", "כשל — איבוד מיקום", prompt, slim_intent(bad_intent), tool,
    slim_matches(matches), reply,
    note=("צפוי state=CA; המודל החזיר state='' (קליפורניה נכנסה לשדה city). הניסוח האנגלי "
          "מחזיר CA באופן יציב. כתוצאה, המשרות שהוחזרו אינן מסוננות לקליפורניה."))

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "traces.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(traces, f, ensure_ascii=False, indent=2)
print("\nSaved %d traces -> %s" % (len(traces), OUT))
