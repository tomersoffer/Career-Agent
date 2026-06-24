# project/tests/test_route_integration.py
# Cara's integration tests — Tasks 7, 8, 9.
# cv_converse action tests live HERE (not in test_router.py which is Beni's file).

import json as _json
import pytest

# =====================================================================
# Task 4 smoke (also owned here as a shared resource check)
# =====================================================================

def test_agent_runner_exposes_analytics_ctx():
    import agent_runner as ag
    import analytics
    # The widened columns the analytics engine needs must be present.
    for col in ("skills", "salary", "remote_allowed", "job_industry"):
        assert col in ag.df.columns
    assert isinstance(ag.ANALYTICS_CTX, analytics.Ctx)


# =====================================================================
# Task 7 — cv_converse emits data_question + advice actions
# =====================================================================

from cv import converse as cv_converse


class _Reply:
    def __init__(self, payload):
        self._payload = payload
        self.messages = self

    def create(self, **kwargs):
        import json
        return type("M", (), {"content": [type("B", (), {
            "type": "text", "text": json.dumps(self._payload)})()]})()


def test_cv_converse_passes_through_data_question(monkeypatch):
    # First call = _extract_roles, second = the reply. Patch reply_json to return per-call.
    calls = iter([
        {"roles": ["data analyst"], "location": "NY"},                      # extract
        {"reply": "יש בערך 300 משרות כאלה.", "action": "data_question",
         "roles": ["data analyst"], "data_query": "how many data analyst jobs in NY",
         "tailor_query": "", "location": "NY"},                              # reply
    ])
    monkeypatch.setattr(cv_converse.matching, "reply_json",
                        lambda *a, **k: next(calls))
    monkeypatch.setattr(cv_converse.role_fit, "assess", lambda *a, **k: {})
    out = cv_converse.respond([], {"titles_held": ["data analyst"]}, "cv",
                              object(), object(), "model")
    assert out["action"] == "data_question"
    assert out["data_query"] == "how many data analyst jobs in NY"


def test_cv_converse_advice_action(monkeypatch):
    calls = iter([
        {"roles": [], "location": ""},
        {"reply": "כדאי להתמקד ב-SQL.", "action": "advice", "roles": [],
         "data_query": "", "tailor_query": "", "location": ""},
    ])
    monkeypatch.setattr(cv_converse.matching, "reply_json", lambda *a, **k: next(calls))
    monkeypatch.setattr(cv_converse.role_fit, "assess", lambda *a, **k: {})
    out = cv_converse.respond([], {}, "cv", object(), object(), "model")
    assert out["action"] == "advice"


# =====================================================================
# Task 8 — /api/chat (no-CV path) delegates to the agentic tool-calling loop
# and forwards its structured result (mode / matches / reply) to the client.
# The loop itself (model issues real tool calls) is tested separately; here we
# mock run_agent_turn at the route boundary and assert the wiring/contract.
# =====================================================================

@pytest.fixture
def client():
    import app
    app.app.config["TESTING"] = True
    return app.app.test_client()


def _post(client, body):
    return client.post("/api/chat", data=_json.dumps(body),
                       content_type="application/json")


def test_chat_data_question_returns_narrated_no_cards(client, monkeypatch):
    import app
    monkeypatch.setattr(app.agent_loop, "run_agent_turn",
                        lambda *a, **k: {"matches": [], "intro": "יש בערך 300 משרות.",
                                         "tip": "", "reply": "יש בערך 300 משרות.",
                                         "mode": "data_question", "state": "NY",
                                         "query": None, "tool_trace": [{"tool": "get_job_stat"}]})
    r = _post(client, {"prompt": "how many data analyst jobs in NY?"})
    data = r.get_json()
    assert data["mode"] == "data_question"
    assert data["matches"] == []
    assert "300" in data["reply"]


def test_chat_advice_returns_text_no_cards(client, monkeypatch):
    import app
    monkeypatch.setattr(app.agent_loop, "run_agent_turn",
                        lambda *a, **k: {"matches": [], "intro": "התמקד ב-SQL.",
                                         "tip": "", "reply": "התמקד ב-SQL.",
                                         "mode": "advice", "state": None,
                                         "query": None, "tool_trace": [{"tool": "give_career_advice"}]})
    r = _post(client, {"prompt": "what should I learn first?"})
    data = r.get_json()
    assert data["mode"] == "advice"
    assert data["matches"] == []
    assert data["reply"] == "התמקד ב-SQL."


# =====================================================================
# Task 9 — _cv_turn dispatches data_question + advice for CV users
# =====================================================================

def test_cv_turn_data_question(client, monkeypatch):
    import app
    monkeypatch.setattr(app.cv_converse, "respond",
                        lambda *a, **k: {"reply": "x", "action": "data_question",
                                         "roles": ["data analyst"],
                                         "data_query": "avg salary data analyst NY",
                                         "tailor_query": "", "location": "NY"})
    monkeypatch.setattr(app.analytics, "answer_data_question",
                        lambda *a, **k: "החציון כ-95 אלף דולר.")
    body = {"prompt": "מה השכר?", "cv_text": "Skills: sql",
            "profile": {"titles_held": ["data analyst"], "state": "NY"}}
    r = _post(client, body)
    data = r.get_json()
    assert data["mode"] == "data_question"
    assert data["matches"] == []
    assert "95" in data["reply"]


def test_cv_turn_advice(client, monkeypatch):
    import app
    monkeypatch.setattr(app.cv_converse, "respond",
                        lambda *a, **k: {"reply": "ignore", "action": "advice", "roles": [],
                                         "data_query": "", "tailor_query": "", "location": ""})
    monkeypatch.setattr(app.ag, "advice_reply", lambda *a, **k: "שפר את ה-summary.")
    body = {"prompt": "איך לשפר קו\"ח?", "cv_text": "Skills: sql",
            "profile": {"titles_held": ["data analyst"]}}
    r = _post(client, body)
    data = r.get_json()
    assert data["mode"] == "advice"
    assert data["reply"] == "שפר את ה-summary."


# =====================================================================
# /api/cv — non-CV upload is detected and short-circuits to the modal path
# =====================================================================

def test_cv_upload_non_cv_returns_not_cv(client, monkeypatch):
    # The document parsed fine but isn't a CV -> {ok:false, not_cv:true}, no
    # titles/scorecard work, no allow_paste.
    from cv import profile as cv_profile
    monkeypatch.setattr(cv_profile, "build_profile",
                        lambda *a, **k: {"is_cv": False, "cv_confidence": 0.9})
    # If the route did NOT short-circuit it would call these; make them blow up.
    from cv import titles as cv_titles
    monkeypatch.setattr(cv_titles, "suggest_titles",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")))
    r = client.post("/api/cv", data=_json.dumps({"cv_text": "INVOICE #4471 — total $1,250"}),
                    content_type="application/json")
    data = r.get_json()
    assert data["ok"] is False
    assert data["not_cv"] is True
    assert "allow_paste" not in data


def test_cv_upload_real_cv_proceeds(client, monkeypatch):
    # is_cv true -> the route continues into the normal success path.
    from cv import profile as cv_profile
    from cv import titles as cv_titles
    from cv import scorecard as cv_scorecard
    monkeypatch.setattr(cv_profile, "build_profile",
                        lambda *a, **k: {"is_cv": True, "skills": ["python"], "titles_held": []})
    monkeypatch.setattr(cv_titles, "suggest_titles", lambda *a, **k: ["data analyst"])
    monkeypatch.setattr(cv_titles, "clean_titles", lambda *a, **k: ["data analyst"])
    monkeypatch.setattr(cv_scorecard, "build", lambda *a, **k: [])
    r = client.post("/api/cv", data=_json.dumps({"cv_text": "Experienced Python developer, SQL, pandas."}),
                    content_type="application/json")
    data = r.get_json()
    assert data["ok"] is True
    assert "not_cv" not in data
