# -*- coding: utf-8 -*-
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

import app as appmod
import agent_runner as ag


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def test_refinement_amends_criteria_and_searches(monkeypatch):
    # criteria present + a refinement -> fast-path re-search with merged criteria
    monkeypatch.setattr(ag, "refine_search_criteria",
        lambda *a, **k: {"criteria": {"roles": ["data analyst"], "remote": True},
                         "reply": "מחפש מרחוק", "ready": False, "changed": True})
    monkeypatch.setattr(ag, "get_recommendations",
        lambda *a, **k: ([{"rank": 1, "title": "Data Analyst", "company": "X",
                            "location": "NY", "salary_band": 2, "experience_rank": 2,
                            "experience_level": "mid", "remote_allowed": 1, "is_anomaly": 0,
                            "cluster_id": 0, "score": 0.9, "title_score": 0.9, "exp_score": 0.5,
                            "job_posting_url": "u", "application_url": "a"}],
                          ag.intent_from_criteria({"roles": ["data analyst"], "remote": True})))
    c = _client()
    r = c.post("/api/chat", json={"prompt": "תעשה את זה מרחוק",
                                  "searchCriteria": {"roles": ["data analyst"], "remote": False}})
    body = r.get_json()
    assert body["mode"] == "search"
    assert body["searchCriteria"]["remote"] is True
    assert body["matches"] and body["matches"][0]["title"] == "Data Analyst"


def test_non_us_location_blocked_preserves_criteria(monkeypatch):
    # blocked refinement (non-US location) -> explain, keep criteria, no fall-through/search
    monkeypatch.setattr(ag, "refine_search_criteria",
        lambda *a, **k: {"criteria": {"roles": ["data analyst"], "state": "NY", "remote": True},
                         "reply": "יש לי רק משרות בארה\"ב", "ready": False,
                         "changed": False, "blocked": True})
    c = _client()
    r = c.post("/api/chat", json={"prompt": "תעביר לתל אביב",
                                  "searchCriteria": {"roles": ["data analyst"], "state": "NY", "remote": True}})
    body = r.get_json()
    assert "ארה" in (body.get("reply") or "")          # explained US-only
    assert body["searchCriteria"]["remote"] is True    # remote preserved
    assert body["searchCriteria"]["state"] == "NY"     # location unchanged
    assert not body.get("matches")                     # no search ran


def test_non_refinement_falls_through(monkeypatch):
    # empty, non-blocked delta -> not a refinement -> falls through to normal handling
    # (mock the downstream LLM calls so the test is hermetic / offline).
    monkeypatch.setattr(ag, "refine_search_criteria",
        lambda *a, **k: {"criteria": {"roles": ["data analyst"]}, "reply": "",
                         "ready": False, "changed": False, "blocked": False})
    monkeypatch.setattr(ag, "parse_query", lambda p: {"mode": "smalltalk", "intent": "info",
                                                      "state": "", "is_search": False})
    monkeypatch.setattr(ag, "chat_reply", lambda *a, **k: "היי, איך אפשר לעזור?")
    c = _client()
    r = c.post("/api/chat", json={"prompt": "היי מה קורה",
                                  "searchCriteria": {"roles": ["data analyst"]}})
    assert r.status_code == 200
    body = r.get_json()
    assert body.get("mode") != "search"        # no spurious re-search
