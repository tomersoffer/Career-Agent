# -*- coding: utf-8 -*-
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

import agent_runner as ag
from cv import state_delta as sd


def test_intent_from_criteria_maps_levers():
    it = ag.intent_from_criteria({"roles": ["data analyst", "bi analyst"], "state": "NY",
                                  "city": "New York", "seniority": "mid-senior",
                                  "years": 4, "remote": True})
    assert it["role_en"] == "data analyst"
    assert it["synonyms"] == ["bi analyst"]
    assert "data analyst" in it["query_text"]
    assert it["state"] == "NY"
    assert it["remote"] is True
    assert it["experience_rank"] > 0          # derived from seniority/years


def test_intent_from_criteria_grounds_bad_state():
    it = ag.intent_from_criteria({"roles": ["nurse"], "state": "ZZ"})
    assert it["state"] == ""                  # ZZ not in VALID_STATES
    assert it["nationwide"] is True


def test_criteria_from_intent_round_trip():
    it = ag.intent_from_criteria({"roles": ["data analyst", "bi analyst"], "remote": True})
    crit = ag.criteria_from_intent(it)
    assert crit["roles"] == ["data analyst", "bi analyst"]
    assert crit["remote"] is True


def test_refine_applies_delta_and_flags_change(monkeypatch):
    monkeypatch.setattr(sd, "extract_delta", lambda *a, **k: {
        "set": {"remote": True}, "add": {}, "remove": {}, "reply": "מחפש מרחוק", "ready": False})
    out = ag.refine_search_criteria({"roles": ["data analyst"], "remote": False},
                                    [], "תעשה את זה מרחוק", object(), "m")
    assert out["changed"] is True
    assert out["criteria"]["remote"] is True
    assert out["reply"] == "מחפש מרחוק"


def test_refine_empty_delta_is_not_a_change(monkeypatch):
    monkeypatch.setattr(sd, "extract_delta", lambda *a, **k: {
        "set": {}, "add": {}, "remove": {}, "reply": "", "ready": False})
    out = ag.refine_search_criteria({"roles": ["data analyst"]}, [], "מה השכר הממוצע?", object(), "m")
    assert out["changed"] is False
    assert out["criteria"] == {"roles": ["data analyst"]}


def test_refine_non_us_location_blocks_and_preserves_remote(monkeypatch):
    # user asks for Tel Aviv: LLM refuses (blocked), keeps location, preserves other fields
    monkeypatch.setattr(sd, "extract_delta", lambda *a, **k: {
        "set": {}, "add": {}, "remove": {}, "reply": "יש רק משרות בארה\"ב",
        "ready": False, "blocked": True})
    prior = {"roles": ["data analyst"], "state": "NY", "remote": True}
    out = ag.refine_search_criteria(prior, [], "תעביר לתל אביב", object(), "m")
    assert out["blocked"] is True
    assert out["changed"] is False
    assert out["criteria"]["state"] == "NY"        # location unchanged
    assert out["criteria"]["remote"] is True       # remote NOT dropped


def test_remote_filter_returns_only_remote():
    # integration: loads the dataset (agent_runner import).
    it = ag.intent_from_criteria({"roles": ["data analyst"], "remote": True})
    matches, _ = ag.get_recommendations("data analyst", top_n=5, intent=it)
    assert matches, "expected some remote data-analyst matches"
    assert all(m.get("remote_allowed") == 1 for m in matches)
