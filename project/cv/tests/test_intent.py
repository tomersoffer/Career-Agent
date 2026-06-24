# -*- coding: utf-8 -*-
import agent_runner as ag


def _patch(monkeypatch, payload):
    monkeypatch.setattr(ag, "_llm_parse", lambda prompt: payload)
    # ensure the LLM branch runs
    monkeypatch.setattr(ag, "claude_client", object())


def test_intent_job_search_passthrough(monkeypatch):
    _patch(monkeypatch, {"role_en": "data analyst", "is_search": True, "intent": "job_search"})
    out = ag.parse_query("data analyst in NY")
    assert out["intent"] == "job_search"


def test_intent_info_passthrough(monkeypatch):
    _patch(monkeypatch, {"role_en": "data analyst", "is_search": False, "intent": "info"})
    out = ag.parse_query("how much does a data analyst make?")
    assert out["intent"] == "info"


def test_intent_defaults_from_is_search_when_absent(monkeypatch):
    # LLM omitted intent -> derive: is_search True => job_search
    _patch(monkeypatch, {"role_en": "nurse", "is_search": True})
    assert ag.parse_query("find me nurse jobs")["intent"] == "job_search"
    # is_search False => info
    _patch(monkeypatch, {"role_en": "", "is_search": False})
    assert ag.parse_query("thanks")["intent"] == "info"


def test_intent_junk_value_coerced_to_default(monkeypatch):
    # an out-of-vocab intent falls back to the is_search-derived default (locks the whitelist)
    _patch(monkeypatch, {"role_en": "data analyst", "is_search": True, "intent": "wat"})
    assert ag.parse_query("data analyst in NY")["intent"] == "job_search"
