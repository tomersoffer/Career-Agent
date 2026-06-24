# -*- coding: utf-8 -*-
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

import matching
from cv import converse


@pytest.fixture
def vec():
    v = TfidfVectorizer()
    v.fit(["data analyst sql python", "nurse hospital"])
    return v


def test_degrades_without_client(vec):
    out = converse.respond([], {"skills": []}, "data analyst sql", vec, None, "m")
    assert out["action"] == "talk"
    assert out["reply"]                      # non-empty fallback
    assert out["roles"] == []


def test_search_action(monkeypatch, vec):
    calls = iter([
        {"roles": ["data analyst"], "location": ""},                       # extract call
        {"reply": "מחפש", "action": "search",
         "roles": ["data analyst"], "location": ""},                       # reply call
    ])
    monkeypatch.setattr(matching, "reply_json", lambda *a, **k: next(calls))
    out = converse.respond(
        [{"role": "user", "content": "כן"}],
        {"skills": ["sql"], "seniority": "entry", "years": 0},
        "data analyst sql", vec, object(), "m")
    assert out["action"] == "search"
    assert out["roles"] == ["data analyst"]


def test_invalid_action_coerced_to_talk(monkeypatch, vec):
    calls = iter([
        {"roles": [], "location": ""},
        {"reply": "...", "action": "frobnicate", "roles": [], "location": ""},
    ])
    monkeypatch.setattr(matching, "reply_json", lambda *a, **k: next(calls))
    out = converse.respond([], {"skills": []}, "x", vec, object(), "m")
    assert out["action"] == "talk"
