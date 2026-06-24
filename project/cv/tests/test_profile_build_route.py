# -*- coding: utf-8 -*-
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

import app as appmod
from cv import interview


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def test_start_returns_opener_question():
    c = _client()
    r = c.post("/api/profile-build", json={"profile": {}, "user_msg": "", "history": []})
    assert r.status_code == 200
    body = r.get_json()
    assert body["question"] or body["reply"]      # an opener is returned
    assert body["done"] is False


def test_turn_merges_and_grounds_state(monkeypatch):
    # stub interview_turn so the route is tested independently of the LLM
    monkeypatch.setattr(interview, "interview_turn",
        lambda *a, **k: {"profile": {"titles_held": ["data analyst"], "state": "ZZ"},
                         "reply": "מעולה", "done": False, "next_field": "experience",
                         "skipped": [], "cv_text": None})
    c = _client()
    r = c.post("/api/profile-build",
               json={"profile": {}, "user_msg": "data analyst", "history": []})
    body = r.get_json()
    assert body["reply"] == "מעולה"
    assert body["profile"]["state"] == ""          # invalid state grounded out
