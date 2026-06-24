# -*- coding: utf-8 -*-
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

import app as appmod
from cv import tailor


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def test_composed_flag_is_forwarded(monkeypatch):
    seen = {}
    def fake_turn(cv_text, job, history, user_msg, client, model, composed=False, **kw):
        seen["composed"] = composed
        return {"reply": "ok", "proposed_cv": None, "done": False}
    monkeypatch.setattr(tailor, "tailor_turn", fake_turn)
    c = _client()
    r = c.post("/api/cv-tailor", json={"cv_text": "CV", "job": {"title": "Analyst"},
                                       "history": [{"role": "assistant", "content": "rewrite"}],
                                       "user_msg": "תוריד Excel", "composed": True})
    assert r.status_code == 200
    assert seen["composed"] is True


def test_section_fields_forwarded(monkeypatch):
    seen = {}
    def fake_turn(cv_text, job, history, user_msg, client, model,
                  composed=False, has_cv=False, section_index=0):
        seen.update(has_cv=has_cv, section_index=section_index)
        return {"reply": "?", "proposed_cv": None, "done": False,
                "section_done": False, "next_section_index": section_index, "total_sections": 6}
    monkeypatch.setattr(tailor, "tailor_turn", fake_turn)
    c = _client()
    r = c.post("/api/cv-tailor", json={"cv_text": "CV", "job": {"title": "Analyst"},
                                       "user_msg": "תשובה", "has_cv": True, "section_index": 2})
    assert r.status_code == 200
    assert seen["has_cv"] is True and seen["section_index"] == 2


def test_compose_turn_allowed_without_user_msg(monkeypatch):
    from cv import prompts
    n = len(prompts.TAILOR_SECTIONS)
    monkeypatch.setattr(tailor, "tailor_turn",
        lambda *a, **k: {"reply": "הנה הקו\"ח", "proposed_cv": "FULL CV", "done": False,
                         "section_done": False, "next_section_index": n, "total_sections": n})
    c = _client()
    # end-of-walk compose turn: no user_msg, section_index at the end -> must NOT 400
    r = c.post("/api/cv-tailor", json={"cv_text": "CV", "job": {"title": "Analyst"},
                                       "user_msg": "", "section_index": n})
    assert r.status_code == 200
    assert r.get_json()["proposed_cv"] == "FULL CV"


def test_open_forces_composed_false(monkeypatch):
    seen = {}
    def fake_turn(cv_text, job, history, user_msg, client, model, composed=False, **kw):
        seen["composed"] = composed
        return {"reply": "ok", "proposed_cv": None, "done": False}
    monkeypatch.setattr(tailor, "tailor_turn", fake_turn)
    c = _client()
    r = c.post("/api/cv-tailor", json={"cv_text": "CV", "job": {"title": "Analyst"},
                                       "open": True, "composed": True})
    assert r.status_code == 200
    assert seen["composed"] is False     # opener can never be an amendment
