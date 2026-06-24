# -*- coding: utf-8 -*-
"""
cv/tests/test_tailor.py
TDD tests for cv/tailor.py — deterministic stage inference + LLM-off degrade.

Run with:
    cd project && python -m pytest cv/tests/test_tailor.py -v
"""

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, "..", ".."))   # …/project
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from cv import tailor


def _asst(n):
    """A history with n assistant turns (interleaved with user turns)."""
    h = []
    for i in range(n):
        h.append({"role": "assistant", "content": f"q{i}"})
        h.append({"role": "user", "content": f"a{i}"})
    return h


class TestIsDone:
    def test_closing_words_detected(self):
        assert tailor._is_done("סיימתי") is True
        assert tailor._is_done("מספיק תודה") is True
        assert tailor._is_done("בוא נכתוב") is True

    def test_normal_answer_is_not_done(self):
        assert tailor._is_done("כן, ניהלתי פרויקט בלימודים") is False
        assert tailor._is_done("") is False


_SECTIONS = [{"key": "personal_details", "label_he": "פרטים אישיים"},
             {"key": "education", "label_he": "השכלה"},
             {"key": "summary", "label_he": "תקציר מקצועי"}]


class TestInferStage:
    def test_opener_is_section_walk_first_section(self):
        stage, key = tailor._infer_tailor_stage([], "", True, _SECTIONS, 0, composed=False)
        assert stage == "SECTION_WALK" and key == "personal_details"

    def test_mid_index_walks_that_section(self):
        stage, key = tailor._infer_tailor_stage(_asst(2), "תשובה", False, _SECTIONS, 1)
        assert stage == "SECTION_WALK" and key == "education"

    def test_past_last_section_composes(self):
        stage, _ = tailor._infer_tailor_stage(_asst(3), "x", False, _SECTIONS, 3)
        assert stage == "COMPOSE"

    def test_done_signal_composes(self):
        stage, _ = tailor._infer_tailor_stage(_asst(1), "מספיק בוא נכתוב", False, _SECTIONS, 1)
        assert stage == "COMPOSE"

    def test_composed_is_amend(self):
        stage, _ = tailor._infer_tailor_stage(_asst(1), "תוריד Excel", False, _SECTIONS, 1,
                                              composed=True)
        assert stage == "AMEND"


class TestDegrade:
    def test_no_client_returns_stub(self):
        out = tailor.tailor_turn("my cv", {"title": "Analyst"}, [], "", None, "model")
        assert out["proposed_cv"] is None
        assert out["done"] is False
        assert isinstance(out["reply"], str) and out["reply"]


class TestSectionWalkTurn:
    """tailor_turn during the section-walk: advances the pointer on section_done,
    surfaces the mode + section, never edits the CV."""

    def _patch(self, monkeypatch, canned):
        cap = {}
        def fake(client, model, context, system, max_tokens=900):
            cap["context"] = context; cap["system"] = system
            return canned
        monkeypatch.setattr(tailor.matching, "reply_json", fake)
        return cap

    def test_section_done_advances_index(self, monkeypatch):
        self._patch(monkeypatch, {"reply": "מעולה", "proposed_cv": None, "section_done": True})
        out = tailor.tailor_turn("STUB CV", {"title": "Analyst", "skills": ["sql"]},
                                 [{"role": "assistant", "content": "q"}], "תיכון, שנת 2026",
                                 object(), "model", has_cv=False, section_index=1)
        assert out["proposed_cv"] is None
        assert out["section_done"] is True
        assert out["next_section_index"] == 2
        assert out["total_sections"] == len(tailor.prompts.TAILOR_SECTIONS)

    def test_section_not_done_holds_index(self, monkeypatch):
        self._patch(monkeypatch, {"reply": "ספר עוד", "proposed_cv": None, "section_done": False})
        out = tailor.tailor_turn("STUB CV", {"title": "Analyst"},
                                 [{"role": "assistant", "content": "q"}], "לא בטוח",
                                 object(), "model", section_index=1)
        assert out["next_section_index"] == 1

    def test_mode_and_section_in_context(self, monkeypatch):
        cap = self._patch(monkeypatch, {"reply": "?", "proposed_cv": None, "section_done": False})
        tailor.tailor_turn("REAL CV TEXT", {"title": "Analyst"}, [], "",
                           object(), "model", has_cv=True, section_index=0)
        assert "enhance" in cap["context"]              # has_cv -> enhance mode surfaced
        assert "SECTION_WALK" in cap["context"]
        assert "הסעיף הבא" in cap["context"]            # next section provided -> agent can chain
        assert "## השלב שלך: SECTION_WALK" in cap["system"]

    def test_compose_passes_proposed_cv_through(self, monkeypatch):
        self._patch(monkeypatch, {"reply": "הנה השכתוב",
                                  "proposed_cv": "REWRITTEN CV", "done": True})
        hist = [{"role": "assistant", "content": "q"}]
        out = tailor.tailor_turn("My CV.", {"title": "Analyst", "skills": ["sql"]},
                                 hist, "מספיק", object(), "model")   # closing -> COMPOSE
        assert out["proposed_cv"] == "REWRITTEN CV"
        assert out["done"] is True


class TestAmendStage:
    def test_composed_non_closing_is_amend(self):
        hist = [{"role": "assistant", "content": "rewrite"}]   # a rewrite was delivered
        stage, key = tailor._infer_tailor_stage(hist, "תוריד את המשפט על Excel", False,
                                                _SECTIONS, 1, composed=True)
        assert stage == "AMEND" and key == ""

    def test_composed_done_exits_without_rewrite(self):
        out = tailor.tailor_turn("CURRENT CV", {"title": "Analyst", "skills": ["sql"]},
                                 [{"role": "assistant", "content": "rewrite"}],
                                 "מספיק תודה", object(), "model", composed=True)
        assert out["done"] is True
        assert out["proposed_cv"] is None

    def test_amend_turn_uses_amend_system_and_ships_cv(self, monkeypatch):
        captured = {}
        def fake(client, model, context, system, max_tokens=900):
            captured["system"] = system; captured["context"] = context
            return {"reply": "הורדתי", "proposed_cv": "EDITED CV", "done": False}
        monkeypatch.setattr(tailor.matching, "reply_json", fake)
        out = tailor.tailor_turn("CURRENT CV", {"title": "Analyst", "skills": ["sql"]},
                                 [{"role": "assistant", "content": "rewrite"}],
                                 "תוריד את המשפט על Excel", object(), "model", composed=True)
        assert "## השלב שלך: AMEND" in captured["system"]
        assert "CURRENT CV" in captured["context"]      # current CV is the base to edit
        assert out["proposed_cv"] == "EDITED CV"


class TestStageScopedSystemPrompt:
    """The system prompt is stage-scoped: only the current stage's instructions,
    plus the static preamble; CV structure only when composing."""

    def _patch(self, monkeypatch, canned):
        captured = {}
        def fake_reply_json(client, model, context, system, max_tokens=900):
            captured["context"] = context
            captured["system"] = system
            return canned
        monkeypatch.setattr(tailor.matching, "reply_json", fake_reply_json)
        return captured

    _H_SECTION = "## השלב שלך: SECTION_WALK"
    _H_COMPOSE = "## השלב שלך: COMPOSE ("

    def test_section_walk_system_no_structure(self, monkeypatch):
        cap = self._patch(monkeypatch, {"reply": "?", "proposed_cv": None, "section_done": False})
        tailor.tailor_turn("My CV.", {"title": "Analyst", "skills": ["sql"]},
                           [], "", object(), "model")          # opener -> SECTION_WALK
        sys = cap["system"]
        assert self._H_SECTION in sys
        assert self._H_COMPOSE not in sys               # other stages' instructions absent
        assert "מבנה קורות החיים" not in sys            # CV structure not loaded yet
        assert "כללי ברזל" in sys                       # iron rules always present

    def test_compose_system_includes_structure(self, monkeypatch):
        cap = self._patch(monkeypatch, {"reply": "ok", "proposed_cv": "CV", "done": True})
        hist = [{"role": "assistant", "content": "q"}]
        tailor.tailor_turn("My CV.", {"title": "Analyst", "skills": ["sql"]},
                           hist, "מספיק", object(), "model")    # closing -> COMPOSE
        sys = cap["system"]
        assert self._H_COMPOSE in sys
        assert "מבנה קורות החיים" in sys                # structure loaded when composing
        assert self._H_SECTION not in sys               # other stages' instructions absent
        # COMPOSE ships CV body in context
        assert "=== קורות חיים נוכחיים ===" in cap["context"]
