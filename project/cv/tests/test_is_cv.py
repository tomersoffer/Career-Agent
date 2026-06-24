# -*- coding: utf-8 -*-
"""
tests/test_is_cv.py
Unit tests for the is_cv classification folded into build_profile().

Run with:
    python -m pytest project/cv/tests/test_is_cv.py -v
"""

import os
import sys

# Put 'project' on sys.path so `import matching` and `import cv.*` resolve.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, "..", ".."))  # …/project
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

import matching
from cv import profile as cv_profile


class _DummyClient:
    """Stand-in for an Anthropic client — build_profile only checks it's not None."""
    pass


def test_is_cv_defaults_true_when_degraded():
    # claude_client=None -> graceful degrade -> never reject a real CV.
    p = cv_profile.build_profile("some cv text", None, "model")
    assert p["is_cv"] is True


def test_is_cv_defaults_true_on_malformed_json(monkeypatch):
    # LLM returned something that is not a dict -> fail open.
    monkeypatch.setattr(matching, "reply_json", lambda *a, **k: None)
    p = cv_profile.build_profile("text", _DummyClient(), "model")
    assert p["is_cv"] is True


def test_is_cv_defaults_true_when_key_absent(monkeypatch):
    # A valid profile dict that simply omits is_cv -> fail open.
    monkeypatch.setattr(matching, "reply_json",
                        lambda *a, **k: {"name": "Dana", "skills": ["python"]})
    p = cv_profile.build_profile("text", _DummyClient(), "model")
    assert p["is_cv"] is True


def test_is_cv_false_is_parsed(monkeypatch):
    monkeypatch.setattr(matching, "reply_json",
                        lambda *a, **k: {"is_cv": False, "cv_confidence": 0.95})
    p = cv_profile.build_profile("this is an invoice", _DummyClient(), "model")
    assert p["is_cv"] is False
    assert p["cv_confidence"] == 0.95


def test_cv_confidence_is_clamped(monkeypatch):
    monkeypatch.setattr(matching, "reply_json",
                        lambda *a, **k: {"is_cv": True, "cv_confidence": 4.2})
    p = cv_profile.build_profile("text", _DummyClient(), "model")
    assert p["cv_confidence"] == 1.0


def test_cv_confidence_default_zero_when_missing(monkeypatch):
    monkeypatch.setattr(matching, "reply_json",
                        lambda *a, **k: {"is_cv": True})
    p = cv_profile.build_profile("text", _DummyClient(), "model")
    assert p["cv_confidence"] == 0.0
