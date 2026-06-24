# project/tests/test_router.py
# Tests for Tasks 4, 5, and 6 — owned by Beni (agent_runner teammate).

import pytest

# ---------------------------------------------------------------------------
# Task 4 — ANALYTICS_CTX smoke (will be SKIPPED until analytics columns exist
# in the gold file, even if analytics.py is present — gold rebuild required)
# ---------------------------------------------------------------------------

def test_analytics_ctx_smoke():
    """
    Verify agent_runner exposes ANALYTICS_CTX and the widened gold columns.
    SKIPPED if analytics.py does not exist yet (Ada's parallel work) OR if the
    gold file has not been regenerated to include the analytics columns yet.
    """
    try:
        import analytics
    except ModuleNotFoundError:
        pytest.skip("analytics.py not yet created (pending Ada's work)")

    import agent_runner as ag

    analytics_cols = ("skills", "salary", "remote_allowed", "job_industry")
    missing = [c for c in analytics_cols if c not in ag.df.columns]
    if missing:
        pytest.skip(
            f"Gold file not yet regenerated with analytics columns (missing: {missing}). "
            "ANALYTICS_CTX smoke is pending gold rebuild."
        )

    assert isinstance(ag.ANALYTICS_CTX, analytics.Ctx)


# ---------------------------------------------------------------------------
# Task 5 — parse_query emits mode (LLM mocked at _llm_parse boundary)
# ---------------------------------------------------------------------------

import agent_runner as ag


def _patch_parse(monkeypatch, payload):
    monkeypatch.setattr(ag, "_llm_parse", lambda prompt: payload)


def test_mode_search(monkeypatch):
    _patch_parse(monkeypatch, {"role_en": "data analyst", "state": "NY", "mode": "search"})
    assert ag.parse_query("data analyst in NY")["mode"] == "search"


def test_mode_data_question(monkeypatch):
    _patch_parse(monkeypatch, {"role_en": "data analyst", "state": "NY",
                               "mode": "data_question"})
    assert ag.parse_query("average data analyst salary in NY?")["mode"] == "data_question"


def test_mode_advice(monkeypatch):
    _patch_parse(monkeypatch, {"role_en": "", "mode": "advice"})
    assert ag.parse_query("how do I prep for a data interview?")["mode"] == "advice"


def test_invalid_mode_defaults_to_search_when_searchy(monkeypatch):
    _patch_parse(monkeypatch, {"role_en": "nurse", "state": "TX", "mode": "garbage"})
    out = ag.parse_query("nurse in TX")
    assert out["mode"] == "search"          # unknown mode + searchy signal -> search


# ---------------------------------------------------------------------------
# Task 6 — advice_reply (mocked at matching.reply boundary)
# ---------------------------------------------------------------------------

def test_advice_reply_degrades_without_client(monkeypatch):
    monkeypatch.setattr(ag, "claude_client", None)
    out = ag.advice_reply("how do I prep?", history=[], matches=[], profile=None)
    assert isinstance(out, str) and out          # graceful Hebrew fallback, never empty


def test_advice_reply_builds_grounded_context(monkeypatch):
    captured = {}

    def fake_reply(client, model, context, system, max_tokens=300):
        captured["context"] = context
        return "תשובת ייעוץ קצרה."

    monkeypatch.setattr(ag, "claude_client", object())   # truthy
    monkeypatch.setattr(ag.matching, "reply", fake_reply)
    out = ag.advice_reply("is the first one good?",
                          history=[{"role": "user", "content": "data analyst NY"}],
                          matches=[{"rank": 1, "title": "Data Analyst", "company": "Acme",
                                    "location": "New York, NY", "experience_level": "Entry level",
                                    "salary_band": 4}],
                          profile={"titles_held": ["data analyst"]})
    assert out == "תשובת ייעוץ קצרה."
    assert "Data Analyst" in captured["context"]         # on-screen job grounded in
    assert "data analyst NY" in captured["context"]      # history grounded in
