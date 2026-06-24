# -*- coding: utf-8 -*-
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

import matching
from cv import scorecard


@pytest.fixture
def vec():
    v = TfidfVectorizer()
    v.fit(["data analyst sql python", "registered nurse hospital", "bi analyst tableau"])
    return v


def test_split_have_gap_deterministic():
    profile = {"skills": ["sql", "python"]}
    have, gap = scorecard.split_have_gap(["sql", "python", "tableau"], profile, "experienced in sql and python")
    assert have == ["sql", "python"]
    assert gap == ["tableau"]


def test_split_have_gap_uses_cv_text():
    # skill present in the raw CV text even if not in profile.skills counts as 'have'
    have, gap = scorecard.split_have_gap(["excel"], {"skills": []}, "proficient with Excel dashboards")
    assert have == ["excel"]
    assert gap == []


def test_build_scores_and_detail_only_on_first(monkeypatch, vec):
    monkeypatch.setattr(matching, "reply_json", lambda *a, **k: {"skills": ["sql", "python", "tableau"]})
    profile = {"skills": ["sql", "python"], "seniority": "entry", "years": 0}
    cards = scorecard.build(["data analyst", "bi analyst", ""],  # blank dropped
                            profile, "data analyst sql python", vec, object(), "m")
    assert len(cards) == 2                               # blank role removed
    assert all(isinstance(c["match"], int) and 0 <= c["match"] <= 100 for c in cards)
    assert all(c["seniority"] for c in cards)            # band mapped to Hebrew
    assert cards[0]["have"] == ["sql", "python"] and cards[0]["gap"] == ["tableau"]
    assert "have" not in cards[1]                        # detail only on the first role


def test_build_degrades_without_client(vec):
    cards = scorecard.build(["data analyst"], {"skills": ["sql"]}, "data analyst sql", vec, None, "m")
    assert cards[0]["match"] >= 0
    assert cards[0]["have"] == [] and cards[0]["gap"] == []   # no client -> no skill vocab
