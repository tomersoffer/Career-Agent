import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

import analytics


def _ctx(df):
    """Build an analytics ctx with a REAL vectorizer fitted on the frame's titles."""
    vec = TfidfVectorizer()
    title_matrix = vec.fit_transform(df["title"].fillna("").str.lower())
    return analytics.Ctx(
        vectorizer=vec,
        title_matrix=title_matrix,
        valid_states=sorted(s for s in df["job_state"].dropna().unique() if s),
        seniority_rank={"intern": 1, "entry": 2, "associate": 3,
                        "mid-senior": 4, "senior": 4, "director": 5, "executive": 6},
    )


@pytest.fixture
def df():
    return pd.DataFrame({
        "title": ["data analyst", "data analyst", "senior data analyst",
                  "software engineer", "nurse"],
        "job_state": ["NY", "NY", "CA", "NY", "TX"],
        "experience_rank": [2, 4, 4, 3, 2],
        "experience_level": ["Entry level", "Mid-Senior level", "Mid-Senior level",
                             "Associate", "Entry level"],
        "salary": [80000.0, 120000.0, 130000.0, 150000.0, np.nan],
        "remote_allowed": [0, 1, 1, 0, 0],
        "skills": ["sql, excel", "sql, python", "python, sql, tableau",
                   "java, python", "patient care"],
        "job_industry": ["Tech", "Tech", "Tech", "Tech", "Healthcare"],
    })


def test_salary_stats_filters_by_role_and_state(df):
    ctx = _ctx(df)
    out = analytics.salary_stats(df, {"role": "data analyst", "state": "NY"}, ctx)
    # Only the two NY data-analyst rows have salary (80k, 120k); the CA row is excluded by state.
    assert out["n"] == 2
    assert out["median"] == 100000
    assert out["function"] == "salary_stats"


def test_salary_stats_empty_returns_n_zero(df):
    ctx = _ctx(df)
    out = analytics.salary_stats(df, {"role": "astronaut", "state": "NY"}, ctx)
    assert out["n"] == 0
    assert out["median"] is None


def test_top_skills_counts_exploded_skills(df):
    ctx = _ctx(df)
    out = analytics.top_skills(df, {"role": "data analyst"}, ctx, )
    assert out["function"] == "top_skills"
    assert out["n"] == 3                      # 3 data-analyst rows
    # sql appears in all 3 -> ranked first
    assert out["items"][0][0] == "sql"


def test_count_jobs_with_remote_filter(df):
    ctx = _ctx(df)
    out = analytics.count_jobs(df, {"role": "data analyst", "remote": True}, ctx)
    assert out["function"] == "count_jobs"
    assert out["count"] == 2                   # the two remote_allowed==1 analyst rows


def test_remote_share(df):
    ctx = _ctx(df)
    out = analytics.remote_share(df, {"role": "data analyst"}, ctx)
    assert out["n"] == 3
    assert out["share_pct"] == 67              # 2 of 3 -> 66.7 -> 67


def test_experience_breakdown(df):
    ctx = _ctx(df)
    out = analytics.experience_breakdown(df, {"role": "data analyst"}, ctx)
    assert out["n"] == 3
    assert dict(out["items"])["Mid-Senior level"] == 2


def test_top_locations(df):
    ctx = _ctx(df)
    out = analytics.top_locations(df, {"role": "data analyst"}, ctx)
    assert out["items"][0] == ["NY", 2]        # most analyst rows are in NY


def test_top_industries(df):
    ctx = _ctx(df)
    out = analytics.top_industries(df, {"role": "data analyst"}, ctx)
    assert out["items"][0][0] == "Tech"


class _FakeClaude:
    """Minimal stand-in: yields a queued JSON string per messages.create call."""
    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.messages = self

    def create(self, **kwargs):
        text = self._payloads.pop(0)
        return type("M", (), {"content": [type("B", (), {"type": "text", "text": text})()]})()


def test_answer_data_question_maps_computes_narrates(df):
    ctx = _ctx(df)
    fake = _FakeClaude([
        '{"function": "salary_stats", "params": {"role": "data analyst", "state": "NY"}}',
        "השכר החציוני למשרות אנליסט נתונים בניו יורק הוא כ-100 אלף דולר.",
    ])
    reply = analytics.answer_data_question("מה השכר לאנליסט נתונים בניו יורק?",
                                           df, ctx, fake, "model-x")
    assert "100" in reply


def test_answer_data_question_rejects_unknown_function(df):
    ctx = _ctx(df)
    fake = _FakeClaude(['{"function": "drop_table", "params": {}}',
                        "מצטער, לא הצלחתי לחשב את זה."])
    reply = analytics.answer_data_question("?", df, ctx, fake, "model-x")
    assert isinstance(reply, str) and reply   # never raises; returns a graceful string
