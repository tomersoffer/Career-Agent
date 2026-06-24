# -*- coding: utf-8 -*-
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from cv import role_fit


@pytest.fixture
def vec():
    corpus = [
        "data analyst sql python excel reporting dashboards",
        "registered nurse patient care hospital",
        "chief executive officer strategy leadership",
        "software engineer java backend",
    ]
    v = TfidfVectorizer()
    v.fit(corpus)
    return v


def test_role_fit_prefers_matching_role(vec):
    # Role-level: the candidate's held titles + skills, compared role-to-role.
    profile = {"titles_held": ["data analyst"], "skills": ["sql", "python", "excel"]}
    cv = "experienced in data analysis sql python excel reporting dashboards"
    da = role_fit.role_fit(profile, cv, "data analyst", vec)
    nurse = role_fit.role_fit(profile, cv, "registered nurse", vec)
    assert da > nurse


def test_role_fit_falls_back_to_cv_when_no_titles_or_skills(vec):
    # Empty profile -> fall back to the full CV text so we never score a blank vector.
    score = role_fit.role_fit({}, "data analyst sql python", "data analyst", vec)
    assert score > 0


def test_seniority_gap_student_to_ceo_is_stretch():
    g = role_fit.seniority_gap({"seniority": "entry", "years": 0},
                               "Chief Executive Officer")
    assert g["band"] == "stretch"
    assert g["gap"] >= 2


def test_seniority_gap_matching_is_fit():
    g = role_fit.seniority_gap({"seniority": "mid-senior", "years": 6},
                               "Senior Data Analyst")
    assert g["band"] == "fit"


def test_seniority_gap_overqualified_is_under():
    g = role_fit.seniority_gap({"seniority": "executive", "years": 20},
                               "Junior Analyst")
    assert g["band"] == "under"


def test_cv_rank_years_fallback():
    g = role_fit.seniority_gap({"seniority": "", "years": 15}, "Manager")
    assert g["cv_rank"] >= 5


def test_assess_shape_and_skips_blank(vec):
    out = role_fit.assess({"skills": ["sql"], "seniority": "entry", "years": 0},
                          "data analyst sql", ["data analyst", ""], vec)
    assert len(out) == 1                      # blank role dropped
    assert set(out[0]) == {"role", "role_fit", "seniority_gap"}
