# -*- coding: utf-8 -*-
"""
cv/tests/test_jd_keywords.py
TDD tests for cv/jd_keywords.py

Run with:
    cd project && python -m pytest cv/tests/test_jd_keywords.py -v
"""

import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Ensure 'project' is on sys.path so `from cv.jd_keywords import ...` works
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, "..", ".."))  # …/project
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from cv.jd_keywords import extract, coverage


# ---------------------------------------------------------------------------
# extract() tests
# ---------------------------------------------------------------------------

class TestExtract:

    def test_skills_field_terms_rank_above_description_only_terms(self):
        """Terms in 'skills' (weight 3) should rank above description-only (weight 1)."""
        job = {
            "title": "",
            "skills": "sql",
            "description": "experience with spark hadoop",
        }
        result = extract(job, top_k=3)
        # sql appears in skills (score 3), spark & hadoop in description (score 1 each)
        assert result[0] == "sql"

    def test_title_terms_rank_above_description_only_terms(self):
        """Terms in 'title' (weight 2) should rank above description-only (weight 1)."""
        job = {
            "title": "python developer",
            "skills": "",
            "description": "uses java and c++",
        }
        result = extract(job, top_k=3)
        # 'python' and 'developer' from title (score 2) > 'java', 'c++' from description (score 1)
        # developer is a stopword? Let's check; 'python' should rank high
        assert "python" in result[:2]

    def test_skills_outranks_title_outranks_description(self):
        """skills(3) > title(2) > description(1) in ranking."""
        job = {
            "title": "go engineer",
            "skills": "rust",
            "description": "writes python code",
        }
        result = extract(job, top_k=3)
        # rust=3, go=2 (or engineer=2), python=1
        assert result[0] == "rust"

    def test_deduplication(self):
        """A term appearing in skills, title, and description is returned once."""
        job = {
            "title": "sql analyst",
            "skills": "sql, python",
            "description": "advanced sql required, python scripting",
        }
        result = extract(job, top_k=12)
        assert result.count("sql") == 1
        assert result.count("python") == 1

    def test_top_k_cap(self):
        """Result length never exceeds top_k."""
        job = {
            "title": "data engineer",
            "skills": "sql, spark, hadoop, kafka, airflow, dbt",
            "description": "experience with flink, scala, java, python, go, rust, c++",
        }
        result = extract(job, top_k=5)
        assert len(result) <= 5

    def test_skills_as_list(self):
        """skills field may be a list of strings instead of a comma string."""
        job = {
            "title": "",
            "skills": ["spark", "kafka", "flink"],
            "description": "",
        }
        result = extract(job, top_k=5)
        assert "spark" in result
        assert "kafka" in result
        assert "flink" in result

    def test_skills_as_list_ranked_higher_than_description(self):
        """skills list items (weight 3) outrank description items (weight 1)."""
        job = {
            "title": "",
            "skills": ["spark"],
            "description": "uses hadoop and hive extensively",
        }
        result = extract(job, top_k=3)
        assert result[0] == "spark"

    def test_stopwords_filtered(self):
        """Generic stopwords are excluded from results."""
        job = {
            "title": "the role",
            "skills": "strong experience with team",
            "description": "working in a great environment",
        }
        result = extract(job, top_k=12)
        stopwords = {"the", "a", "an", "strong", "experience", "with", "team",
                     "working", "in", "great", "environment", "role", "skills",
                     "skill", "knowledge", "work"}
        for term in result:
            assert term not in stopwords, f"Stopword '{term}' should not appear in result"

    def test_ties_broken_alphabetically(self):
        """Terms with equal scores are sorted alphabetically."""
        job = {
            "title": "",
            "skills": "banana apple cherry",
            "description": "",
        }
        result = extract(job, top_k=3)
        # All three have score 3 (from skills); alphabetical order expected
        assert result == ["apple", "banana", "cherry"]

    def test_empty_job_dict_returns_empty_list(self):
        """Empty job dict returns [] without raising."""
        assert extract({}) == []

    def test_none_job_returns_empty_list(self):
        """None-ish / missing fields degrade gracefully."""
        assert extract(None) == []

    def test_all_blank_fields_returns_empty_list(self):
        """Job with all blank string fields returns []."""
        job = {"title": "", "skills": "", "description": ""}
        assert extract(job) == []

    def test_returns_list_type(self):
        """extract always returns a list."""
        job = {"title": "engineer", "skills": "python", "description": ""}
        result = extract(job)
        assert isinstance(result, list)

    def test_combined_weights_accumulate(self):
        """A term present in skills, title, and description accumulates all weights (3+2+1=6)."""
        job = {
            "title": "python lead",
            "skills": "python, java",
            "description": "build python services and java microservices",
        }
        result = extract(job, top_k=5)
        # python: 3+2+1=6, java: 3+1=4, lead: 2 (if not stopword)
        assert result[0] == "python"
        assert result[1] == "java"


# ---------------------------------------------------------------------------
# coverage() tests
# ---------------------------------------------------------------------------

class TestCoverage:

    def test_basic_split_covered_vs_missing(self):
        """Classic scenario: cv has sql but not spark."""
        job = {
            "title": "",
            "skills": "sql, spark",
            "description": "",
        }
        result = coverage("experienced in sql and python", job, top_k=5)
        assert "sql" in result["covered"]
        assert "spark" in result["missing"]

    def test_hadoop_missing_when_not_in_cv(self):
        """Full example from spec: skills='sql, spark, hadoop', cv has sql and python."""
        job = {
            "title": "",
            "skills": "sql, spark, hadoop",
            "description": "",
        }
        result = coverage("experienced in sql and python", job, top_k=5)
        assert "sql" in result["covered"]
        assert "spark" in result["missing"]
        assert "hadoop" in result["missing"]

    def test_returns_dict_with_covered_and_missing_keys(self):
        """coverage() always returns {"covered": list, "missing": list}."""
        job = {"title": "engineer", "skills": "python", "description": ""}
        result = coverage("I use python daily", job, top_k=5)
        assert set(result.keys()) == {"covered", "missing"}
        assert isinstance(result["covered"], list)
        assert isinstance(result["missing"], list)

    def test_all_covered(self):
        """If cv contains all terms, missing is empty."""
        job = {"title": "", "skills": "sql python spark", "description": ""}
        result = coverage("sql python spark developer", job, top_k=5)
        assert "sql" in result["covered"]
        assert "python" in result["covered"]
        assert "spark" in result["covered"]
        assert result["missing"] == []

    def test_none_covered(self):
        """If cv contains none of the terms, covered is empty."""
        job = {"title": "", "skills": "cobol fortran", "description": ""}
        result = coverage("experienced python developer", job, top_k=5)
        assert result["covered"] == []
        assert "cobol" in result["missing"]
        assert "fortran" in result["missing"]

    def test_importance_order_preserved_covered(self):
        """covered list preserves importance order (most important first)."""
        job = {
            "title": "",
            "skills": "spark sql hadoop",  # all score 3; alphabetical: hadoop, spark, sql
            "description": "",
        }
        result = coverage("spark sql hadoop developer", job, top_k=3)
        # alphabetical tie-breaking → hadoop, spark, sql
        assert result["covered"] == ["hadoop", "spark", "sql"]

    def test_importance_order_preserved_missing(self):
        """missing list preserves importance order (most important first)."""
        job = {
            "title": "",
            "skills": "spark",          # score 3
            "description": "uses hadoop",  # score 1
        }
        result = coverage("experienced developer", job, top_k=5)
        # spark (score 3) should appear before hadoop (score 1)
        assert result["missing"].index("spark") < result["missing"].index("hadoop")

    def test_empty_job_returns_empty_dicts(self):
        """Empty job degrades gracefully."""
        result = coverage("experienced python developer", {})
        assert result == {"covered": [], "missing": []}

    def test_none_job_returns_empty_dicts(self):
        """None job degrades gracefully."""
        result = coverage("experienced python developer", None)
        assert result == {"covered": [], "missing": []}

    def test_empty_cv_text(self):
        """Empty CV text means everything is missing."""
        job = {"title": "", "skills": "python spark", "description": ""}
        result = coverage("", job, top_k=5)
        assert result["covered"] == []
        assert set(result["missing"]) == {"python", "spark"}

    def test_covered_and_missing_are_disjoint(self):
        """No term appears in both covered and missing."""
        job = {"title": "data engineer", "skills": "sql spark kafka", "description": "python required"}
        result = coverage("I use sql and python", job, top_k=10)
        covered_set = set(result["covered"])
        missing_set = set(result["missing"])
        assert covered_set.isdisjoint(missing_set)

    def test_top_k_applies(self):
        """coverage respects top_k passed to extract."""
        job = {
            "title": "",
            "skills": "sql spark hadoop kafka airflow dbt flink scala java golang",
            "description": "",
        }
        result = coverage("sql spark hadoop", job, top_k=3)
        total = len(result["covered"]) + len(result["missing"])
        assert total <= 3


# ---------------------------------------------------------------------------
# must_haves() tests
# ---------------------------------------------------------------------------
from cv.jd_keywords import must_haves


class TestMustHaves:
    def test_whole_phrase_skill_missing_vs_covered(self):
        job = {"title": "Analyst", "skills": ["project management", "sql"],
               "description": ""}
        cv = "Built dashboards in SQL and Excel for the finance team."
        out = must_haves(cv, job)
        assert "sql" in [t.lower() for t in out["covered"]]
        assert "project management" in [t.lower() for t in out["missing"]]

    def test_multiword_phrase_fully_present_is_covered(self):
        job = {"title": "ML Eng", "skills": ["machine learning"], "description": ""}
        cv = "Worked as a machine learning engineer on recommendation models."
        out = must_haves(cv, job)
        assert "machine learning" in [t.lower() for t in out["covered"]]
        assert out["missing"] == []

    def test_cue_context_description_term_picked_up(self):
        job = {"title": "DevOps", "skills": [],
               "description": "Experience with Kubernetes is required. Coffee is a plus."}
        out = must_haves("Linux sysadmin with Bash scripting.", job)
        missing_l = [t.lower() for t in out["missing"]]
        assert "kubernetes" in missing_l          # in the 'required' clause
        assert "coffee" not in missing_l          # 'plus' clause, no cue word

    def test_skills_string_form_is_split_on_commas(self):
        job = {"title": "Analyst", "skills": "tableau, sql", "description": ""}
        out = must_haves("I know SQL well.", job)
        assert "tableau" in [t.lower() for t in out["missing"]]
        assert "sql" in [t.lower() for t in out["covered"]]

    def test_dedup_skill_and_cue_term(self):
        job = {"title": "Analyst", "skills": ["sql"],
               "description": "SQL is required for this role."}
        out = must_haves("No databases here.", job)
        # 'sql' appears in both sources but must appear once total across the lists
        all_terms = [t.lower() for t in out["missing"] + out["covered"]]
        assert all_terms.count("sql") == 1

    def test_top_k_caps_each_list(self):
        job = {"title": "X", "skills": ["a1 skill", "b2 skill", "c3 skill",
                                        "d4 skill", "e5 skill"], "description": ""}
        out = must_haves("", job, top_k=3)
        assert len(out["missing"]) <= 3

    def test_thin_and_none_job_are_safe(self):
        assert must_haves("anything", {}) == {"covered": [], "missing": []}
        assert must_haves("anything", {"title": "Only a title"}) == {"covered": [], "missing": []}
        assert must_haves("", None) == {"covered": [], "missing": []}
