# -*- coding: utf-8 -*-
"""
tests/test_cv.py
Unit tests for the cv/ package.

Run with:
    python -m pytest project/cv/tests/test_cv.py -v
or from the project root:
    python -m pytest project/cv/tests/ -v
"""

import io
import sys
import os

import pytest

# ---------------------------------------------------------------------------
# Make sure 'project' directory is on sys.path so that `import matching` and
# `import cv.*` both resolve correctly regardless of where pytest is invoked.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, "..", ".."))  # …/project
_REPO_ROOT = os.path.abspath(os.path.join(_PROJECT, ".."))  # …/Agent
for _p in [_PROJECT, _REPO_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cv.extract import extract_text, ExtractError
from cv.titles import suggest_titles
from cv.fit import fit_and_gap
from cv.tailor import tailor_turn


# ===========================================================================
# Helpers
# ===========================================================================

def _make_docx_bytes(text: str) -> bytes:
    """Create a minimal in-memory DOCX containing *text*."""
    import docx  # python-docx
    doc = docx.Document()
    doc.add_paragraph(text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _make_tiny_vectorizer(titles_list):
    """Return a (fitted TfidfVectorizer, sparse matrix, titles list) triple."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    vec = TfidfVectorizer()
    mat = vec.fit_transform([t.lower() for t in titles_list])
    return vec, mat, titles_list


# ===========================================================================
# 1. extract_text
# ===========================================================================

class TestExtractText:
    def test_docx_returns_nonempty(self):
        content = "Software Engineer with 5 years of Python and Django experience."
        docx_bytes = _make_docx_bytes(content)
        result = extract_text(docx_bytes, "cv.docx")
        assert isinstance(result, str)
        assert len(result.strip()) > 0
        assert "Python" in result or "python" in result.lower()

    def test_docx_multiline(self):
        lines = ["Name: John Doe", "Skills: Python, SQL, Machine Learning"]
        docx_bytes = _make_docx_bytes("\n".join(lines))
        result = extract_text(docx_bytes, "resume.docx")
        assert "Python" in result or "python" in result.lower()

    def test_unsupported_extension_raises(self):
        with pytest.raises(ExtractError) as exc_info:
            extract_text(b"some bytes", "cv.txt")
        assert "unsupported" in str(exc_info.value).lower() or ".txt" in str(exc_info.value)

    def test_unsupported_xlsx_raises(self):
        with pytest.raises(ExtractError):
            extract_text(b"\x50\x4b\x03\x04", "cv.xlsx")

    def test_empty_docx_raises(self):
        """A DOCX with only empty paragraphs should raise ExtractError."""
        docx_bytes = _make_docx_bytes("   \n   \n   ")
        with pytest.raises(ExtractError) as exc_info:
            extract_text(docx_bytes, "empty.docx")
        assert "could not extract" in str(exc_info.value).lower() or "empty" in str(exc_info.value).lower()

    def test_corrupt_pdf_raises(self):
        """Random bytes labelled as .pdf must raise ExtractError (not crash)."""
        with pytest.raises(ExtractError):
            extract_text(b"not a real pdf", "cv.pdf")


# ===========================================================================
# 2. suggest_titles
# ===========================================================================

FAKE_TITLES = [
    "Data Scientist",
    "Machine Learning Engineer",
    "Data Analyst",
    "Software Engineer",
    "Product Manager",
    "Business Analyst",
    "DevOps Engineer",
    "Frontend Developer",
    "Backend Developer",
    "Data Engineer",
]

CV_DATA_TEXT = (
    "Experienced data scientist with strong skills in Python, machine learning, "
    "statistical modelling, data analysis, SQL, and Spark. "
    "Worked on NLP projects and built ML pipelines."
)


class TestSuggestTitles:
    def setup_method(self):
        self.vec, self.mat, self.titles = _make_tiny_vectorizer(FAKE_TITLES)

    def test_returns_list(self):
        result = suggest_titles(CV_DATA_TEXT, self.vec, self.mat, self.titles)
        assert isinstance(result, list)

    def test_titles_are_subset_of_input(self):
        result = suggest_titles(CV_DATA_TEXT, self.vec, self.mat, self.titles)
        for t in result:
            assert t in FAKE_TITLES, f"'{t}' is not in the input title list"

    def test_top_k_default(self):
        """Default top_k=8 but only 10 titles available; should return at most 8."""
        result = suggest_titles(CV_DATA_TEXT, self.vec, self.mat, self.titles, top_k=8)
        assert len(result) <= 8

    def test_top_k_small(self):
        result = suggest_titles(CV_DATA_TEXT, self.vec, self.mat, self.titles, top_k=3)
        assert len(result) == 3

    def test_top_k_larger_than_available(self):
        """If top_k > distinct titles, return all distinct titles (no error)."""
        result = suggest_titles(CV_DATA_TEXT, self.vec, self.mat, self.titles, top_k=100)
        assert len(result) <= len(FAKE_TITLES)

    def test_no_duplicates(self):
        result = suggest_titles(CV_DATA_TEXT, self.vec, self.mat, self.titles, top_k=8)
        lower = [t.lower() for t in result]
        assert len(lower) == len(set(lower)), "Duplicate titles returned"


# ===========================================================================
# 3. fit_and_gap
# ===========================================================================

CV_FIT_TEXT = (
    "Python developer with 4 years of experience. "
    "Proficient in Python, SQL, pandas, and scikit-learn. "
    "Built ETL pipelines and dashboards."
)

JOB_SKILLS_TEXT = "python, sql, tableau"

PROFILE_EMPTY = {
    "skills": [],
    "titles_held": [],
    "seniority": "",
    "years": None,
    "city": "",
    "state": "",
    "domains": [],
}


class TestFitAndGap:
    def setup_method(self):
        """Build a small vectorizer trained on the CV text and job title."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        corpus = [
            CV_FIT_TEXT.lower(),
            "data analyst",
            "python developer",
            "business analyst",
        ]
        self.vec = TfidfVectorizer()
        self.vec.fit(corpus)

    def test_returns_expected_keys(self):
        job = {"title": "Data Analyst", "skills": JOB_SKILLS_TEXT}
        result = fit_and_gap(PROFILE_EMPTY, CV_FIT_TEXT, job, self.vec)
        assert set(result.keys()) == {"fit_score", "missing", "present"}

    def test_fit_score_in_range(self):
        job = {"title": "Data Analyst", "skills": JOB_SKILLS_TEXT}
        result = fit_and_gap(PROFILE_EMPTY, CV_FIT_TEXT, job, self.vec)
        assert 0.0 <= result["fit_score"] <= 1.0

    def test_tableau_in_missing(self):
        """CV mentions python and sql but NOT tableau → tableau must be in missing."""
        job = {"title": "Data Analyst", "skills": "python, sql, tableau"}
        result = fit_and_gap(PROFILE_EMPTY, CV_FIT_TEXT, job, self.vec)
        assert "tableau" in result["missing"], (
            f"Expected 'tableau' in missing, got missing={result['missing']}"
        )

    def test_python_in_present(self):
        """CV explicitly mentions python → python must be in present."""
        job = {"title": "Data Analyst", "skills": "python, sql, tableau"}
        result = fit_and_gap(PROFILE_EMPTY, CV_FIT_TEXT, job, self.vec)
        assert "python" in result["present"], (
            f"Expected 'python' in present, got present={result['present']}"
        )

    def test_sql_in_present(self):
        job = {"title": "Data Analyst", "skills": "python, sql, tableau"}
        result = fit_and_gap(PROFILE_EMPTY, CV_FIT_TEXT, job, self.vec)
        assert "sql" in result["present"], (
            f"Expected 'sql' in present, got present={result['present']}"
        )

    def test_missing_and_present_are_disjoint(self):
        job = {"title": "Data Analyst", "skills": "python, sql, tableau"}
        result = fit_and_gap(PROFILE_EMPTY, CV_FIT_TEXT, job, self.vec)
        overlap = set(result["missing"]) & set(result["present"])
        assert not overlap, f"Overlap between missing and present: {overlap}"

    def test_profile_skills_extend_cv_terms(self):
        """Skills listed in the profile should count as 'present'."""
        profile_with_tableau = {**PROFILE_EMPTY, "skills": ["tableau"]}
        job = {"title": "Data Analyst", "skills": "python, sql, tableau"}
        result = fit_and_gap(profile_with_tableau, CV_FIT_TEXT, job, self.vec)
        assert "tableau" in result["present"]


# ===========================================================================
# 4. tailor_turn (stubbed: claude_client=None)
# ===========================================================================

class TestTailorTurn:
    CV_TEXT = "Experienced software engineer with Python, Django, REST APIs."
    JOB = {
        "title": "Backend Python Developer",
        "company": "Acme Corp",
        "description": "Looking for Python expert with Django and PostgreSQL skills.",
        "skills": "python, django, postgresql",
    }

    def test_keys_present_when_llm_disabled(self):
        result = tailor_turn(
            cv_text=self.CV_TEXT,
            job=self.JOB,
            history=[],
            user_msg="מה אני צריך לשפר בקורות החיים שלי?",
            claude_client=None,
            model="claude-sonnet-4-6",
        )
        assert "reply" in result, "Key 'reply' missing from tailor_turn output"
        assert "proposed_cv" in result, "Key 'proposed_cv' missing from tailor_turn output"

    def test_proposed_cv_is_none_when_llm_disabled(self):
        result = tailor_turn(
            cv_text=self.CV_TEXT,
            job=self.JOB,
            history=[],
            user_msg="תן לי עצות כלליות",
            claude_client=None,
            model="claude-sonnet-4-6",
        )
        assert result["proposed_cv"] is None

    def test_reply_is_string_when_llm_disabled(self):
        result = tailor_turn(
            cv_text=self.CV_TEXT,
            job=self.JOB,
            history=[{"role": "user", "content": "שלום"}, {"role": "assistant", "content": "שלום!"}],
            user_msg="כיצד אשפר את הסיכום שלי?",
            claude_client=None,
            model="claude-sonnet-4-6",
        )
        assert isinstance(result["reply"], str)
        assert len(result["reply"]) > 0


# ===========================================================================
# Run directly (in addition to pytest)
# ===========================================================================
if __name__ == "__main__":
    import traceback

    suites = [
        TestExtractText,
        TestSuggestTitles,
        TestFitAndGap,
        TestTailorTurn,
    ]
    passed = failed = 0
    for Suite in suites:
        obj = Suite()
        for name in [m for m in dir(Suite) if m.startswith("test_")]:
            method = getattr(obj, name)
            if hasattr(obj, "setup_method"):
                obj.setup_method()
            try:
                method()
                print(f"  PASS  {Suite.__name__}::{name}")
                passed += 1
            except Exception as exc:
                print(f"  FAIL  {Suite.__name__}::{name}  →  {exc}")
                traceback.print_exc()
                failed += 1
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
