# -*- coding: utf-8 -*-
import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from cv import gap


_JOB = {"title": "Data Analyst", "skills": ["SQL", "Tableau", "Python"],
        "description": "Strong SQL required. Experience with BI tools and A/B testing."}
_CV = "Daniel. Skills: Python, PostgreSQL, Power BI. Ran A/B tests and built dashboards."


def test_none_job_is_empty():
    assert gap.analyze_gaps(_CV, None, None, "m") == {"covered": [], "missing": []}


def test_llm_disabled_falls_back_to_deterministic():
    # client=None -> deterministic jd_keywords baseline, always a valid shape.
    r = gap.analyze_gaps(_CV, _JOB, None, "m")
    assert isinstance(r, dict) and "covered" in r and "missing" in r


class _FakeContent:
    def __init__(self, text): self.type = "text"; self.text = text
class _FakeMsg:
    def __init__(self, text): self.content = [_FakeContent(text)]
class _FakeMessages:
    def __init__(self, payload): self._payload = payload
    def create(self, **kw): return _FakeMsg(self._payload)
class _FakeClient:
    def __init__(self, payload): self.messages = _FakeMessages(payload)


def test_llm_result_is_normalized():
    # The semantic analyzer recognizes PostgreSQL covers SQL — something the token
    # matcher cannot. We stub the client to return that classification.
    payload = json.dumps({"covered": ["SQL", "Python", "A/B testing", "sql"],   # dup -> deduped
                          "missing": ["Tableau", "", "Tableau"]})              # blank/dup dropped
    r = gap.analyze_gaps(_CV, _JOB, _FakeClient(payload), "m")
    assert "SQL" in r["covered"] and "Python" in r["covered"]
    assert r["covered"].count("SQL") == 1            # case-insensitive dedupe
    assert r["missing"] == ["Tableau"]               # blanks + dups removed


def test_garbage_llm_output_falls_back():
    # Unparseable model output -> deterministic fallback, never an exception.
    r = gap.analyze_gaps(_CV, _JOB, _FakeClient("sorry, no JSON here"), "m")
    assert isinstance(r, dict) and "covered" in r and "missing" in r


def test_cap_is_respected():
    big = {"title": "X", "skills": [f"skill{i}" for i in range(30)], "description": ""}
    r = gap.analyze_gaps("nothing matches", big, None, "m", top_k=8)
    assert len(r["covered"]) <= 8 and len(r["missing"]) <= 8
