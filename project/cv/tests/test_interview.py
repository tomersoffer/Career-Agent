# -*- coding: utf-8 -*-
import pytest

import matching
from cv import interview
from cv import state_delta as sd


def _patch_delta(monkeypatch, delta):
    # interview_turn calls state_delta.extract_delta -> patch it to return a canned delta
    monkeypatch.setattr(sd, "extract_delta", lambda *a, **k: {
        "set": delta.get("set", {}), "add": delta.get("add", {}),
        "remove": delta.get("remove", {}), "reply": delta.get("reply", "אוקיי"),
        "ready": delta.get("ready", False), "skip": delta.get("skip", False)})


def test_interview_turn_adds_role_and_asks_next(monkeypatch):
    _patch_delta(monkeypatch, {"add": {"titles_held": ["data analyst"]}, "reply": "מעולה"})
    out = interview.interview_turn({}, [], "אני מחפש data analyst", object(), "m")
    assert out["profile"]["titles_held"] == ["data analyst"]
    assert out["done"] is False
    assert out["next_field"] == "experience"     # role filled -> next missing essential


def test_interview_turn_changes_location(monkeypatch):
    _patch_delta(monkeypatch, {"set": {"city": "Tel Aviv"}, "reply": "עדכנתי"})
    out = interview.interview_turn({"titles_held": ["data analyst"], "city": "Haifa"},
                                   [], "שנה מיקום לתל אביב", object(), "m")
    assert out["profile"]["city"] == "Tel Aviv"


def test_interview_turn_removes_role(monkeypatch):
    _patch_delta(monkeypatch, {"remove": {"titles_held": ["product manager"]}, "reply": "הורדתי"})
    out = interview.interview_turn(
        {"titles_held": ["data analyst", "product manager"]}, [], "תוריד product", object(), "m")
    assert out["profile"]["titles_held"] == ["data analyst"]


def test_interview_turn_requires_full_profile(monkeypatch):
    # role-only is NOT enough to finish — the agent must still gather the rest.
    _patch_delta(monkeypatch, {"ready": True, "reply": "בוא נחפש"})
    out = interview.interview_turn({"titles_held": ["data analyst"]}, [], "זהו", object(), "m")
    assert out["done"] is False
    assert out["next_field"] == "experience"         # keeps walking, doesn't search early

    # full profile (role, experience, location, skills) -> done + synthesized cv_text.
    full = {"titles_held": ["data analyst"], "years": 2, "seniority": "entry",
            "city": "Tel Aviv", "skills": ["sql"]}
    _patch_delta(monkeypatch, {"reply": "מה הרמה שלך?"})   # model still asking — must be overridden
    out2 = interview.interview_turn(full, [], "זהו", object(), "m")
    assert out2["done"] is True
    assert out2["cv_text"] and "data analyst" in out2["cv_text"]
    assert "?" not in out2["reply"][:-1] and "נצא לחפש" in out2["reply"]   # completion, not a question


def test_interview_turn_skip_marks_field_and_can_finish(monkeypatch):
    # role/experience/location present, skills missing -> user has none -> skip -> done.
    prof = {"titles_held": ["data analyst"], "years": 2, "seniority": "entry", "city": "Tel Aviv"}
    _patch_delta(monkeypatch, {"skip": True, "reply": "בסדר"})
    out = interview.interview_turn(prof, [], "אין לי כישורים מיוחדים", object(), "m", skipped=[])
    assert "skills" in out["skipped"]
    assert out["done"] is True


def test_interview_turn_degrades_without_client():
    out = interview.interview_turn({}, [], "data analyst", None, "m")
    assert out["done"] is False and out["next_field"] == "role"
    assert isinstance(out["reply"], str) and out["reply"]


def test_steps_shape():
    keys = [s["key"] for s in interview.STEPS]
    assert keys == ["role", "experience", "location", "skills"]
    for s in interview.STEPS:
        assert s["label_he"] and s["question_he"]


def test_parse_role(monkeypatch):
    monkeypatch.setattr(matching, "reply_json",
                        lambda *a, **k: {"titles_held": ["data analyst", "bi analyst"]})
    out = interview.parse_answer("role", "I want a data analyst job", object(), "m")
    assert out["titles_held"] == ["data analyst", "bi analyst"]


def test_parse_experience(monkeypatch):
    monkeypatch.setattr(matching, "reply_json",
                        lambda *a, **k: {"years": 3, "seniority": "associate"})
    out = interview.parse_answer("experience", "3 years, mid", object(), "m")
    assert out["years"] == 3
    assert out["seniority"] == "associate"


def test_parse_experience_rejects_bool_years_and_bad_seniority(monkeypatch):
    # bool is not a valid int for years; an out-of-vocab seniority falls back to ""
    monkeypatch.setattr(matching, "reply_json",
                        lambda *a, **k: {"years": True, "seniority": "wizard"})
    out = interview.parse_answer("experience", "lots", object(), "m")
    assert out["years"] is None
    assert out["seniority"] == ""


def test_parse_step_captures_out_of_step_added_role(monkeypatch):
    # The user "answers" the experience question by adding a role instead — the role must
    # NOT be lost; it surfaces in add_titles (deduped + lowercased) for additive merge.
    monkeypatch.setattr(matching, "reply_json",
                        lambda *a, **k: {"years": None, "seniority": "",
                                         "add_titles": ["Product Manager", "product manager"]})
    out = interview.parse_answer("experience", "תוסיף למשרות גם PRODUCT", object(), "m")
    assert out["years"] is None and out["seniority"] == ""
    assert out["add_titles"] == ["product manager"]


def test_parse_step_without_added_role_has_no_add_titles_key(monkeypatch):
    # When no extra role is volunteered, add_titles is absent (no empty-key noise).
    monkeypatch.setattr(matching, "reply_json",
                        lambda *a, **k: {"years": 2, "seniority": "entry"})
    out = interview.parse_answer("experience", "שנתיים", object(), "m")
    assert "add_titles" not in out


def test_parse_location(monkeypatch):
    monkeypatch.setattr(matching, "reply_json",
                        lambda *a, **k: {"city": "Tel Aviv", "state": "ny"})
    out = interview.parse_answer("location", "תל אביב", object(), "m")
    assert out["city"] == "Tel Aviv"
    assert out["state"] == "NY"          # upper-cased by _coerce


def test_parse_skills(monkeypatch):
    monkeypatch.setattr(matching, "reply_json",
                        lambda *a, **k: {"skills": ["Python", "SQL", "python"]})
    out = interview.parse_answer("skills", "python, sql", object(), "m")
    assert out["skills"] == ["python", "sql"]   # lowercased + deduped


def test_empty_answer_skips(monkeypatch):
    # an empty/blank answer never calls the LLM and yields empty fields
    monkeypatch.setattr(matching, "reply_json",
                        lambda *a, **k: pytest.fail("LLM should not be called"))
    assert interview.parse_answer("role", "   ", object(), "m") == {"titles_held": []}


def test_degrade_stores_raw_text_when_llm_returns_empty(monkeypatch):
    # reply_json returns {} when the client is None or output isn't JSON -> fallback
    monkeypatch.setattr(matching, "reply_json", lambda *a, **k: {})
    role = interview.parse_answer("role", "data analyst", None, "m")
    assert role["titles_held"] == ["data analyst"]
    skills = interview.parse_answer("skills", "python, sql", None, "m")
    assert skills["skills"] == ["python", "sql"]
    loc = interview.parse_answer("location", "Haifa", None, "m")
    assert loc["city"] == "Haifa"


def test_captured_label():
    p = {"titles_held": ["data analyst", "bi analyst"], "years": 3,
         "seniority": "associate", "city": "Tel Aviv", "state": "",
         "skills": ["python", "sql", "excel"]}
    assert interview.captured_label(0, p) == "data analyst"
    assert "3" in interview.captured_label(1, p)              # years surfaced
    assert interview.captured_label(2, p) == "Tel Aviv"
    assert interview.captured_label(3, p) == "python, sql, excel"


def test_captured_label_empty():
    assert interview.captured_label(0, {"titles_held": []}) == "—"


def test_finalize_has_full_key_set():
    p = interview.finalize({"titles_held": ["data analyst"], "skills": ["sql"]})
    for k in ("name", "location", "skills", "titles_held", "seniority", "years",
              "city", "state", "domains", "work_experience", "projects"):
        assert k in p
    assert p["titles_held"] == ["data analyst"]
    assert p["years"] is None
    assert p["work_experience"] == []


def test_synthesize_cv_text_includes_present_sections():
    p = interview.finalize({"titles_held": ["data analyst"], "years": 3,
                            "seniority": "associate", "city": "Tel Aviv",
                            "skills": ["python", "sql"]})
    txt = interview.synthesize_cv_text(p)
    assert "data analyst" in txt
    assert "3" in txt
    assert "Tel Aviv" in txt
    assert "python" in txt


def test_synthesize_cv_text_omits_empty_sections():
    p = interview.finalize({"titles_held": ["nurse"]})
    txt = interview.synthesize_cv_text(p)
    assert "nurse" in txt
    assert "Location:" not in txt
    assert "Skills:" not in txt


def test_seed_from_query_role_and_location():
    profile, seeded, captured = interview.seed_from_query(
        "data analyst", ["bi analyst"], "New York", "NY", None, "")
    assert profile["titles_held"] == ["data analyst", "bi analyst"]
    assert profile["city"] == "New York" and profile["state"] == "NY"
    assert set(seeded) == {"role", "location"}        # experience not provided
    assert captured[0] == "data analyst"              # step 0 = role
    assert captured[2] == "New York"                  # step 2 = location


def test_seed_from_query_experience_only():
    profile, seeded, captured = interview.seed_from_query("", [], "", "", 3, "associate")
    assert seeded == ["experience"]
    assert "3" in captured[1]


def test_seed_from_query_empty():
    profile, seeded, captured = interview.seed_from_query("", [], "", "", None, "")
    assert seeded == []
    assert captured == {}


def test_first_and_next_unseeded_step():
    # role+location seeded -> first unseeded is experience (index 1)
    assert interview.first_unseeded_step(["role", "location"]) == 1
    # from experience, next unseeded skipping location is skills (index 3)
    assert interview.next_unseeded_step(1, ["role", "location"]) == 3
    # nothing seeded -> first is 0
    assert interview.first_unseeded_step([]) == 0
    # all seeded -> returns len(STEPS)
    allkeys = [s["key"] for s in interview.STEPS]
    assert interview.first_unseeded_step(allkeys) == len(interview.STEPS)
