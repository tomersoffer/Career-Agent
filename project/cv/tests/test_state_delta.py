# -*- coding: utf-8 -*-
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

import matching  # noqa: F401  (kept for parity / monkeypatch target via sd.matching)
from cv import state_delta as sd

S = sd.PROFILE_SCHEMA


def test_set_replaces_scalar_and_list():
    out = sd.apply_delta({"city": "Haifa", "titles_held": ["x"]},
                         {"set": {"city": "Tel Aviv", "titles_held": ["data analyst"]}}, S)
    assert out["city"] == "Tel Aviv"
    assert out["titles_held"] == ["data analyst"]


def test_add_unions_list_dedup_order():
    out = sd.apply_delta({"titles_held": ["data analyst"]},
                         {"add": {"titles_held": ["Product Manager", "data analyst"]}}, S)
    assert out["titles_held"] == ["data analyst", "Product Manager"]


def test_remove_drops_case_insensitive():
    out = sd.apply_delta({"titles_held": ["data analyst", "product manager"]},
                         {"remove": {"titles_held": ["PRODUCT MANAGER"]}}, S)
    assert out["titles_held"] == ["data analyst"]


def test_remove_after_set_nets_to_removed():
    out = sd.apply_delta({"titles_held": []},
                         {"add": {"titles_held": ["x"]}, "remove": {"titles_held": ["x"]}}, S)
    assert out["titles_held"] == []


def test_remove_scalar_clears():
    out = sd.apply_delta({"city": "Tel Aviv"}, {"remove": {"city": []}}, S)
    assert out["city"] in ("", None)


def test_unknown_field_ignored():
    out = sd.apply_delta({}, {"set": {"evil": "x"}}, S)
    assert "evil" not in out


def test_search_schema_shape():
    s = sd.SEARCH_SCHEMA
    assert s["roles"] == "list"
    for f in ("city", "state", "seniority", "years", "remote"):
        assert s[f] == "scalar"


def test_search_delta_set_remote_and_add_role():
    out = sd.apply_delta({"roles": ["data analyst"], "remote": False},
                         {"add": {"roles": ["product manager"]}, "set": {"remote": True}},
                         sd.SEARCH_SCHEMA)
    assert out["roles"] == ["data analyst", "product manager"]
    assert out["remote"] is True


def test_coerce_delta_filters_and_types():
    raw = {"set": {"city": "Tel Aviv", "bogus": 1}, "add": {"titles_held": ["x"], "years": [2]},
           "remove": {"skills": ["py"]}, "reply": "עדכנתי", "ready": True}
    d = sd._coerce_delta(raw, sd.PROFILE_SCHEMA)
    assert d["set"] == {"city": "Tel Aviv"}          # bogus dropped
    assert d["add"] == {"titles_held": ["x"]}        # years is scalar -> dropped from add
    assert d["remove"] == {"skills": ["py"]}
    assert d["reply"] == "עדכנתי" and d["ready"] is True


def test_coerce_delta_handles_garbage():
    d = sd._coerce_delta("not a dict", sd.PROFILE_SCHEMA)
    assert d == {"set": {}, "add": {}, "remove": {}, "reply": "", "ready": False, "skip": False, "blocked": False}


def test_extract_delta_degrades_without_client():
    d = sd.extract_delta({}, [], "כל דבר", sd.PROFILE_SCHEMA, "SYS", None, "m")
    assert d == {"set": {}, "add": {}, "remove": {}, "reply": "", "ready": False, "skip": False, "blocked": False}


def test_extract_delta_uses_reply_json(monkeypatch):
    captured = {}
    def fake(client, model, context, system, max_tokens=500):
        captured["context"] = context; captured["system"] = system
        return {"set": {"city": "Tel Aviv"}, "reply": "עדכנתי מיקום", "ready": False}
    monkeypatch.setattr(sd.matching, "reply_json", fake)
    d = sd.extract_delta({"city": "Haifa"}, [{"role": "user", "content": "קודם"}],
                         "שנה לתל אביב", sd.PROFILE_SCHEMA, "SYSPROMPT", object(), "m")
    assert d["set"] == {"city": "Tel Aviv"} and d["reply"] == "עדכנתי מיקום"
    assert "שנה לתל אביב" in captured["context"] and captured["system"] == "SYSPROMPT"
