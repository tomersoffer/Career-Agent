# -*- coding: utf-8 -*-
from cv import profile as cv_profile


def test_profile_has_name_key_when_degraded():
    # claude_client=None -> graceful degrade -> empty profile, but every key must exist.
    p = cv_profile.build_profile("some cv text", None, "model")
    assert "name" in p
    assert p["name"] == ""
