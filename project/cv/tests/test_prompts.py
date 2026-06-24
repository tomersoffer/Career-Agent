# -*- coding: utf-8 -*-
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from cv import prompts


def test_delta_system_lists_fields_and_contract():
    sysp = prompts.delta_system(prompts.PROFILE_LABELS,
                                prompts.interview_extra_rules("location", "מיקום", {}))
    assert '"set"' in sysp and '"add"' in sysp and '"remove"' in sysp
    assert "titles_held" in sysp                 # a schema field is listed
    assert prompts.AMENDMENT_RULE in sysp        # cross-cutting rule injected
    assert "מיקום" in sysp                        # next-field hint threaded in
    # the skills field definition (always present) forbids roles/degrees as skills,
    # so an experience-turn answer like "QA Engineer / מערכות מידע" isn't stored as a skill
    assert "QA Engineer" in sysp and "אינם כישורים" in sysp


def test_interview_rules_skills_excludes_titles():
    r = prompts.interview_extra_rules("skills", "כישורים", {})
    assert "QA Engineer" in r                    # explicitly tells it not to use the role
    assert "כישורים = כלים" in r


def test_interview_rules_location_us_only():
    r = prompts.interview_extra_rules("location", "מיקום", {}, us_only=True)
    assert "ארה" in r                            # mentions US-only
    # without us_only, no US restriction note
    assert "ארה" not in prompts.interview_extra_rules("location", "מיקום", {}, us_only=False)


def test_amendment_rule_in_tailor_preamble():
    assert prompts.AMENDMENT_RULE in prompts.TAILOR_PREAMBLE


def test_search_delta_system_lists_levers():
    sysp = prompts.delta_system(prompts.SEARCH_LABELS, prompts.search_extra_rules())
    assert "roles" in sysp and "remote" in sysp
    assert prompts.AMENDMENT_RULE in sysp


def test_amend_stage_prompt_preserves_and_excludes_structure():
    sysp = prompts.tailor_system_prompt("AMEND")
    assert "## השלב שלך: AMEND" in sysp
    assert "מבנה קורות החיים" not in sysp          # no CV-structure block -> no restructuring
    assert prompts.AMENDMENT_RULE in sysp          # preamble rule present
    assert "## השלב שלך: COMPOSE (" not in sysp    # other stages' instructions absent


def test_tailor_sections_order():
    keys = [s["key"] for s in prompts.TAILOR_SECTIONS]
    assert keys == ["personal_details", "education", "work_experience",
                    "projects", "skills", "summary"]
    for s in prompts.TAILOR_SECTIONS:
        assert s["label_he"]


def test_section_walk_prompt_no_structure():
    sysp = prompts.tailor_system_prompt("SECTION_WALK")
    assert "## השלב שלך: SECTION_WALK" in sysp
    assert "section_done" in sysp                  # the walk must request the flag
    assert "מבנה קורות החיים" not in sysp           # no CV-structure block (nothing written yet)
    assert prompts.AMENDMENT_RULE in sysp


def test_compose_lite_removed():
    assert "COMPOSE_LITE" not in prompts._STAGE_BLOCKS
