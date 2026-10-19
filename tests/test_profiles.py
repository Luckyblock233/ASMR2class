from pathlib import Path

import pytest

from src.profiles.loader import get_profile, load_profiles

LANGUAGES_DIR = Path(__file__).resolve().parent.parent / "config" / "languages"


def test_scan_profiles():
    profiles = load_profiles(LANGUAGES_DIR)
    assert {"ja", "en", "zh"} <= set(profiles)
    ja = profiles["ja"]
    assert ja.language == "ja"
    assert ja.source_lang == "ja"
    assert ja.target_lang == "zh"
    assert ja.language_hint == "ja"
    assert ja.prompt_path("translate") == "ja/translate.txt"
    assert ja.prompt_path("grammar") == "ja/grammar.txt"
    sections = ja.raw["grammar_card"]["sections"]
    assert [section["key"] for section in sections] == [
        "structure",
        "grammar_points",
        "vocabulary",
    ]


def test_get_profile_missing():
    with pytest.raises(KeyError, match="不存在"):
        get_profile(LANGUAGES_DIR, "fr")


def test_load_profiles_empty_dir(tmp_path):
    assert load_profiles(tmp_path) == {}
