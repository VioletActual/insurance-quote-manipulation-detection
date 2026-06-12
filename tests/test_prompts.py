from src.qm.prompts import PROMPT_VERSIONS


def test_all_versions_present():
    assert "v1_basic" in PROMPT_VERSIONS
    assert "v2_grounded_concise" in PROMPT_VERSIONS
    assert "v3_structured" in PROMPT_VERSIONS


def test_each_version_has_required_keys():
    for version, config in PROMPT_VERSIONS.items():
        assert "system" in config, f"{version} missing system prompt"
        assert "created" in config, f"{version} missing created date"
        assert "change_note" in config, f"{version} missing change_note"


def test_system_prompts_are_nonempty():
    for version, config in PROMPT_VERSIONS.items():
        assert len(config["system"].strip()) > 0, f"{version} has empty system prompt"


def test_v3_enforces_format_in_prompt():
    prompt = PROMPT_VERSIONS["v3_structured"]["system"]
    assert "REASONS:" in prompt
    assert "ACTION:" in prompt


def test_v2_mentions_word_limit():
    prompt = PROMPT_VERSIONS["v2_grounded_concise"]["system"]
    assert "90" in prompt