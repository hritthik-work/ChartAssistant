from __future__ import annotations

from backend.core.config import ROOT, Settings


def test_system_prompt_history_contains_five_distinct_versions():
    prompt_dir = ROOT / "backend" / "prompts"
    prompts = {
        version: (prompt_dir / f"system_{version}.txt").read_text(encoding="utf-8")
        for version in ("v1", "v2", "v3", "v4", "v5")
    }

    assert len(set(prompts.values())) == 5
    assert "question-answering assistant" in prompts["v1"]
    assert "zero-shot workflow" in prompts["v2"]
    assert "Few-shot examples" in prompts["v3"]
    assert "Instruction priority" in prompts["v4"]
    assert "Citation contract" in prompts["v5"]
    assert "Final validation" in prompts["v5"]


def test_v5_is_the_default_runtime_prompt(settings: Settings):
    assert settings.system_prompt_version == "v5"
    selected = settings.prompt_dir / f"system_{settings.system_prompt_version}.txt"
    assert selected.name == "system_v5.txt"
    assert selected.is_file()
