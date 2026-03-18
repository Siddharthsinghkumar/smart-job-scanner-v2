from __future__ import annotations

import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_REQUIREMENTS = {
    "configs/gemini_config.json": {
        "google_api_keys",
        "api_key_labels",
        "serpapi_key",
        "google_search_api_key",
        "google_search_cx",
        "num_results",
    },
    "configs/newspaper_config.json": {
        "web_sources",
        "telegram_sources",
        "direct_pdf_sources",
        "settings",
    },
    "configs/pipeline_config.json": {
        "pipeline",
        "scheduler",
    },
}


def _load_json(relative_path: str) -> dict:
    path = PROJECT_ROOT / relative_path
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.mark.parametrize("relative_path", CONFIG_REQUIREMENTS.keys())
def test_config_exists_and_is_valid_json(relative_path: str):
    path = PROJECT_ROOT / relative_path
    assert path.is_file(), f"Config file not found: {relative_path}"

    payload = _load_json(relative_path)
    assert isinstance(payload, dict), f"Config root must be a JSON object: {relative_path}"


@pytest.mark.parametrize("relative_path,required_keys", CONFIG_REQUIREMENTS.items())
def test_config_has_required_top_level_keys(relative_path: str, required_keys: set[str]):
    payload = _load_json(relative_path)
    missing = sorted(required_keys - set(payload.keys()))
    assert not missing, f"Missing required key(s) in {relative_path}: {missing}"


def test_pipeline_config_has_required_sections():
    payload = _load_json("configs/pipeline_config.json")

    pipeline = payload["pipeline"]
    scheduler = payload["scheduler"]

    assert "runner" in pipeline, "Missing pipeline.runner in configs/pipeline_config.json"
    assert isinstance(pipeline.get("stages"), list), "pipeline.stages must be a list"
    assert pipeline.get("stages"), "pipeline.stages must not be empty"

    assert "config" in scheduler, "Missing scheduler.config in configs/pipeline_config.json"
    assert isinstance(scheduler.get("jobs"), list), "scheduler.jobs must be a list"
