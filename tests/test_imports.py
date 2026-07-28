from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODULES_TO_IMPORT = [
    "src.pipeline.stage01_pdf_to_images",
    "src.pipeline.stage02_block_detection",
    "src.pipeline.stage03_block_refiner",
    "src.pipeline.stage04_ocr",
    "src.pipeline.stage05_translation",
    "src.pipeline.stage06_batch_builder",
    "src.pipeline.stage07_llm_extraction",
    "src.pipeline.stage08_post_processing",
    "src.pipeline.stage09_dynamic_resumes",
    "src.pipeline.stage09_local_filter",
    "src.pipeline.stage09_llm_filter",
    "src.pipeline.stage09_shortlist",
    "src.pipeline.stage09_resume_matching",
    "src.pipeline.stage10_notification",
    "src.pipeline.stage11_cleanup",
    "src.downloader.newspaper_downloader",
    "src.llm.gemini_client",
    "src.vision.block_detector",
    "src.ocr.easyocr_engine",
    "src.utils.config_loader",
]


@pytest.fixture(autouse=True)
def _set_placeholder_env(monkeypatch):
    test_home = Path("/tmp/sjs_pytest_home")
    test_xdg_data_home = Path("/tmp/sjs_pytest_xdg_data")
    test_xdg_cache_home = Path("/tmp/sjs_pytest_xdg_cache")
    test_home.mkdir(parents=True, exist_ok=True)
    test_xdg_data_home.mkdir(parents=True, exist_ok=True)
    test_xdg_cache_home.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "placeholder_bot_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456789")
    monkeypatch.setenv("TELEGRAM_API_ID", "12345")
    monkeypatch.setenv("TELEGRAM_API_HASH", "placeholder_api_hash")
    monkeypatch.setenv("GOOGLE_API_KEYS", "placeholder_key_1,placeholder_key_2")
    monkeypatch.setenv("HOME", str(test_home))
    monkeypatch.setenv("XDG_DATA_HOME", str(test_xdg_data_home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(test_xdg_cache_home))


@pytest.mark.parametrize("module_name", MODULES_TO_IMPORT)
def test_module_imports_successfully(module_name: str):
    importlib.invalidate_caches()
    try:
        importlib.import_module(module_name)
    except BaseException as exc:  # noqa: BLE001
        pytest.fail(f"Import failed for {module_name}: {exc!r}")
