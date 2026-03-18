from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_PIPELINE_FILES = [
    "src/pipeline/stage01_pdf_to_images.py",
    "src/pipeline/stage02_block_detection.py",
    "src/pipeline/stage03_block_refiner.py",
    "src/pipeline/stage04_ocr.py",
    "src/pipeline/stage05_translation.py",
    "src/pipeline/stage06_batch_builder.py",
    "src/pipeline/stage07_llm_extraction.py",
    "src/pipeline/stage08_post_processing.py",
    "src/pipeline/stage09_resume_matching.py",
    "src/pipeline/stage10_notification.py",
    "src/pipeline/stage11_cleanup.py",
]

EXPECTED_ORCHESTRATION_FILES = [
    "scripts/run_pipeline.sh",
    "scripts/scheduler.sh",
    "scripts/jobs.conf",
]

EXPECTED_INFRA_FILES = [
    "scripts/health_check.py",
    ".github/workflows/tests.yml",
    "Makefile",
    "requirements-dev.txt",
    "data/pipeline_metrics.json",
]


def test_expected_pipeline_files_exist():
    missing = []
    for relative_path in EXPECTED_PIPELINE_FILES:
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            missing.append(relative_path)
    assert not missing, f"Missing expected pipeline file(s): {missing}"


def test_orchestration_files_exist():
    missing = []
    for relative_path in EXPECTED_ORCHESTRATION_FILES:
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            missing.append(relative_path)
    assert not missing, f"Missing orchestration file(s): {missing}"


def test_infra_files_exist():
    missing = []
    for relative_path in EXPECTED_INFRA_FILES:
        path = PROJECT_ROOT / relative_path
        if not path.exists():
            missing.append(relative_path)
    assert not missing, f"Missing infra file(s): {missing}"


def test_stage_entrypoints_are_present_and_syntax_valid():
    stage_files = sorted((PROJECT_ROOT / "src" / "pipeline").glob("stage*.py"))
    assert stage_files, "No stage files found under src/pipeline"

    missing_entrypoint = []
    syntax_errors = []

    for stage_file in stage_files:
        content = stage_file.read_text(encoding="utf-8")
        try:
            compile(content, str(stage_file), "exec")
        except SyntaxError as exc:
            syntax_errors.append(f"{stage_file}: {exc}")
            continue

        if "__main__" not in content:
            missing_entrypoint.append(str(stage_file.relative_to(PROJECT_ROOT)))

    assert not syntax_errors, f"Syntax errors detected in stage files: {syntax_errors}"
    assert not missing_entrypoint, (
        "Stage entrypoint guard not found in: "
        f"{missing_entrypoint}"
    )
