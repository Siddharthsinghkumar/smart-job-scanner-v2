from __future__ import annotations

from pathlib import Path
import os
import shlex
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_PIPELINE = PROJECT_ROOT / "scripts" / "run_pipeline.sh"
SCHEDULER = PROJECT_ROOT / "scripts" / "scheduler.sh"
JOBS_CONF = PROJECT_ROOT / "scripts" / "jobs.conf"


def _extract_run_pipeline_commands(script_text: str) -> list[str]:
    commands = []
    in_commands_block = False

    for raw_line in script_text.splitlines():
        line = raw_line.strip()
        if line.startswith("commands=("):
            in_commands_block = True
            continue
        if in_commands_block and line == ")":
            break
        if in_commands_block and line.startswith('"') and line.endswith('"'):
            commands.append(line.strip('"'))

    return commands


def _extract_script_references(command: str) -> list[str]:
    refs: list[str] = []
    try:
        tokens = shlex.split(command)
    except ValueError:
        return refs

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token in {"-c", "-lc"} and i + 1 < len(tokens):
            refs.extend(_extract_script_references(tokens[i + 1]))
            i += 2
            continue

        if token.endswith(".py") or token.endswith(".sh"):
            refs.append(token)
        i += 1

    return refs


def _resolve_script_path(path_token: str) -> Path:
    raw = Path(path_token)
    if not raw.is_absolute():
        return PROJECT_ROOT / raw

    if raw.exists():
        return raw

    # jobs.conf can store machine-specific absolute paths. If so, remap from the
    # project folder name onward so validation still works in other checkouts.
    parts = raw.parts
    if "smart-job-scanner-v2" in parts:
        idx = parts.index("smart-job-scanner-v2")
        return PROJECT_ROOT / Path(*parts[idx + 1 :])
    return raw


def test_runner_and_scheduler_shell_syntax_is_valid():
    for script in (RUN_PIPELINE, SCHEDULER):
        assert script.is_file(), f"Missing script: {script}"
        assert os.access(script, os.X_OK), f"Script is not executable: {script}"
        result = subprocess.run(  # noqa: S603
            ["bash", "-n", str(script)],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"Shell syntax check failed for {script}: {result.stderr.strip()}"
        )


def test_run_pipeline_references_existing_stage_scripts():
    content = RUN_PIPELINE.read_text(encoding="utf-8")
    commands = _extract_run_pipeline_commands(content)
    assert commands, "No commands parsed from scripts/run_pipeline.sh"

    missing = []
    for command in commands:
        tokens = shlex.split(command)
        if not tokens:
            continue
        script_rel = tokens[0]
        script_path = PROJECT_ROOT / script_rel
        if not script_path.is_file():
            missing.append(script_rel)

    assert not missing, f"run_pipeline references missing script(s): {missing}"


def test_run_pipeline_supports_dry_run_flag():
    content = RUN_PIPELINE.read_text(encoding="utf-8")
    assert "--dry-run" in content, "scripts/run_pipeline.sh must support --dry-run"
    assert "run_dry_run" in content, "scripts/run_pipeline.sh dry-run path is missing"


def test_jobs_conf_command_scripts_resolve():
    assert JOBS_CONF.is_file(), "Missing scripts/jobs.conf"

    missing = []
    malformed = []

    for raw_line in JOBS_CONF.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split("|")
        if len(parts) < 6:
            malformed.append(line)
            continue

        command = parts[4].strip()
        refs = _extract_script_references(command)
        if not refs:
            malformed.append(line)
            continue

        for ref in refs:
            resolved = _resolve_script_path(ref)
            if not resolved.exists():
                missing.append(ref)

    assert not malformed, f"Malformed or unparsable jobs.conf line(s): {malformed}"
    assert not missing, f"jobs.conf references missing script(s): {missing}"
