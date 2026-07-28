#!/usr/bin/env python3
"""Read-only system and pipeline health checks."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_ENV_VARS = [
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "TELEGRAM_API_ID",
    "TELEGRAM_API_HASH",
    "GOOGLE_API_KEYS",
]

CONFIG_REQUIREMENTS = {
    "configs/gemini_config.json": {"google_api_keys"},
    "configs/newspaper_config.json": {"settings"},
    "configs/pipeline_config.json": {"pipeline", "scheduler"},
}

CHECK_DIRS = ["logs", "run_state", "data", "configs", "scripts", "src/pipeline"]


def _status(name: str, status: str, details: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "status": status, "details": details}


def check_python() -> dict[str, Any]:
    version_info = sys.version_info
    status = "ok" if version_info >= (3, 10) else "error"
    return _status(
        "python",
        status,
        {
            "version": platform.python_version(),
            "executable": sys.executable,
            "meets_minimum_3_10": version_info >= (3, 10),
        },
    )


def check_cuda_and_torch() -> dict[str, Any]:
    details: dict[str, Any] = {}
    status = "warning"
    try:
        import torch  # type: ignore

        details["torch_version"] = torch.__version__
        details["torch_cuda_version"] = torch.version.cuda
        details["cuda_available"] = bool(torch.cuda.is_available())
        details["gpu_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        details["gpu_names"] = (
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else []
        )
        status = "ok" if torch.cuda.is_available() else "warning"
    except Exception as exc:  # noqa: BLE001
        details["error"] = repr(exc)
        status = "warning"
    return _status("cuda_torch", status, details)


def check_ollama() -> dict[str, Any]:
    binary = shutil.which("ollama")
    details: dict[str, Any] = {"binary_found": bool(binary), "binary_path": binary}
    if not binary:
        return _status("ollama", "warning", details)

    try:
        result = subprocess.run(  # noqa: S603
            [binary, "--version"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        details["version_output"] = (result.stdout or result.stderr).strip()
        details["exit_code"] = result.returncode
        return _status("ollama", "ok" if result.returncode == 0 else "warning", details)
    except Exception as exc:  # noqa: BLE001
        details["error"] = repr(exc)
        return _status("ollama", "warning", details)


def check_env_vars() -> dict[str, Any]:
    missing = [key for key in REQUIRED_ENV_VARS if not os.getenv(key)]
    return _status(
        "environment",
        "ok" if not missing else "warning",
        {"required": REQUIRED_ENV_VARS, "missing": missing},
    )


def check_configs() -> dict[str, Any]:
    details: dict[str, Any] = {}
    status = "ok"

    for rel_path, required_keys in CONFIG_REQUIREMENTS.items():
        cfg_path = PROJECT_ROOT / rel_path
        cfg_details: dict[str, Any] = {"exists": cfg_path.is_file(), "missing_keys": []}
        if not cfg_path.is_file():
            status = "error"
            details[rel_path] = cfg_details
            continue

        try:
            payload = json.loads(cfg_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("top-level JSON is not an object")
            cfg_details["valid_json"] = True
            cfg_details["missing_keys"] = sorted(required_keys - set(payload.keys()))
            if cfg_details["missing_keys"]:
                status = "error"
        except Exception as exc:  # noqa: BLE001
            cfg_details["valid_json"] = False
            cfg_details["error"] = repr(exc)
            status = "error"
        details[rel_path] = cfg_details

    return _status("configs", status, details)


def check_directory_permissions() -> dict[str, Any]:
    details: dict[str, Any] = {}
    status = "ok"
    for rel in CHECK_DIRS:
        path = PROJECT_ROOT / rel
        item = {
            "exists": path.exists(),
            "is_dir": path.is_dir(),
            "writable": os.access(path, os.W_OK) if path.exists() else False,
        }
        if not item["exists"] or not item["is_dir"] or not item["writable"]:
            status = "warning"
        details[rel] = item
    return _status("directory_permissions", status, details)


def compute_overall(results: list[dict[str, Any]]) -> str:
    statuses = [r["status"] for r in results]
    if "error" in statuses:
        return "error"
    if "warning" in statuses:
        return "warning"
    return "ok"


def main() -> int:
    checks = [
        check_python(),
        check_cuda_and_torch(),
        check_ollama(),
        check_env_vars(),
        check_configs(),
        check_directory_permissions(),
    ]
    report = {
        "project_root": str(PROJECT_ROOT),
        "overall_status": compute_overall(checks),
        "checks": checks,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report["overall_status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
