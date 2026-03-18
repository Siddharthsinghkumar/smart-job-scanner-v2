from __future__ import annotations

from pathlib import Path
import os
import re
import warnings

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_ENV_VARS = [
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "TELEGRAM_API_ID",
    "TELEGRAM_API_HASH",
    "GOOGLE_API_KEYS",
]


def test_required_environment_variables_exist_or_warn():
    dotenv_path = PROJECT_ROOT / ".env"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)

    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing:
        warnings.warn(
            "Missing environment variable(s): "
            f"{', '.join(missing)}. "
            "This is a warning-only validation and does not fail the suite.",
            UserWarning,
            stacklevel=1,
        )


def test_env_example_declares_required_environment_variables():
    env_example = PROJECT_ROOT / ".env.example"
    assert env_example.is_file(), "Missing .env.example"

    content = env_example.read_text(encoding="utf-8")
    missing = [
        name
        for name in REQUIRED_ENV_VARS
        if re.search(rf"^{re.escape(name)}=", content, re.MULTILINE) is None
    ]
    assert not missing, f".env.example is missing required variable(s): {missing}"
