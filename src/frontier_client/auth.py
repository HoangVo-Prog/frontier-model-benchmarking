"""Read-only .env configuration for local Kaggle Model Proxy inference."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from .accounts import configured_accounts
from .errors import ModelRequestError

MODEL_SETUP = (
    "Provide LLMS_AVAILABLE as a comma-separated local model allowlist in your private .env. "
    "Kaggle's hosted model catalog is not a guarantee of local access."
)


def load_project_env(path: str | Path | None = None) -> None:
    """Read .env only; never create or modify it, never override shell values."""
    from dotenv import load_dotenv

    explicit = path or os.getenv("FRONTIER_ENV_FILE")
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file():
            load_dotenv(candidate, override=False)
        return
    start = Path.cwd().resolve()
    for directory in (start, *start.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return


def local_models() -> tuple[str, ...]:
    raw = os.getenv("LLMS_AVAILABLE", "")
    return tuple(dict.fromkeys(x.strip() for x in raw.split(",") if x.strip()))


def validate_local_model(model: str) -> tuple[str, ...]:
    models = local_models()
    if not models:
        raise ModelRequestError("Local LLMS_AVAILABLE is missing or empty. " + MODEL_SETUP)
    if model not in models:
        raise ModelRequestError(
            f"Model {model!r} is not in local LLMS_AVAILABLE. "
            "Run `frontier models list` to inspect the configured local allowlist."
        )
    return models


def auth_status() -> dict[str, Any]:
    """Display only account aliases and configuration readiness."""
    accounts = configured_accounts()
    return {
        "kaggle_cli_installed": shutil.which("kaggle") is not None,
        "accounts_configured": len(accounts),
        "account_names": [account.name for account in accounts],
        "local_models_configured": bool(local_models()),
        "authentication": "Kaggle API token; in-memory Model Proxy credentials",
    }
