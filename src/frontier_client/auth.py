"""Kaggle account and short-lived Model Proxy credentials are separate."""
from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import AuthenticationError

REFRESH = "kaggle benchmarks auth -y --env-file .env"


def load_project_env(path: str | Path | None = None) -> None:
    """Load existing .env without overwriting shell values or creating files."""
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


def _expiry() -> datetime | None:
    raw = os.getenv("MODEL_PROXY_EXPIRY_TIME", "").strip()
    if not raw:
        return None
    try:
        return datetime.fromtimestamp(float(raw), tz=timezone.utc) if raw.replace(".", "", 1).isdigit() else datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None


def auth_status() -> dict[str, Any]:
    """Return booleans only: never print the API key, URL or token metadata."""
    kaggle_dir = Path.home() / ".kaggle"
    expiry = _expiry()
    account = bool(os.getenv("KAGGLE_API_TOKEN") or (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")) or (kaggle_dir / "kaggle.json").is_file() or (kaggle_dir / "access_token").is_file())
    return {
        "kaggle_cli": shutil.which("kaggle") is not None,
        "account_credentials_present": account,
        "proxy_credentials_present": bool(os.getenv("MODEL_PROXY_API_KEY") and os.getenv("MODEL_PROXY_URL")),
        "proxy_expiry_known": expiry is not None,
        "proxy_expired": (expiry <= datetime.now(timezone.utc)) if expiry else None,
    }


def require_proxy_credentials() -> None:
    state = auth_status()
    if not state["proxy_credentials_present"]:
        raise AuthenticationError(f"Kaggle Model Proxy credentials are missing. Set KAGGLE_API_TOKEN, then run `{REFRESH}`.")
    if state["proxy_expired"] is True:
        raise AuthenticationError(f"Kaggle Model Proxy credentials are expired. Refresh with `{REFRESH}`.")
