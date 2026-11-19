"""Authentication and the locally permitted Kaggle Model Proxy model set."""
from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import AuthenticationError, ModelRequestError

REFRESH = "kaggle benchmarks auth -y --env-file .env"
# `auth` fetches credentials, but `init` also obtains LLMS_AVAILABLE.
MODEL_SETUP = "Run `kaggle benchmarks init -y` in a temporary directory, then copy LLMS_AVAILABLE into your private .env."


def load_project_env(path: str | Path | None = None) -> None:
    """Read a private .env, never overriding exported shell variables."""
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
    """Return the current local allowlist; never consult the hosted catalog."""
    raw = os.getenv("LLMS_AVAILABLE", "")
    return tuple(dict.fromkeys(x.strip() for x in raw.split(",") if x.strip()))


def validate_local_model(model: str) -> tuple[str, ...]:
    models = local_models()
    if not models:
        raise ModelRequestError("Local LLMS_AVAILABLE is missing or empty. " + MODEL_SETUP)
    if model not in models:
        raise ModelRequestError(
            f"Model {model!r} is not in local LLMS_AVAILABLE. "
            "Run `frontier models list` to inspect your credential's local allowlist."
        )
    return models


def _expiry() -> datetime | None:
    raw = os.getenv("MODEL_PROXY_EXPIRY_TIME", "").strip()
    if not raw:
        return None
    try:
        parsed = (datetime.fromtimestamp(float(raw), tz=timezone.utc)
                  if raw.replace(".", "", 1).isdigit()
                  else datetime.fromisoformat(raw.replace("Z", "+00:00")))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else None
    except (ValueError, OverflowError, OSError):
        return None


def auth_status() -> dict[str, Any]:
    """Return status only: never expose token, API URL or headers."""
    kaggle_dir = Path.home() / ".kaggle"
    expiry = _expiry()
    account = bool(os.getenv("KAGGLE_API_TOKEN") or
                   (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")) or
                   (kaggle_dir / "kaggle.json").is_file() or
                   (kaggle_dir / "access_token").is_file())
    return {
        "kaggle_cli": shutil.which("kaggle") is not None,
        "account_credentials_present": account,
        "proxy_credentials_present": bool(os.getenv("MODEL_PROXY_API_KEY") and os.getenv("MODEL_PROXY_URL")),
        "proxy_expiry_known": expiry is not None,
        "proxy_expired": (expiry <= datetime.now(timezone.utc)) if expiry else None,
        "local_models_configured": bool(local_models()),
    }


def require_proxy_credentials() -> None:
    state = auth_status()
    if not state["proxy_credentials_present"]:
        raise AuthenticationError(
            f"Kaggle Model Proxy credentials are missing. Configure your Kaggle account and run `{REFRESH}`."
        )
    if state["proxy_expired"] is True:
        raise AuthenticationError(f"Kaggle Model Proxy credentials expired. Refresh with `{REFRESH}`.")
