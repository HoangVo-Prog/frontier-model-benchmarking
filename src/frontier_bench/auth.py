"""Credential status checks that avoid printing secrets."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROXY_KEYS = ("MODEL_PROXY_API_KEY", "KAGGLE_MODEL_PROXY_TOKEN", "KAGGLE_BENCHMARKS_TOKEN", "KAGGLE_BENCHMARKS_AUTH_TOKEN")


def load_project_env(root: str | Path | None = None) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    explicit = os.environ.get("FRONTIER_ENV_FILE")
    if explicit:
        env_file = Path(explicit).expanduser()
        if env_file.is_file():
            load_dotenv(env_file, override=False)
        return
    start = Path(root or Path.cwd()).expanduser().resolve()
    for directory in (start, *start.parents):
        env_file = directory / ".env"
        if env_file.is_file():
            load_dotenv(env_file, override=False)
            return


def auth_status() -> dict[str, Any]:
    cli = shutil.which("kaggle")
    if not cli:
        executable_dir = Path(sys.executable).parent
        cli = next((str(executable_dir / name) for name in ("kaggle", "kaggle.exe", "kaggle-script.py", "kaggle.bat")
                    if (executable_dir / name).is_file()), None)
    status: dict[str, Any] = {
        "kaggle_cli": bool(cli),
        "account_auth": False,
        "account_token_env_present": bool(os.environ.get("KAGGLE_API_TOKEN")),
        "account_username_key_present": bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")),
        "proxy_credential_configured": any(os.environ.get(key) for key in PROXY_KEYS),
        "proxy_url_configured": bool(os.environ.get("MODEL_PROXY_URL")),
        "account_credential_files": [],
    }
    home = Path.home() / ".kaggle"
    for filename in ("kaggle.json", "access_token"):
        path = home / filename
        if path.is_file():
            status["account_credential_files"].append(filename)
    if cli:
        try:
            proc = subprocess.run([cli, "auth", "print-access-token"], capture_output=True,
                                  text=True, timeout=15, check=False)
            status["account_auth"] = proc.returncode == 0 and bool(proc.stdout.strip())
        except (OSError, subprocess.TimeoutExpired):
            pass
    status["proxy_hint"] = "Proxy credential presence is reported separately; expiry is not validated here. Refresh with kaggle benchmarks auth -y --env-file .env"
    return status
