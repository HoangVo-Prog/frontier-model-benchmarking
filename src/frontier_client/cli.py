"""Minimal local CLI for Kaggle Model Proxy inference."""
from __future__ import annotations

import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from .auth import MODEL_SETUP, REFRESH, auth_status, load_project_env, local_models
from .client import FrontierClient
from .errors import FrontierError

app = typer.Typer(help="Local chat through Kaggle Model Proxy.", no_args_is_help=True)
auth_app = typer.Typer(help="Check or refresh Model Proxy credentials.")
models_app = typer.Typer(help="Inspect the local Model Proxy allowlist.")
app.add_typer(auth_app, name="auth")
app.add_typer(models_app, name="models")


def _official_cli(args: list[str]) -> None:
    kaggle = shutil.which("kaggle")
    if not kaggle:
        typer.echo("Kaggle CLI is missing. Install with `python -m pip install -e '.[dev]'`.", err=True)
        raise typer.Exit(1)
    result = subprocess.run([kaggle, *args], check=False)
    if result.returncode:
        raise typer.Exit(result.returncode)


def _make_client(model: str, log_dir: Path, no_log: bool) -> FrontierClient:
    return FrontierClient(model=model, log_dir=log_dir, logging=not no_log)


@app.command()
def doctor() -> None:
    """Inspect Python, dependencies, credentials and local model configuration."""
    load_project_env()
    typer.echo(json.dumps({
        "python": sys.executable,
        "version": sys.version.split()[0],
        "conda_environment": os.getenv("CONDA_DEFAULT_ENV"),
        "modules": {name: importlib.util.find_spec(name) is not None for name in
                    ("kaggle", "kaggle_benchmarks", "typer", "dotenv", "pytest")},
        "auth": auth_status(),
        "local_model_count": len(local_models()),
    }, indent=2))


@auth_app.command("check")
def check_auth() -> None:
    """Inspect credentials without printing keys or authorization headers."""
    load_project_env()
    state = auth_status()
    typer.echo(json.dumps(state, indent=2))
    if not state["proxy_credentials_present"] or state["proxy_expired"] is True:
        typer.echo(f"Refresh Model Proxy credentials with: {REFRESH}", err=True)
        raise typer.Exit(1)


@auth_app.command("refresh")
def refresh_auth() -> None:
    """Refresh local credentials using the official Kaggle CLI."""
    _official_cli(["benchmarks", "auth", "-y", "--env-file", ".env"])
    typer.echo("Model Proxy credentials refreshed in .env. Restart running Python processes if necessary.")


@models_app.command("list")
def list_models() -> None:
    """List only LLMS_AVAILABLE, not Kaggle's broader hosted catalog."""
    load_project_env()
    models = local_models()
    if not models:
        typer.echo("Local LLMS_AVAILABLE is not configured. " + MODEL_SETUP, err=True)
        raise typer.Exit(1)
    default = os.getenv("LLM_DEFAULT", "").strip()
    for model in models:
        typer.echo(model + (" (default)" if model == default else ""))
    typer.echo(f"{len(models)} model(s) configured locally; server permissions still apply.")


@app.command()
def quota() -> None:
    """Inspect Model Proxy inference spend (not Kaggle GPU hours)."""
    _official_cli(["benchmarks", "quota"])


@app.command()
def ask(
    model: str = typer.Option(..., "--model", "-m"),
    prompt: str = typer.Option(..., "--prompt", "-p"),
    image: Optional[str] = typer.Option(None, "--image", help="Local PNG/JPEG path or public HTTPS URL."),
    log_dir: Path = typer.Option(Path("logs"), "--log-dir"),
    no_log: bool = typer.Option(False, "--no-log"),
) -> None:
    """Send one independent text or image prompt."""
    try:
        client = _make_client(model, log_dir, no_log)
        typer.echo(client.prompt(prompt, image=image).text)
    except (FrontierError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


@app.command()
def chat(
    model: str = typer.Option(..., "--model", "-m"),
    system_prompt: Optional[str] = typer.Option(None, "--system-prompt"),
    log_dir: Path = typer.Option(Path("logs"), "--log-dir"),
    no_log: bool = typer.Option(False, "--no-log"),
) -> None:
    """Interactive chat: /reset, /model ID, /image PATH PROMPT, /export FILE.json, /quit."""
    try:
        client = _make_client(model, log_dir, no_log)
        current = client.chat(system_prompt=system_prompt)
        current.__enter__()
    except (FrontierError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    try:
        typer.echo("/reset | /model MODEL_ID | /image PATH PROMPT | /export FILE.json | /quit")
        while True:
            try:
                line = input("you> ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not line:
                continue
            if line == "/quit":
                break
            if line == "/reset":
                current.reset()
                typer.echo("Conversation reset.")
                continue
            if line.startswith("/export "):
                try:
                    typer.echo(f"Exported {current.export(line[len('/export '):].strip())}")
                except ValueError as exc:
                    typer.echo(str(exc), err=True)
                continue
            if line.startswith("/model "):
                new_model = line[len("/model "):].strip()
                try:
                    next_client = _make_client(new_model, log_dir, no_log)
                    next_chat = next_client.chat(system_prompt=system_prompt)
                    next_chat.__enter__()
                    current.__exit__(None, None, None)
                    current = next_chat
                    typer.echo(f"Switched to {new_model}; new session started.")
                except (FrontierError, ValueError) as exc:
                    typer.echo(str(exc), err=True)
                continue
            image = None
            if line.startswith("/image "):
                try:
                    argv = shlex.split(line)
                    if len(argv) < 3:
                        raise ValueError("Usage: /image path/to/photo.png Describe this")
                    image, line = argv[1], " ".join(argv[2:])
                except ValueError as exc:
                    typer.echo(str(exc), err=True)
                    continue
            try:
                typer.echo("model> " + current.send(line, image=image).text)
            except (FrontierError, ValueError) as exc:
                typer.echo(str(exc), err=True)
    finally:
        current.__exit__(None, None, None)


if __name__ == "__main__":
    app()
