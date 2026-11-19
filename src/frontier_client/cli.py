"""Command-line interface for Kaggle-backed frontier inference."""
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

from .auth import auth_status, load_project_env
from .client import FrontierClient
from .errors import FrontierError

app = typer.Typer(help="Call frontier models through Kaggle Model Proxy.", no_args_is_help=True)
auth_app = typer.Typer(help="Check Kaggle and Model Proxy authentication.")
models_app = typer.Typer(help="Inspect available Kaggle models.")
app.add_typer(auth_app, name="auth")
app.add_typer(models_app, name="models")


def _official_cli(args: list[str]) -> None:
    kaggle = shutil.which("kaggle")
    if not kaggle:
        typer.echo("Kaggle CLI is missing; install with `python -m pip install -e '.[dev]'`.", err=True)
        raise typer.Exit(1)
    result = subprocess.run([kaggle, *args], check=False)
    if result.returncode:
        raise typer.Exit(result.returncode)


def _make_client(model: str, temperature: float | None, reasoning: str | None,
                 log_dir: Path, no_log: bool, log_prompts: bool, api: str) -> FrontierClient:
    return FrontierClient(model=model, temperature=temperature, reasoning=reasoning,
                          log_dir=log_dir, logging=not no_log,
                          log_prompts=log_prompts, api=api)


@app.command()
def doctor() -> None:
    """Show environment/dependency readiness without exposing secrets."""
    load_project_env()
    typer.echo(json.dumps({
        "python": sys.executable,
        "version": sys.version.split()[0],
        "conda_environment": os.getenv("CONDA_DEFAULT_ENV"),
        "modules": {name: importlib.util.find_spec(name) is not None for name in ("kaggle_benchmarks", "typer", "dotenv", "pytest")},
        "auth": auth_status(),
    }, indent=2))


@auth_app.command("check")
def check_auth() -> None:
    """Inspect credential presence and expiry; never show actual tokens."""
    load_project_env()
    status = auth_status()
    typer.echo(json.dumps(status, indent=2))
    if not status["proxy_credentials_present"] or status["proxy_expired"] is True:
        typer.echo("Refresh Model Proxy credentials with: kaggle benchmarks auth -y --env-file .env", err=True)
        raise typer.Exit(1)


@models_app.command("list")
def list_models() -> None:
    """List the current Kaggle catalog, not guaranteed accessible locally."""
    _official_cli(["benchmarks", "tasks", "models"])


@app.command()
def quota() -> None:
    """Show Kaggle Model Proxy inference spend quota (not GPU hours)."""
    _official_cli(["benchmarks", "quota"])


@app.command()
def ask(
    model: str = typer.Option(..., "--model", "-m"),
    prompt: str = typer.Option(..., "--prompt", "-p"),
    image: Optional[Path] = typer.Option(None, "--image", help="Local PNG/JPEG path. Public URLs can be passed via --image-url."),
    image_url: Optional[str] = typer.Option(None, "--image-url"),
    temperature: Optional[float] = typer.Option(None),
    reasoning: Optional[str] = typer.Option(None),
    api: str = typer.Option("openai", help="Kaggle SDK backend: openai or genai (both Kaggle Model Proxy)"),
    log_dir: Path = typer.Option(Path("logs")),
    no_log: bool = typer.Option(False, "--no-log"),
    log_prompts: bool = typer.Option(False, "--log-prompts"),
) -> None:
    """Make one text-only or text+image request; always print the answer."""
    if image is not None and image_url is not None:
        raise typer.BadParameter("Choose only one of --image or --image-url")
    try:
        client = _make_client(model, temperature, reasoning, log_dir, no_log, log_prompts, api)
        response = client.prompt(prompt, image=image_url if image_url is not None else image)
        typer.echo(response.text)
    except (FrontierError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


@app.command()
def chat(
    model: str = typer.Option(..., "--model", "-m"),
    system_prompt: Optional[str] = typer.Option(None),
    temperature: Optional[float] = typer.Option(None),
    reasoning: Optional[str] = typer.Option(None),
    api: str = typer.Option("openai"),
    log_dir: Path = typer.Option(Path("logs")),
    no_log: bool = typer.Option(False, "--no-log"),
    log_prompts: bool = typer.Option(False, "--log-prompts"),
) -> None:
    """Interactive chat: /reset, /model ID, /export FILE.json, /image FILE PROMPT, /quit."""
    try:
        client = _make_client(model, temperature, reasoning, log_dir, no_log, log_prompts, api)
        current = client.chat(system_prompt=system_prompt)
        current.__enter__()
    except (FrontierError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    try:
        typer.echo(f"Chat with {model}. /reset, /model ID, /export FILE.json, /image FILE PROMPT, /quit")
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
                    typer.echo(f"Exported {current.export(line.split(maxsplit=1)[1])}")
                except ValueError as exc:
                    typer.echo(str(exc), err=True)
                continue
            if line.startswith("/model "):
                new_model = line.split(maxsplit=1)[1].strip()
                try:
                    next_client = _make_client(new_model, temperature, reasoning, log_dir, no_log, log_prompts, api)
                    next_chat = next_client.chat(system_prompt=system_prompt)
                    next_chat.__enter__()
                    current.__exit__(None, None, None)
                    current = next_chat
                    typer.echo(f"Switched to {new_model}; started a new session.")
                except (FrontierError, ValueError) as exc:
                    typer.echo(str(exc), err=True)
                continue
            image = None
            if line.startswith("/image "):
                try:
                    argv = shlex.split(line)
                    if len(argv) < 3:
                        raise ValueError("Usage: /image path/to/image.png Describe the image")
                    image = argv[1]
                    line = " ".join(argv[2:])
                except ValueError as exc:
                    typer.echo(str(exc), err=True)
                    continue
            try:
                response = current.send(line, image=image)
                typer.echo(f"model> {response.text}")
            except (FrontierError, ValueError) as exc:
                typer.echo(str(exc), err=True)
    finally:
        current.__exit__(None, None, None)


if __name__ == "__main__":
    app()
