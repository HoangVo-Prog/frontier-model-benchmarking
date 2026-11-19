"""Local-first Kaggle Model Proxy CLI; no .env writes or CLI login required."""
from __future__ import annotations

import importlib.util
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Optional

import typer

from .accounts import KaggleAccountPool, configured_accounts
from .auth import MODEL_SETUP, auth_status, load_project_env, local_models
from .client import FrontierClient
from .errors import FrontierError

app = typer.Typer(help="Local chat through Kaggle Model Proxy.", no_args_is_help=True)
auth_app = typer.Typer(help="Manage Kaggle API tokens and in-memory Proxy credentials.")
models_app = typer.Typer(help="List local Model Proxy allowlist.")
accounts_app = typer.Typer(help="Show configured Kaggle account aliases (never secrets).")
app.add_typer(auth_app, name="auth")
app.add_typer(models_app, name="models")
app.add_typer(accounts_app, name="accounts")


def _pool() -> KaggleAccountPool:
    load_project_env()
    return KaggleAccountPool.from_env()


def _make_client(model: str, log_dir: Path, no_log: bool) -> FrontierClient:
    return FrontierClient(model=model, log_dir=log_dir, logging=not no_log)


@app.command()
def doctor() -> None:
    """Show dependency/configuration readiness without network or secrets."""
    load_project_env()
    try:
        state = auth_status()
    except FrontierError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps({
        "python": sys.executable,
        "version": sys.version.split()[0],
        "conda_environment": os.getenv("CONDA_DEFAULT_ENV"),
        "modules": {name: importlib.util.find_spec(name) is not None for name in
                    ("kaggle", "kagglesdk", "kaggle_benchmarks", "typer", "dotenv", "pytest")},
        "auth": state,
        "local_model_count": len(local_models()),
        "env_file_writes": False,
    }, indent=2))


@auth_app.command("check")
def check_auth(verify: bool = typer.Option(False, "--verify", help="Check live quota and account API access.")) -> None:
    """Check .env accounts; optionally test the official Kaggle API."""
    try:
        pool = _pool()
        rows = pool.quotas(refresh=True) if verify else []
        typer.echo(json.dumps({"accounts": [a.name for a in pool.accounts],
                               "model_proxy_credentials": "auto-minted in memory",
                               "live_quota": rows if verify else "not checked; use --verify"}, indent=2))
        if verify and not any(row.get("available") for row in rows):
            raise typer.Exit(1)
    except FrontierError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


@auth_app.command("refresh")
def refresh_auth() -> None:
    """Explicitly re-mint temporary Model Proxy credentials only in memory."""
    try:
        report = _pool().refresh_credentials()
        typer.echo(json.dumps(report, indent=2))
        typer.echo("Nothing written to .env; each new process automatically mints temporary credentials.")
        if any(row["status"] != "ready" for row in report):
            raise typer.Exit(1)
    except FrontierError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


@accounts_app.command("list")
def list_accounts() -> None:
    """Show account aliases in failover priority order."""
    try:
        load_project_env()
        accounts = configured_accounts()
        if not accounts:
            typer.echo("No account configured. Set KAGGLE_API_TOKEN or KAGGLE_ACCOUNTS in .env.", err=True)
            raise typer.Exit(1)
        for index, account in enumerate(accounts, 1):
            typer.echo(f"{index}. {account.name}")
    except FrontierError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


@models_app.command("list")
def list_models() -> None:
    """List LLMS_AVAILABLE, never the hosted model catalog."""
    load_project_env()
    models = local_models()
    if not models:
        typer.echo(MODEL_SETUP, err=True)
        raise typer.Exit(1)
    default = os.getenv("LLM_DEFAULT", "").strip()
    for model in models:
        typer.echo(model + (" (default)" if model == default else ""))
    typer.echo(f"{len(models)} locally configured model(s); server permissions still apply.")


@app.command()
def quota() -> None:
    """Inspect daily/monthly Model Proxy quota for each Kaggle API token."""
    try:
        report = _pool().quotas(refresh=True)
        typer.echo(json.dumps(report, indent=2))
        if not any(row.get("available") for row in report):
            raise typer.Exit(1)
    except FrontierError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


@app.command()
def ask(
    model: str = typer.Option(..., "--model", "-m"),
    prompt: str = typer.Option(..., "--prompt", "-p"),
    image: Optional[str] = typer.Option(None, "--image", help="Local PNG/JPEG path or public HTTPS URL."),
    log_dir: Path = typer.Option(Path("logs"), "--log-dir"),
    no_log: bool = typer.Option(False, "--no-log"),
) -> None:
    """Send one independent text or image prompt, with automatic quota failover."""
    try:
        client = _make_client(model, log_dir, no_log)
        response = client.prompt(prompt, image=image)
        typer.echo(response.text)
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
    """Interactive chat: /reset, /model, /image, /export, /quit."""
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
