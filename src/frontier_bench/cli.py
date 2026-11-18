from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint

from frontier_bench.auth import auth_status, load_project_env
from frontier_bench.config import load_config

app = typer.Typer(help="Evaluate and chat with Kaggle frontier models.", no_args_is_help=True)
auth_app = typer.Typer(help="Kaggle account and Model Proxy authentication.")
benchmark_app = typer.Typer(help="Run reproducible local evaluations.")
models_app = typer.Typer(help="Inspect models available through Kaggle.")
app.add_typer(auth_app, name="auth")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(models_app, name="models")


def _run_cli(args: list[str]) -> int:
    executable = _find_kaggle_cli()
    if not executable:
        raise typer.BadParameter("Kaggle CLI is not installed. Install it with: python -m pip install -e '.[kaggle]'")
    command = [sys.executable, executable, *args] if executable.endswith(".py") else [executable, *args]
    proc = subprocess.run(command, text=True, capture_output=True)
    if proc.stdout:
        typer.echo(proc.stdout.rstrip())
    if proc.stderr:
        typer.echo(proc.stderr.rstrip(), err=proc.returncode != 0)
    return proc.returncode


def _find_kaggle_cli() -> str | None:
    found = shutil.which("kaggle")
    if found:
        return found
    executable_dir = Path(sys.executable).parent
    for name in ("kaggle", "kaggle.exe", "kaggle-script.py", "kaggle.bat"):
        candidate = executable_dir / name
        if candidate.is_file():
            return str(candidate)
    return None


def _find_conda() -> str | None:
    executable = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if executable:
        return executable
    for root in (Path(sys.prefix), *Path(sys.prefix).parents):
        for relative in ("bin/conda", "condabin/conda", "Scripts/conda.exe", "condabin/conda.bat"):
            candidate = root / relative
            if candidate.is_file():
                return str(candidate)
    return None


@app.command()
def doctor(python: Optional[str] = typer.Option(None, help="Python executable to inspect."),
           conda_env: Optional[str] = typer.Option(None, help="Conda environment name to inspect."),
           conda_prefix: Optional[str] = typer.Option(None, help="Conda environment prefix to inspect.")):
    """Inspect Python/dependencies and optional environment selectors."""
    if sum(x is not None for x in (python, conda_env, conda_prefix)) > 1:
        raise typer.BadParameter("Choose only one of --python, --conda-env, or --conda-prefix")
    probe = "import sys,importlib.util,json,shutil,pathlib;mods=['typer','yaml','rich','dotenv','kaggle_benchmarks','kagglehub','pytest'];found={m:bool(importlib.util.find_spec(m)) for m in mods};cli=shutil.which('kaggle') or next((str(pathlib.Path(sys.executable).parent/n) for n in ['kaggle','kaggle.exe','kaggle-script.py','kaggle.bat'] if (pathlib.Path(sys.executable).parent/n).is_file()),None);print(json.dumps({'executable':sys.executable,'version':sys.version.split()[0],'modules':found,'required_missing':[m for m in ['typer','yaml','rich','dotenv'] if not found[m]],'kaggle_cli':cli}))"
    command = [sys.executable, "-c", probe]
    if python:
        command = [python, "-c", probe]
    elif conda_prefix:
        prefix = Path(conda_prefix).expanduser()
        python_executable = next((p for p in (prefix / "bin/python", prefix / "python.exe",
                                               prefix / "Scripts/python.exe") if p.is_file()), None)
        if python_executable:
            command = [str(python_executable), "-c", probe]
        else:
            conda = _find_conda()
            if not conda:
                raise typer.BadParameter("Conda was requested but no conda executable was found")
            command = [conda, "run", "-p", conda_prefix, "python", "-c", probe]
    elif conda_env:
        conda = _find_conda()
        if not conda:
            raise typer.BadParameter("Conda was requested but no conda executable was found")
        command = [conda, "run", "-n", conda_env, "python", "-c", probe]
    proc = subprocess.run(command, text=True, capture_output=True)
    if proc.returncode:
        raise typer.Exit(code=1)
    rprint({"selected_environment": json.loads(proc.stdout), "running_python": sys.executable,
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "kaggle_cli": json.loads(proc.stdout).get("kaggle_cli"),
            "sdk_installed_in_running_environment": importlib.util.find_spec("kaggle_benchmarks") is not None})


@auth_app.command("check")
def auth_check():
    """Check account login and look for Model Proxy credentials without revealing them."""
    load_project_env()
    status = auth_status()
    rprint(status)
    if not status["kaggle_cli"] or not status["account_auth"]:
        rprint("[yellow]Run `kaggle auth login` for Kaggle account access.[/yellow]")
    if not status["proxy_credential_configured"] or not status["proxy_url_configured"]:
        rprint("[yellow]Model Proxy credentials are missing or incomplete. Refresh with `kaggle benchmarks auth -y --env-file .env`.[/yellow]")
    else:
        rprint("[yellow]Model Proxy credential presence is confirmed; expiry is not validated by this check.[/yellow]")
    if not status["kaggle_cli"] or not status["account_auth"]:
        raise typer.Exit(code=1)


@models_app.command("list")
def models_list():
    """List models exposed by the installed Kaggle Benchmarks CLI."""
    raise typer.Exit(code=_run_cli(["benchmarks", "tasks", "models"]))


@app.command()
def quota():
    """Show Kaggle Model Proxy quota using the official CLI."""
    raise typer.Exit(code=_run_cli(["benchmarks", "quota"]))


@benchmark_app.command("run")
def benchmark_run(config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
                  model: Optional[str] = typer.Option(None, help="Override configured models; comma-separated."),
                  max_samples: Optional[int] = typer.Option(None, min=1)):
    """Run a benchmark against a local or authorized Kaggle dataset."""
    from frontier_bench.evaluation import run_benchmark
    load_project_env(config.expanduser().resolve().parent)
    overrides = {"models": [x.strip() for x in model.split(",")] if model else None,
                 "max_samples": max_samples}
    cfg = load_config(config, overrides=overrides)
    if not cfg.get("models"):
        raise typer.BadParameter("No models configured; add models to YAML or pass --model MODEL_ID")
    try:
        summary = run_benchmark(cfg)
    except Exception as exc:
        typer.echo(f"Benchmark failed: {exc}", err=True)
        raise typer.Exit(code=1)
    rprint(summary)
    if summary["failed"]:
        raise typer.Exit(code=1)


@app.command()
def chat(model: str = typer.Option(..., "--model", "-m", help="Model ID from `frontier models`."),
         system_prompt: Optional[str] = typer.Option(None),
         temperature: Optional[float] = typer.Option(None, min=0, max=2)):
    """Start an interactive text chat; /reset, /model ID, /export FILE, /quit."""
    load_project_env()
    from frontier_bench.chat import ChatSession
    from frontier_bench.models import KaggleModelClient

    def make_session(model_id: str):
        generation = {"temperature": temperature} if temperature is not None else {}
        client = KaggleModelClient(model_id, system_prompt=system_prompt, generation=generation)
        return ChatSession(model_id, client, system_prompt)

    try:
        session = make_session(model)
        context = session.client.conversation()
        context.__enter__()
        typer.echo(f"Chat with {session.model_id}. Use /help for commands.")
        while True:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if line == "/quit":
                break
            if line == "/help":
                typer.echo("/reset | /model MODEL_ID | /export FILE.json|FILE.md | /quit")
                continue
            if line == "/reset":
                context.__exit__(None, None, None)
                session.reset()
                session = make_session(session.model_id)
                context = session.client.conversation()
                context.__enter__()
                typer.echo("Conversation reset.")
                continue
            if line.startswith("/model "):
                context.__exit__(None, None, None)
                session = make_session(line.split(maxsplit=1)[1])
                context = session.client.conversation()
                context.__enter__()
                typer.echo(f"Switched to {session.model_id}; started a new session.")
                continue
            if line.startswith("/export "):
                typer.echo(f"Exported {session.export(line.split(maxsplit=1)[1])}")
                continue
            response = session.ask(line)
            typer.echo(f"model> {response.text}")
            if response.usage:
                typer.echo(f"[usage: {json.dumps(response.usage)}; {response.latency_seconds:.2f}s]")
    except Exception as exc:
        from frontier_bench.models.client import _redact
        typer.echo(f"Chat failed: {_redact(str(exc))}", err=True)
        raise typer.Exit(code=1)
    finally:
        if "context" in locals():
            context.__exit__(None, None, None)


if __name__ == "__main__":
    app()
