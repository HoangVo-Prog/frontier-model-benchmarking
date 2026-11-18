"""Configuration loading and deterministic path resolution."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "models": [], "prompt_template": "{question}", "max_samples": None,
    "metrics": ["exact_match"], "output_dir": "./runs", "generation": {},
    "dataset": {"source": "local", "path": "./data", "columns": {}},
}


def load_config(path: str | Path | None = None, *, overrides: dict[str, Any] | None = None,
                environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Merge defaults < YAML < FRONTIER_* environment < explicit CLI overrides."""
    env = os.environ if environ is None else environ
    config_path = Path(path).expanduser().resolve() if path else None
    config = _merge({}, DEFAULTS)
    if config_path:
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("Configuration YAML must contain a mapping")
        config = _merge(config, raw)
        config["_config_dir"] = str(config_path.parent)
    else:
        config["_config_dir"] = str(Path.cwd())
    if env.get("FRONTIER_MODEL"):
        config["models"] = [m.strip() for m in env["FRONTIER_MODEL"].split(",") if m.strip()]
    if env.get("FRONTIER_DATASET_PATH"):
        config["dataset"]["path"] = env["FRONTIER_DATASET_PATH"]
        config["dataset"]["source"] = "local"
    if env.get("FRONTIER_OUTPUT_DIR"):
        config["output_dir"] = env["FRONTIER_OUTPUT_DIR"]
    if overrides:
        config = _merge(config, {k: v for k, v in overrides.items() if v is not None})
    base = Path(config["_config_dir"])
    for key in ("output_dir",):
        config[key] = str(resolve_path(config[key], base))
    dataset = config.get("dataset", {})
    for key in ("path",):
        if dataset.get(key):
            dataset[key] = str(resolve_path(dataset[key], base))
    return config


def resolve_path(value: str | Path, base: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(base) / path).resolve()


def _merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result
