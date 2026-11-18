from pathlib import Path
import pytest

from frontier_bench.config import load_config, resolve_path


def test_precedence_and_config_relative_paths(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("models: [yaml-model]\noutput_dir: ./reports\ndataset:\n  source: local\n  path: data.jsonl\n", encoding="utf-8")
    value = load_config(cfg, environ={"FRONTIER_MODEL": "env-a,env-b", "FRONTIER_OUTPUT_DIR": "./env-runs"},
                        overrides={"models": ["cli-model"]})
    assert value["models"] == ["cli-model"]
    assert Path(value["dataset"]["path"]) == tmp_path / "data.jsonl"
    assert Path(value["output_dir"]) == (tmp_path / "env-runs").resolve()


def test_absolute_path_unchanged(tmp_path):
    absolute = (tmp_path / "data.csv").resolve()
    assert resolve_path(absolute, Path.cwd()) == absolute


def test_missing_config_is_clear(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")
