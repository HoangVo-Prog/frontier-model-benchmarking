"""Local model allowlisting and SDK bootstrap regression tests."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typer.testing import CliRunner
import pytest

from frontier_client import FrontierClient
from frontier_client.auth import local_models, validate_local_model
from frontier_client.cli import app
from frontier_client.errors import ModelRequestError


def test_local_models_list_is_env_only(monkeypatch):
    monkeypatch.setenv("LLMS_AVAILABLE", "  acme/vision , acme/text, acme/vision ")
    monkeypatch.setenv("LLM_DEFAULT", "acme/vision")
    assert local_models() == ("acme/vision", "acme/text")
    result = CliRunner().invoke(app, ["models", "list"])
    assert result.exit_code == 0, result.output
    assert "acme/vision (default)" in result.output
    assert "acme/text" in result.output
    assert "benchmarks tasks models" not in result.output


def test_local_models_missing_fails_helpfully(monkeypatch):
    monkeypatch.delenv("LLMS_AVAILABLE", raising=False)
    result = CliRunner().invoke(app, ["models", "list"])
    assert result.exit_code == 1
    assert "kaggle benchmarks init" in result.output
    with pytest.raises(ModelRequestError, match="LLMS_AVAILABLE"):
        validate_local_model("acme/test")


def test_unlisted_model_is_rejected(monkeypatch):
    monkeypatch.setenv("LLMS_AVAILABLE", "acme/allowed")
    with pytest.raises(ModelRequestError, match="not in local LLMS_AVAILABLE"):
        validate_local_model("acme/blocked")


def test_sdk_import_missing_llm_default(monkeypatch, tmp_path, sdk):
    """SDK import checks LLM_DEFAULT exists; no actual Kaggle credentials needed."""
    from frontier_client import client as client_module
    monkeypatch.setenv("MODEL_PROXY_URL", "https://proxy.invalid")
    monkeypatch.setenv("MODEL_PROXY_API_KEY", "not-a-real-credential")
    monkeypatch.setenv("MODEL_PROXY_EXPIRY_TIME", (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat())
    monkeypatch.setenv("LLMS_AVAILABLE", "fake/model, fake/other")
    monkeypatch.delenv("LLM_DEFAULT", raising=False)
    imported = []

    def mock_import(name):
        assert name == "kaggle_benchmarks"
        assert __import__("os").environ["LLM_DEFAULT"] == "fake/model"
        imported.append(name)
        return sdk

    monkeypatch.setattr(client_module.importlib, "import_module", mock_import)
    client = FrontierClient("fake/model", log_dir=tmp_path)
    assert client.prompt("hello").text == "reply:hello"
    assert imported == ["kaggle_benchmarks"]


def test_existing_allowed_sdk_default_preserved(monkeypatch, tmp_path, sdk):
    from frontier_client import client as client_module
    monkeypatch.setenv("MODEL_PROXY_URL", "https://proxy.invalid")
    monkeypatch.setenv("MODEL_PROXY_API_KEY", "not-a-real-credential")
    monkeypatch.setenv("LLMS_AVAILABLE", "fake/model, fake/other")
    monkeypatch.setenv("LLM_DEFAULT", "fake/other")
    monkeypatch.setattr(client_module.importlib, "import_module", lambda name: sdk)
    FrontierClient("fake/model", log_dir=tmp_path)
    assert __import__("os").environ["LLM_DEFAULT"] == "fake/other"


def test_disallowed_model_does_not_import_sdk(monkeypatch):
    from frontier_client import client as client_module
    monkeypatch.setenv("MODEL_PROXY_URL", "https://proxy.invalid")
    monkeypatch.setenv("MODEL_PROXY_API_KEY", "not-a-real-credential")
    monkeypatch.setenv("LLMS_AVAILABLE", "fake/allowed")
    monkeypatch.setattr(client_module.importlib, "import_module", lambda _: pytest.fail("should not import SDK"))
    with pytest.raises(ModelRequestError, match="not in local"):
        FrontierClient("fake/forbidden", logging=False)
