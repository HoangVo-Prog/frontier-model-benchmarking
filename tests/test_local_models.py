"""Local model allowlisting and official SDK import bootstrap."""
from typer.testing import CliRunner
import pytest

from frontier_client import FrontierClient
from frontier_client.accounts import KaggleAccount, KaggleAccountPool, ProxyCredentials, QuotaBalance
from frontier_client.auth import local_models, validate_local_model
from frontier_client.cli import app
from frontier_client.errors import ModelRequestError


class FixedAPI:
    def create_proxy_credentials(self, token):
        return ProxyCredentials("https://proxy.invalid", "proxy-token")
    def get_quotas(self, token):
        return (QuotaBalance("daily", 0, 10, 10),)


def _pool():
    return KaggleAccountPool((KaggleAccount("test", "secret"),), api=FixedAPI())


def test_local_models_list_is_env_only(monkeypatch):
    monkeypatch.setenv("LLMS_AVAILABLE", "  acme/vision , acme/text, acme/vision ")
    monkeypatch.setenv("LLM_DEFAULT", "acme/vision")
    assert local_models() == ("acme/vision", "acme/text")
    result = CliRunner().invoke(app, ["models", "list"])
    assert result.exit_code == 0, result.output
    assert "acme/vision (default)" in result.output
    assert "acme/text" in result.output
    assert "benchmarks tasks models" not in result.output


def test_local_models_missing_fails_helpfully(monkeypatch, tmp_path):
    # A developer's real .env must not repopulate LLMS_AVAILABLE during this test.
    # FRONTIER_ENV_FILE takes precedence over cwd/parent .env discovery.
    monkeypatch.setenv("FRONTIER_ENV_FILE", str(tmp_path / "nonexistent.env"))
    monkeypatch.delenv("LLMS_AVAILABLE", raising=False)
    result = CliRunner().invoke(app, ["models", "list"])
    assert result.exit_code == 1, result.output
    assert "LLMS_AVAILABLE" in result.output
    with pytest.raises(ModelRequestError, match="LLMS_AVAILABLE"):
        validate_local_model("acme/test")


def test_local_models_list_reads_existing_env_file(tmp_path):
    # Conversely, when a local config is available, the CLI should read it.
    # patch.dict restores the entire environment, including dotenv injections.
    import os
    from unittest.mock import patch

    env_file = tmp_path / ".env"
    env_file.write_text("LLMS_AVAILABLE=acme/from-dotenv\nLLM_DEFAULT=acme/from-dotenv\n", encoding="utf-8")
    with patch.dict(os.environ, {"FRONTIER_ENV_FILE": str(env_file)}, clear=True):
        result = CliRunner().invoke(app, ["models", "list"])
        assert result.exit_code == 0, result.output
        assert "acme/from-dotenv (default)" in result.output


def test_unlisted_model_is_rejected(monkeypatch):
    monkeypatch.setenv("LLMS_AVAILABLE", "acme/allowed")
    with pytest.raises(ModelRequestError, match="not in local LLMS_AVAILABLE"):
        validate_local_model("acme/blocked")


def test_sdk_import_missing_llm_default(monkeypatch, tmp_path, sdk):
    from frontier_client import client as client_module
    monkeypatch.setenv("LLMS_AVAILABLE", "fake/model,fake/other")
    monkeypatch.delenv("LLM_DEFAULT", raising=False)
    imported = []
    def mock_import(name):
        assert name == "kaggle_benchmarks"
        assert __import__("os").environ["LLM_DEFAULT"] == "fake/model"
        imported.append(name)
        return sdk
    monkeypatch.setattr(client_module.importlib, "import_module", mock_import)
    monkeypatch.setattr(client_module.KaggleAccountPool, "from_env", lambda: _pool())
    # SDK factory accepts explicit Proxy credentials, not global MODEL_PROXY_*.
    from types import SimpleNamespace
    sdk.kaggle = SimpleNamespace(ModelProxy=lambda **kwargs: sdk.model)
    client = FrontierClient("fake/model", log_dir=tmp_path)
    assert client.prompt("hello").text == "reply:hello"
    assert imported == ["kaggle_benchmarks"]


def test_existing_allowed_sdk_default_preserved(monkeypatch, tmp_path, sdk):
    from frontier_client import client as client_module
    monkeypatch.setenv("LLMS_AVAILABLE", "fake/model,fake/other")
    monkeypatch.setenv("LLM_DEFAULT", "fake/other")
    monkeypatch.setattr(client_module.importlib, "import_module", lambda name: sdk)
    monkeypatch.setattr(client_module.KaggleAccountPool, "from_env", lambda: _pool())
    FrontierClient("fake/model", log_dir=tmp_path)
    assert __import__("os").environ["LLM_DEFAULT"] == "fake/other"


def test_disallowed_model_does_not_import_sdk(monkeypatch):
    from frontier_client import client as client_module
    monkeypatch.setenv("LLMS_AVAILABLE", "fake/allowed")
    monkeypatch.setattr(client_module.KaggleAccountPool, "from_env", lambda: _pool())
    monkeypatch.setattr(client_module.importlib, "import_module", lambda _: pytest.fail("should not import SDK"))
    with pytest.raises(ModelRequestError, match="not in local"):
        FrontierClient("fake/forbidden", logging=False)
