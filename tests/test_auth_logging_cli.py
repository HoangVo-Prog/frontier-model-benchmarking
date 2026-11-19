"""CLI/auth/logging regressions with no live Kaggle requests."""
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from frontier_client import AuthenticationError, FrontierClient
from frontier_client.auth import auth_status
from frontier_client.cli import app
from frontier_client.logging import JSONLLogger


def test_account_env_status_no_secrets(monkeypatch):
    monkeypatch.setenv("KAGGLE_ACCOUNTS", "lab,personal")
    monkeypatch.setenv("KAGGLE_API_TOKEN_LAB", "secret-lab")
    monkeypatch.setenv("KAGGLE_API_TOKEN_PERSONAL", "secret-personal")
    status = auth_status()
    assert status["accounts_configured"] == 2
    assert status["account_names"] == ["lab", "personal"]
    assert "secret" not in json.dumps(status)
    assert "expiry" not in json.dumps(status)


def test_missing_named_token(monkeypatch):
    from frontier_client.accounts import configured_accounts
    monkeypatch.setenv("KAGGLE_ACCOUNTS", "lab,personal")
    monkeypatch.setenv("KAGGLE_API_TOKEN_LAB", "first")
    monkeypatch.delenv("KAGGLE_API_TOKEN_PERSONAL", raising=False)
    with pytest.raises(AuthenticationError, match="PERSONAL"):
        configured_accounts()


def test_jsonl_concurrent_safe(tmp_path):
    logger = JSONLLogger(tmp_path)
    with ThreadPoolExecutor(max_workers=12) as executor:
        list(executor.map(lambda i: logger.append({"index": i, "text": "hello" * 400}), range(75)))
    lines = (tmp_path / "requests.jsonl").read_text().splitlines()
    assert len(lines) == 75
    assert {json.loads(line)["index"] for line in lines} == set(range(75))
    assert all(json.loads(line)["text"] == "hello" * 400 for line in lines)


def test_cli_ask_mocks_client(monkeypatch, tmp_path):
    from frontier_client import cli
    class MockClient:
        def __init__(self, **kwargs):
            assert kwargs["model"] == "fake/model"
        def prompt(self, prompt, image=None):
            assert prompt == "hello"
            return SimpleNamespace(text="MOCK ANSWER")
    monkeypatch.setattr(cli, "FrontierClient", MockClient)
    result = CliRunner().invoke(app, ["ask", "--model", "fake/model", "--prompt", "hello", "--log-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "MOCK ANSWER" in result.output


def test_cli_chat_commands(monkeypatch, sdk, png, tmp_path):
    from frontier_client import cli
    sdk.llms["other/model"] = sdk.model
    monkeypatch.setattr(cli, "FrontierClient", lambda **kwargs: FrontierClient(sdk=sdk, **kwargs))
    exported = tmp_path / "chat.json"
    commands = ("hello\n/reset\n" + f'/image "{png}" describe photo\n'
                + f"/export {exported}\n/model other/model\nnew question\n/quit\n")
    result = CliRunner().invoke(app, ["chat", "--model", "fake/model", "--no-log"], input=commands)
    assert result.exit_code == 0, result.output
    assert "reply:hello" in result.output
    assert "Conversation reset" in result.output
    assert "reply:describe photo" in result.output
    assert "Switched to other/model" in result.output
    assert "reply:new question" in result.output
    data = json.loads(exported.read_text())
    assert len(data["messages"]) == 2
    assert data["messages"][0]["image"] == str(png)


def test_cli_auth_refresh_in_memory_only(monkeypatch, tmp_path):
    from frontier_client import cli
    path = tmp_path / ".env"
    path.write_text("KAGGLE_API_TOKEN=SECRET\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_pool", lambda: SimpleNamespace(refresh_credentials=lambda: [{"account": "lab", "status": "ready"}]))
    result = CliRunner().invoke(app, ["auth", "refresh"])
    assert result.exit_code == 0, result.output
    assert "Nothing written to .env" in result.output
    assert "SECRET" not in result.output
    assert path.read_text() == "KAGGLE_API_TOKEN=SECRET\n"


def test_cli_quota_uses_api_not_subprocess(monkeypatch):
    from frontier_client import cli
    rows = [{"account": "lab", "available": True, "balances": [
        {"period": "daily", "remaining_usd": 4.0}, {"period": "monthly", "remaining_usd": 80.0}]}]
    monkeypatch.setattr(cli, "_pool", lambda: SimpleNamespace(quotas=lambda refresh: rows))
    result = CliRunner().invoke(app, ["quota"])
    assert result.exit_code == 0, result.output
    assert '"account": "lab"' in result.output
    assert '"period": "monthly"' in result.output


def test_cli_accounts_and_auth_doctor_no_secrets(monkeypatch):
    monkeypatch.setenv("KAGGLE_ACCOUNTS", "alpha,beta")
    monkeypatch.setenv("KAGGLE_API_TOKEN_ALPHA", "SUPER_SECRET")
    monkeypatch.setenv("KAGGLE_API_TOKEN_BETA", "TOP_SECRET")
    for args in (("accounts", "list"), ("auth", "check"), ("doctor",)):
        result = CliRunner().invoke(app, list(args))
        assert result.exit_code == 0, result.output
        assert "SUPER_SECRET" not in result.output
        assert "TOP_SECRET" not in result.output
        assert "alpha" in result.output
        assert "beta" in result.output
