import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from frontier_client import AuthenticationError, FrontierClient
from frontier_client.auth import auth_status, require_proxy_credentials
from frontier_client.cli import app
from frontier_client.logging import JSONLLogger


def test_missing_expired_and_valid_proxy_auth(monkeypatch):
    monkeypatch.delenv("MODEL_PROXY_URL", raising=False)
    monkeypatch.delenv("MODEL_PROXY_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_PROXY_EXPIRY_TIME", raising=False)
    with pytest.raises(AuthenticationError, match="benchmarks auth"):
        require_proxy_credentials()
    monkeypatch.setenv("MODEL_PROXY_URL", "https://test.invalid/openapi")
    monkeypatch.setenv("MODEL_PROXY_API_KEY", "hidden-secret")
    monkeypatch.setenv("MODEL_PROXY_EXPIRY_TIME", (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat())
    assert auth_status()["proxy_expired"] is True
    with pytest.raises(AuthenticationError, match="expired"):
        require_proxy_credentials()
    monkeypatch.setenv("MODEL_PROXY_EXPIRY_TIME", (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat())
    require_proxy_credentials()
    assert "hidden-secret" not in json.dumps(auth_status())


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


def test_cli_auth_refresh_delegates(monkeypatch):
    from frontier_client import cli
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/kaggle")
    calls = []
    monkeypatch.setattr(cli.subprocess, "run", lambda args, **kwargs: (calls.append(args) or SimpleNamespace(returncode=0)))
    result = CliRunner().invoke(app, ["auth", "refresh"])
    assert result.exit_code == 0, result.output
    assert calls == [["/usr/bin/kaggle", "benchmarks", "auth", "-y", "--env-file", ".env"]]


def test_cli_quota_delegates(monkeypatch):
    from frontier_client import cli
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/kaggle")
    calls = []
    monkeypatch.setattr(cli.subprocess, "run", lambda args, **kwargs: (calls.append(args) or SimpleNamespace(returncode=0)))
    result = CliRunner().invoke(app, ["quota"])
    assert result.exit_code == 0, result.output
    assert calls == [["/usr/bin/kaggle", "benchmarks", "quota"]]


def test_cli_doctor_no_secrets(monkeypatch):
    monkeypatch.setenv("MODEL_PROXY_API_KEY", "TOP_SECRET")
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "TOP_SECRET" not in result.output
    assert "local_model_count" in result.output
