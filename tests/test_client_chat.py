from contextlib import contextmanager
from types import SimpleNamespace
import importlib.util

import pytest

from frontier_bench.chat import ChatSession
from frontier_bench.models import KaggleModelClient, ModelResponse
from frontier_bench.models.client import ModelRequestError, _generation_kwargs


class FakeChat:
    def __init__(self):
        self.usage = SimpleNamespace(input_tokens=2, output_tokens=3)
    def __enter__(self): return self
    def __exit__(self, *_): return False


class FakeModel:
    def __init__(self): self.calls = []
    def prompt(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return "reply:" + prompt


class FakeSDK:
    def __init__(self):
        self.model = FakeModel()
        self.llms = {"fake/model": self.model}
        self.chats = SimpleNamespace(new=lambda name, system_instructions=None: FakeChat())


def test_client_mock_prompt_and_usage():
    sdk = FakeSDK()
    client = KaggleModelClient("fake/model", generation={"temperature": 0}, sdk=sdk)
    response = client.prompt("hi")
    assert response.text == "reply:hi"
    assert response.usage == {"input_tokens": 2, "output_tokens": 3}


def test_chat_session_reset_export_and_isolation(tmp_path):
    class Client:
        def prompt(self, text, **kwargs): return ModelResponse(text.upper(), .1, {})
    a = ChatSession("a", Client())
    b = ChatSession("b", Client())
    a.ask("hello")
    assert len(a.history) == 2 and b.history == []
    exported = a.export(tmp_path / "chat.json")
    assert '"model_id": "a"' in exported.read_text(encoding="utf-8")
    a.reset()
    assert a.history == []


def test_sdk_conversation_keeps_context_between_turns():
    sdk = FakeSDK()
    client = KaggleModelClient("fake/model", sdk=sdk)
    with client.conversation():
        client.prompt("one")
        client.prompt("two")
    assert [call[0] for call in sdk.model.calls] == ["one", "two"]


def test_auth_error_has_actionable_refresh_guidance():
    class Expired(FakeModel):
        def prompt(self, prompt, **kwargs): raise RuntimeError("401 unauthorized")
    sdk = FakeSDK()
    sdk.llms["fake/model"] = Expired()
    client = KaggleModelClient("fake/model", sdk=sdk)
    try:
        client.prompt("hello")
    except ModelRequestError as exc:
        assert "kaggle benchmarks auth" in str(exc)
    else:
        raise AssertionError("expected an actionable authentication error")


def test_generation_options_follow_sdk_signature():
    assert _generation_kwargs({"temperature": 0.2, "reasoning_effort": "low"}) == {
        "temperature": 0.2, "reasoning": "low"}


@pytest.mark.skipif(importlib.util.find_spec("kaggle_benchmarks") is None, reason="Kaggle SDK optional")
def test_installed_sdk_without_proxy_credentials_fails_actionably(monkeypatch):
    monkeypatch.delenv("MODEL_PROXY_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_PROXY_URL", raising=False)
    with pytest.raises(ModelRequestError, match="kaggle benchmarks auth"):
        KaggleModelClient("unconfigured-model")
