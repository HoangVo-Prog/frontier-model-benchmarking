from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier_client import FrontierClient


class FakeMessage:
    def __init__(self, role, content, usage=None):
        self.sender = SimpleNamespace(role=role)
        self.content = content
        self.usage = usage


class FakeChat:
    def __init__(self, system_instructions=None):
        self.history = []
        self.messages = self.history
        if system_instructions:
            self.history.append(FakeMessage("system", system_instructions))


class FakeModel:
    support_temperature = True

    def __init__(self, sdk):
        self.sdk = sdk
        self.calls = []
        self.failure = None

    def prompt(self, prompt, **kwargs):
        chat = self.sdk.active_chat
        chat.history.append(FakeMessage("user", prompt))
        self.calls.append({"prompt": prompt, "kwargs": kwargs, "thread": chat})
        if self.failure:
            raise self.failure
        answer = "reply:" + prompt
        chat.history.append(FakeMessage("assistant", answer, SimpleNamespace(
            input_tokens=2, output_tokens=3,
            input_tokens_cost_nanodollars=100,
            output_tokens_cost_nanodollars=200,
            total_cost_nanodollars=300,
            total_backend_latency_ms=15,
        )))
        if "schema" in kwargs:
            return {"reply": answer}
        return answer


class FakeSDK:
    def __init__(self):
        self.active_chat = None
        self.model = FakeModel(self)
        self.llms = {"fake/model": self.model}
        self.content_types = SimpleNamespace(images=SimpleNamespace(
            from_path=lambda path: {"local": path},
            from_url=lambda url: {"url": url},
        ))
        self.chats = SimpleNamespace(new=self.new)
        self.contexts = SimpleNamespace(enter=self.enter)

    @contextmanager
    def enter(self, chat):
        old = self.active_chat
        self.active_chat = chat
        try:
            yield chat
        finally:
            self.active_chat = old

    @contextmanager
    def new(self, name, system_instructions=None, orphan=False):
        with self.enter(FakeChat(system_instructions)) as chat:
            yield chat


@pytest.fixture
def sdk():
    return FakeSDK()


@pytest.fixture
def client(tmp_path, sdk):
    return FrontierClient("fake/model", sdk=sdk, log_dir=tmp_path, log_prompts=True)


@pytest.fixture
def png(tmp_path):
    target = tmp_path / "photo.png"
    target.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 15)
    return target
