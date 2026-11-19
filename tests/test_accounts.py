"""Multi-account routing via first-party Kaggle API mocks; never live inference."""
from __future__ import annotations

import json
import sys
import types
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from frontier_client import FrontierClient
from frontier_client.accounts import (
    KaggleAccount, KaggleAccountPool, OfficialKaggleAPI, ProxyCredentials,
    QuotaBalance, configured_accounts,
)
from frontier_client.errors import AuthenticationError, ModelRequestError
from conftest import FakeModel


class FakeAccountAPI:
    def __init__(self, remaining=None, expiry=None):
        self.remaining = remaining or {"first": (10, 100), "second": (10, 100)}
        self.expiry = expiry
        self.issued = []
        self.quota_calls = []
        self.unavailable = set()

    def create_proxy_credentials(self, token):
        self.issued.append(token)
        return ProxyCredentials("https://proxy.invalid", f"proxy-{token}-{self.issued.count(token)}", self.expiry)

    def get_quotas(self, token):
        self.quota_calls.append(token)
        if token in self.unavailable:
            raise RuntimeError(f"network error secret={token}")
        daily, monthly = self.remaining[token]
        return (QuotaBalance("daily", 10 - daily, 10, daily),
                QuotaBalance("monthly", 100 - monthly, 100, monthly))


def make_client(sdk, tmp_path, *, remaining=None, expiry=None, ttl=30):
    api = FakeAccountAPI(remaining=remaining, expiry=expiry)
    pool = KaggleAccountPool((KaggleAccount("primary", "first"),
                              KaggleAccount("secondary", "second")),
                             api=api, min_remaining_usd=0.05, quota_cache_seconds=ttl)
    models = {}
    def factory(*, model, api, api_key, base_url):
        if api_key not in models:
            models[api_key] = FakeModel(sdk)
        return models[api_key]
    sdk.kaggle = SimpleNamespace(ModelProxy=factory)
    client = FrontierClient(model="fake/model", sdk=sdk, account_pool=pool, log_dir=tmp_path)
    return client, pool, api, models


def _logs(path):
    return [json.loads(row) for row in (path / "requests.jsonl").read_text().splitlines()]


def test_configure_multiple_or_single_tokens(monkeypatch):
    monkeypatch.delenv("KAGGLE_ACCOUNTS", raising=False)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "one-token")
    assert [(a.name, a.api_token) for a in configured_accounts()] == [("default", "one-token")]
    monkeypatch.setenv("KAGGLE_API_TOKENS", "token-a,token-b")
    assert [a.name for a in configured_accounts()] == ["account_1", "account_2"]
    monkeypatch.setenv("KAGGLE_ACCOUNTS", "personal,lab")
    monkeypatch.setenv("KAGGLE_API_TOKEN_PERSONAL", "one")
    monkeypatch.setenv("KAGGLE_API_TOKEN_LAB", "two")
    assert [(a.name, a.api_token) for a in configured_accounts()] == [("personal", "one"), ("lab", "two")]
    assert "one" not in repr(configured_accounts()[0])
    monkeypatch.delenv("KAGGLE_ACCOUNTS")
    monkeypatch.setenv("KAGGLE_API_TOKENS", "token-a,,token-b")
    with pytest.raises(AuthenticationError, match="empty token"):
        configured_accounts()
    monkeypatch.setenv("KAGGLE_ACCOUNTS", "lab,LAB")
    with pytest.raises(AuthenticationError, match="Duplicate"):
        configured_accounts()


def test_preflight_skips_key_below_any_quota_limit(sdk, tmp_path):
    c, pool, api, models = make_client(sdk, tmp_path, remaining={"first": (0.01, 5), "second": (8, 20)})
    result = c.prompt("hello")
    assert result.metadata["account"] == "secondary"
    assert result.text == "reply:hello"
    assert api.issued == ["second"]
    assert _logs(tmp_path)[0]["accounts_attempted"] == ["secondary"]
    assert "first" not in models


def test_quota_error_rotates_and_logs_only_aliases(sdk, tmp_path):
    c, pool, api, models = make_client(sdk, tmp_path)
    # First request mints proxy for first account; it is exhausted on request.
    primary_model = FakeModel(sdk)
    primary_model.failure = RuntimeError("429 quota exceeded secret=first")
    sdk.kaggle.ModelProxy = lambda *, model, api, api_key, base_url: (
        primary_model if api_key.startswith("proxy-first-") else FakeModel(sdk))
    response = c.prompt("Explain attention")
    assert response.text == "reply:Explain attention"
    assert response.metadata["account"] == "secondary"
    log = _logs(tmp_path)[0]
    assert log["status"] == "success"
    assert log["accounts_attempted"] == ["primary", "secondary"]
    assert log["account"] == "secondary"
    assert "proxy-first" not in json.dumps(log)
    assert "secret=first" not in json.dumps(log)
    # Sticky rotation: subsequent requests go to secondary rather than retry primary.
    c.prompt("Next")
    assert _logs(tmp_path)[-1]["accounts_attempted"] == ["secondary"]


def test_chat_history_survives_quota_rotation(sdk, tmp_path):
    c, pool, api, models = make_client(sdk, tmp_path)
    with c.chat(system_prompt="Explain briefly") as chat:
        first = chat.send("first")
        assert first.metadata["account"] == "primary"
        primary_model = models["proxy-first-1"]
        primary_model.failure = RuntimeError("quota exhausted")
        second = chat.send("second")
        assert second.metadata["account"] == "secondary"
        assert len(chat.history) == 4
        # Failed turn rolled back from SDK thread before secondary retry.
        assert [m.content for m in chat._thread.messages] == [
            "Explain briefly", "first", "reply:first", "second", "reply:second"]
        assert c.prompt("independent").metadata["account"] == "secondary"


def test_generic_rate_limit_does_not_rotate(sdk, tmp_path):
    c, pool, api, models = make_client(sdk, tmp_path)
    primary_model = FakeModel(sdk)
    primary_model.failure = RuntimeError("HTTP 429 rate limit: too many requests")
    sdk.kaggle.ModelProxy = lambda **kw: primary_model if kw["api_key"].startswith("proxy-first") else FakeModel(sdk)
    with pytest.raises(ModelRequestError, match="rate limited"):
        c.prompt("hello")
    assert api.issued == ["first"]
    assert _logs(tmp_path)[0]["status"] == "failed"
    assert _logs(tmp_path)[0]["accounts_attempted"] == ["primary"]


def test_expired_proxy_reissued_automatically(sdk, tmp_path):
    c, pool, api, models = make_client(sdk, tmp_path)
    def factory(**kwargs):
        new = FakeModel(sdk)
        if kwargs["api_key"] == "proxy-first-1":
            new.failure = RuntimeError("401 Unauthorized: token=secret-key")
        return new
    sdk.kaggle.ModelProxy = factory
    response = c.prompt("hello")
    assert response.metadata["account"] == "primary"
    assert api.issued == ["first", "first"]
    assert _logs(tmp_path)[0]["accounts_attempted"] == ["primary", "primary"]
    assert "secret-key" not in json.dumps(_logs(tmp_path)[0])


def test_near_expiry_refreshes_without_env_file(sdk, tmp_path):
    c, pool, api, models = make_client(sdk, tmp_path, expiry=datetime.now(timezone.utc) + timedelta(seconds=25))
    c.prompt("hello")
    c.prompt("again")
    assert api.issued == ["first", "first"]  # refresh before expiry buffer
    assert not (tmp_path / ".env").exists()


def test_all_accounts_exhausted_fails_closed(sdk, tmp_path):
    c, pool, api, models = make_client(sdk, tmp_path, remaining={"first": (0, 10), "second": (9, 0)})
    with pytest.raises(ModelRequestError, match="No configured Kaggle account"):
        c.prompt("hello")
    assert api.issued == []
    assert _logs(tmp_path)[0]["status"] == "failed"


def test_quota_api_failure_safely_skips_invalid_account(sdk, tmp_path):
    c, pool, api, models = make_client(sdk, tmp_path)
    api.unavailable.add("first")
    assert c.prompt("hello").metadata["account"] == "secondary"
    assert api.issued == ["second"]
    assert "secret" not in json.dumps(pool.quotas(refresh=True))


def test_quota_cache_reuses_balance_until_expiry(sdk, tmp_path):
    c, pool, api, models = make_client(sdk, tmp_path, ttl=1000)
    c.prompt("hello")
    c.prompt("again")
    assert api.quota_calls == ["first"]
    assert api.issued == ["first"]


def test_real_provider_uses_official_kaggle_sdk_methods(monkeypatch):
    """Smoke-test the official adapter without importing Kaggle or networking."""
    calls = []
    class ProxyApi:
        def create_default_model_proxy_token(self, request):
            calls.append("proxy")
            return SimpleNamespace(token="temporary", base_uri="https://proxy.invalid", expiry_time=None)
        def get_model_proxy_quotas(self, request):
            calls.append("quota")
            return SimpleNamespace(quota_balances=[SimpleNamespace(
                quota_used=1.5, total_quota_allowed=4.0,
                refill_period=SimpleNamespace(name="DAILY"), refill_time=None)])
    class ContextClient:
        def __enter__(self):
            return SimpleNamespace(models=SimpleNamespace(model_proxy_api_client=ProxyApi()))
        def __exit__(self, *args):
            pass
    class KaggleApi:
        @staticmethod
        def build_kaggle_client_with_params(*, args, api_token):
            assert args == [] and api_token == "KGAT_dummy"
            return ContextClient()
    mod_names = ["kaggle", "kaggle.api", "kaggle.api.kaggle_api_extended",
                 "kagglesdk", "kagglesdk.models", "kagglesdk.models.types",
                 "kagglesdk.models.types.model_proxy_api_service"]
    for name in mod_names:
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["kaggle.api.kaggle_api_extended"].KaggleApi = KaggleApi
    svc = sys.modules["kagglesdk.models.types.model_proxy_api_service"]
    svc.ApiCreateDefaultModelProxyTokenRequest = type("ApiCreateDefaultModelProxyTokenRequest", (), {})
    svc.ApiGetModelProxyQuotasRequest = type("ApiGetModelProxyQuotasRequest", (), {})
    official = OfficialKaggleAPI()
    assert official.create_proxy_credentials("KGAT_dummy").token == "temporary"
    balances = official.get_quotas("KGAT_dummy")
    assert balances[0].remaining_usd == 2.5
    assert calls == ["proxy", "quota"]
