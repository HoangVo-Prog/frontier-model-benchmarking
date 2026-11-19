"""In-memory Kaggle account and Model Proxy credential management.

Only Kaggle's first-party API mints Model Proxy credentials or reports spend.
Long-lived account tokens come from environment variables; no credential files
are created or rewritten by this module.
"""
from __future__ import annotations

import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from .errors import AuthenticationError, ModelRequestError


@dataclass(frozen=True, repr=False)
class KaggleAccount:
    name: str
    api_token: str = field(repr=False)

    def __repr__(self) -> str:
        return f"KaggleAccount(name={self.name!r}, api_token=<hidden>)"


@dataclass(frozen=True, repr=False)
class ProxyCredentials:
    base_url: str
    token: str = field(repr=False)
    expires_at: datetime | None = None


@dataclass(frozen=True)
class QuotaBalance:
    period: str
    used_usd: float
    total_usd: float
    remaining_usd: float
    refill_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "used_usd": self.used_usd,
            "total_usd": self.total_usd,
            "remaining_usd": self.remaining_usd,
            "refill_at": self.refill_at,
        }


class KaggleAPI(Protocol):
    def create_proxy_credentials(self, token: str) -> ProxyCredentials: ...
    def get_quotas(self, token: str) -> tuple[QuotaBalance, ...]: ...


def _date(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(value, tz=timezone.utc)
    else:
        dt = datetime.fromisoformat(str(value).rstrip("Z") + "+00:00" if str(value).endswith("Z") else str(value))
    # Kaggle's expiry_time may be naive UTC as in their CLI implementation.
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


class OfficialKaggleAPI:
    """Call the same public SDK endpoints as ``kaggle benchmarks auth/quota``.

    Unlike those CLI commands, these methods never write an env file and
    never mutate global os.environ to switch accounts.
    """

    @staticmethod
    def _client(token: str) -> Any:
        from kaggle.api.kaggle_api_extended import KaggleApi
        return KaggleApi.build_kaggle_client_with_params(args=[], api_token=token)

    def create_proxy_credentials(self, token: str) -> ProxyCredentials:
        from kagglesdk.models.types.model_proxy_api_service import ApiCreateDefaultModelProxyTokenRequest
        with self._client(token) as client:
            result = client.models.model_proxy_api_client.create_default_model_proxy_token(
                ApiCreateDefaultModelProxyTokenRequest()
            )
        if not getattr(result, "token", None) or not getattr(result, "base_uri", None):
            raise AuthenticationError("Kaggle did not return usable Model Proxy credentials.")
        return ProxyCredentials(str(result.base_uri), str(result.token), _date(getattr(result, "expiry_time", None)))

    def get_quotas(self, token: str) -> tuple[QuotaBalance, ...]:
        from kagglesdk.models.types.model_proxy_api_service import ApiGetModelProxyQuotasRequest
        with self._client(token) as client:
            result = client.models.model_proxy_api_client.get_model_proxy_quotas(ApiGetModelProxyQuotasRequest())
        output: list[QuotaBalance] = []
        for balance in getattr(result, "quota_balances", ()) or ():
            used = float(balance.quota_used)
            total = float(balance.total_quota_allowed)
            if not math.isfinite(used) or not math.isfinite(total) or total < 0 or used < 0:
                raise ModelRequestError("Kaggle returned invalid quota values.")
            period = getattr(balance.refill_period, "name", str(balance.refill_period))
            refill = getattr(balance, "refill_time", None)
            output.append(QuotaBalance(str(period).lower(), used, total, max(0.0, total - used),
                                       _date(refill).isoformat() if refill is not None else None))
        if not output:
            raise ModelRequestError("Kaggle did not provide quota balances; cannot safely choose an account.")
        return tuple(output)


def configured_accounts() -> tuple[KaggleAccount, ...]:
    """Parse named, compact CSV, or single Kaggle API tokens from .env.

    Priority: KAGGLE_ACCOUNTS > KAGGLE_API_TOKENS > KAGGLE_API_TOKEN.
    Only account aliases are exposed in CLI and JSONL logs.
    """
    names = os.getenv("KAGGLE_ACCOUNTS", "").strip()
    if not names:
        compact = os.getenv("KAGGLE_API_TOKENS", "").strip()
        if compact:
            tokens = [token.strip() for token in compact.split(",")]
            if any(not token for token in tokens):
                raise AuthenticationError("KAGGLE_API_TOKENS contains an empty token; remove extra commas.")
            return tuple(KaggleAccount(f"account_{index}", token) for index, token in enumerate(tokens, 1))
        token = os.getenv("KAGGLE_API_TOKEN", "").strip()
        return (KaggleAccount("default", token),) if token else ()
    output: list[KaggleAccount] = []
    seen: set[str] = set()
    for name in names.split(","):
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise AuthenticationError("KAGGLE_ACCOUNTS must contain comma-separated account names (letters, digits, underscores).")
        key = name.upper()
        if key in seen:
            raise AuthenticationError("Duplicate account name in KAGGLE_ACCOUNTS.")
        seen.add(key)
        token = os.getenv(f"KAGGLE_API_TOKEN_{key}", "").strip()
        if not token:
            raise AuthenticationError(f"Missing KAGGLE_API_TOKEN_{key} for configured account {name!r}.")
        output.append(KaggleAccount(name, token))
    return tuple(output)


def _positive_env_float(name: str, default: float, *, allow_zero: bool = False) -> float:
    raw = os.getenv(name, str(default))
    try:
        result = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(result) or (result < 0 if allow_zero else result <= 0):
        raise ValueError(f"{name} must be {'non-negative' if allow_zero else 'positive'} and finite")
    return result


class KaggleAccountPool:
    """Sticky-first quota-aware failover between explicitly configured accounts.

    Provider quota includes daily AND monthly limits; all periods must have
    enough remaining. Fail closed when the quota endpoint is unreachable.
    """

    def __init__(self, accounts: tuple[KaggleAccount, ...], *, api: KaggleAPI | None = None,
                 min_remaining_usd: float = 0.05, quota_cache_seconds: float = 30.0):
        if not accounts:
            raise AuthenticationError("No Kaggle API token configured. Set KAGGLE_API_TOKEN or KAGGLE_ACCOUNTS in .env.")
        if not math.isfinite(min_remaining_usd) or min_remaining_usd < 0:
            raise ValueError("min_remaining_usd must be non-negative")
        if not math.isfinite(quota_cache_seconds) or quota_cache_seconds < 0:
            raise ValueError("quota_cache_seconds must be non-negative")
        self.accounts = accounts
        self.api = api if api is not None else OfficialKaggleAPI()
        self.min_remaining_usd = min_remaining_usd
        self.quota_cache_seconds = quota_cache_seconds
        self._lock = threading.RLock()
        self._selected = 0
        self._quota_cache: dict[str, tuple[float, tuple[QuotaBalance, ...]]] = {}
        self._creds: dict[str, tuple[float, ProxyCredentials]] = {}
        self._blocked_until: dict[str, float] = {}

    @classmethod
    def from_env(cls, *, api: KaggleAPI | None = None) -> "KaggleAccountPool":
        return cls(configured_accounts(), api=api,
                   min_remaining_usd=_positive_env_float("FRONTIER_MIN_REMAINING_USD", 0.05, allow_zero=True),
                   quota_cache_seconds=_positive_env_float("FRONTIER_QUOTA_CACHE_SECONDS", 30, allow_zero=True))

    def _quota(self, account: KaggleAccount, *, refresh: bool = False) -> tuple[QuotaBalance, ...]:
        cached = self._quota_cache.get(account.name)
        if cached and not refresh and time.monotonic() < cached[0]:
            return cached[1]
        try:
            balances = self.api.get_quotas(account.api_token)
        except Exception as exc:
            raise ModelRequestError(f"Cannot inspect Model Proxy quota for account {account.name!r}; check Kaggle API authentication/network.") from exc
        if not balances:
            raise ModelRequestError(f"Empty Model Proxy quota for account {account.name!r}.")
        self._quota_cache[account.name] = (time.monotonic() + self.quota_cache_seconds, balances)
        return balances

    def quotas(self, *, refresh: bool = True) -> list[dict[str, Any]]:
        """Safe account-scoped quota rows for CLI reports (no secrets)."""
        with self._lock:
            result = []
            for account in self.accounts:
                try:
                    balances = self._quota(account, refresh=refresh)
                    result.append({"account": account.name, "available": all(
                        b.remaining_usd > self.min_remaining_usd for b in balances),
                        "balances": [b.to_dict() for b in balances]})
                except ModelRequestError:
                    result.append({"account": account.name, "available": False,
                                   "error": "Quota unavailable; check credentials or network."})
            return result

    def next_available(self, *, exclude: set[str] | None = None) -> tuple[KaggleAccount, ProxyCredentials]:
        with self._lock:
            excluded = exclude or set()
            last_error: Exception | None = None
            for step in range(len(self.accounts)):
                index = (self._selected + step) % len(self.accounts)
                account = self.accounts[index]
                if account.name in excluded or time.monotonic() < self._blocked_until.get(account.name, 0.0):
                    continue
                try:
                    balances = self._quota(account)
                    # Strictly greater ensures zero remaining is always exhausted.
                    if not all(b.remaining_usd > self.min_remaining_usd for b in balances):
                        continue
                    creds = self._credentials(account)
                except (ModelRequestError, AuthenticationError) as exc:
                    last_error = exc
                    continue
                self._selected = index
                return account, creds
            raise ModelRequestError(
                "No configured Kaggle account has verified available Model Proxy quota and valid credentials. "
                "Run `frontier quota` or check your account tokens."
            ) from last_error

    def _credentials(self, account: KaggleAccount) -> ProxyCredentials:
        cached = self._creds.get(account.name)
        now = datetime.now(timezone.utc)
        if cached:
            created, creds = cached
            if creds.expires_at is not None:
                if creds.expires_at > now + timedelta(seconds=60):
                    return creds
            elif time.monotonic() - created < 600:  # Unknown TTL: re-mint periodically.
                return creds
        try:
            creds = self.api.create_proxy_credentials(account.api_token)
        except Exception as exc:
            raise AuthenticationError(f"Cannot obtain Model Proxy credentials for account {account.name!r} using its Kaggle API token.") from exc
        if not creds.base_url or not creds.token:
            raise AuthenticationError(f"Kaggle returned empty Model Proxy credentials for account {account.name!r}.")
        if creds.expires_at and creds.expires_at <= now:
            raise AuthenticationError(f"Kaggle returned expired Model Proxy credentials for account {account.name!r}.")
        self._creds[account.name] = (time.monotonic(), creds)
        return creds

    def invalidate_credentials(self, name: str) -> None:
        with self._lock:
            self._creds.pop(name, None)

    def block_exhausted(self, name: str) -> None:
        with self._lock:
            self._quota_cache.pop(name, None)
            self._blocked_until[name] = time.monotonic() + max(30.0, self.quota_cache_seconds)
            for idx, account in enumerate(self.accounts):
                if account.name == name:
                    self._selected = (idx + 1) % len(self.accounts)
                    break

    def refresh_credentials(self) -> list[dict[str, str]]:
        """Explicit in-memory refresh; next process will mint its own credentials."""
        with self._lock:
            report = []
            for account in self.accounts:
                self._creds.pop(account.name, None)
                try:
                    self._credentials(account)
                    report.append({"account": account.name, "status": "ready"})
                except AuthenticationError:
                    report.append({"account": account.name, "status": "failed"})
            return report
