"""Text and image chat via Kaggle's official Model Proxy SDK."""
from __future__ import annotations

import importlib
import os
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .accounts import KaggleAccount, KaggleAccountPool, ProxyCredentials
from .auth import load_project_env, validate_local_model
from .errors import (AuthenticationError, FrontierError, ImageInputError,
                     ModelRequestError, QuotaExhaustedError, UnsupportedParameterError)
from .image import prepare_image
from .logging import JSONLLogger
from .schemas import FrontierResponse, response_text, safe_json


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _usage(obj: Any) -> dict[str, Any]:
    fields = (
        "input_tokens", "output_tokens", "input_tokens_cost_nanodollars",
        "output_tokens_cost_nanodollars", "total_cost_nanodollars", "total_backend_latency_ms",
    )
    result = {field: getattr(obj, field, None) if obj is not None else None for field in fields}
    total = result["total_cost_nanodollars"]
    result["total_cost"] = total / 1_000_000_000 if total is not None else None
    return result


def _map_sdk_error(exc: Exception) -> FrontierError:
    """Classify common service errors without reflecting raw provider messages."""
    if isinstance(exc, FrontierError):
        return exc
    message = str(exc).lower()
    # Quota exhaustion is not the same as a generic 429 rate limit: only the
    # former authorizes rotation. This also prevents retrying charged requests.
    if any(x in message for x in ("quota", "budget exceeded", "insufficient funds", "spend limit")):
        return QuotaExhaustedError("Model Proxy quota exhausted for this account.")
    if any(x in message for x in ("401", "403", "unauthorized", "expired", "invalid api key", "authentication")):
        return AuthenticationError("Model Proxy authentication failed; temporary credentials may be invalid.")
    if any(x in message for x in ("429", "rate limit", "too many requests")):
        return ModelRequestError("Model Proxy rate limited this request; retry later (no automatic account rotation).")
    if any(x in message for x in ("unsupported image", "image not supported", "vision", "does not support image")):
        return ModelRequestError("The selected model does not support this image input.")
    if any(x in message for x in ("model not found", "unknown model", "404", "not available")):
        return ModelRequestError("Model not available to the current local credential. Check `frontier models list`.")
    if any(x in message for x in ("parameter", "invalid argument", "bad request", "400")):
        return ModelRequestError("The model rejected this request or a parameter.")
    return ModelRequestError(f"Model Proxy request failed ({type(exc).__name__}).")


class FrontierClient:
    """Minimal public API with account-aware failover and append-only JSONL.

    ``sdk=`` and ``account_pool=`` allow credential-free tests. Without SDK
    injection, tokens are minted in memory through Kaggle's official API.
    """

    def __init__(self, model: str, *, temperature: float | None = None,
                 reasoning: str | None = None, api: str = "openai",
                 log_dir: str | Path | None = None, logging: bool = True,
                 log_prompts: bool = False, env_file: str | Path | None = None,
                 sdk: Any = None, account_pool: KaggleAccountPool | None = None) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty Kaggle Model Proxy model ID")
        if temperature is not None and (not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2):
            raise UnsupportedParameterError("temperature must be between 0 and 2")
        if reasoning is not None and reasoning not in {"none", "low", "medium", "high"}:
            raise UnsupportedParameterError("reasoning must be one of none, low, medium, high")
        if api not in {"openai", "genai"}:
            raise UnsupportedParameterError("api must be 'openai' or 'genai' (both through Kaggle Model Proxy)")
        load_project_env(env_file)
        self.model = model.strip()
        self.temperature = temperature
        self.reasoning = reasoning
        self.api = api
        self.log_prompts = log_prompts
        self.logger = JSONLLogger(log_dir or os.getenv("FRONTIER_LOG_DIR") or "logs") if logging else None
        self._model_lock = threading.RLock()
        self._models: dict[str, tuple[str, Any]] = {}
        # Preserve existing sdk-injection API in tests without network/account config.
        self.account_pool = account_pool if account_pool is not None else (KaggleAccountPool.from_env() if sdk is None else None)
        if sdk is None:
            local = validate_local_model(self.model)
            if os.getenv("LLM_DEFAULT") not in local:
                # The official SDK eagerly constructs a default model at import;
                # setting a valid LOCAL default avoids a KeyError.
                os.environ["LLM_DEFAULT"] = self.model
            try:
                sdk = importlib.import_module("kaggle_benchmarks")
            except ImportError as exc:
                raise ModelRequestError("Missing kaggle-benchmarks SDK; install with `python -m pip install -e '.[dev]'`.") from exc
            except Exception as exc:
                raise _map_sdk_error(exc) from exc
        self.sdk = sdk
        self._model = None
        if self.account_pool is None:
            # Compatibility path used only by injected unit-test SDKs.
            try:
                loader = getattr(getattr(sdk, "kaggle", None), "load_model", None)
                self._model = loader(self.model, api=self.api) if callable(loader) else sdk.llms[self.model]
            except Exception as exc:
                raise _map_sdk_error(exc) from exc
            self._check_temperature(self._model)

    def _check_temperature(self, model: Any) -> None:
        if self.temperature is not None and getattr(model, "support_temperature", True) is False:
            raise UnsupportedParameterError("Kaggle SDK/backend does not support temperature for this model.")

    def _account_model(self, account: KaggleAccount, credentials: ProxyCredentials) -> Any:
        with self._model_lock:
            cached = self._models.get(account.name)
            if cached is not None and cached[0] == credentials.token:
                return cached[1]
            # ModelProxy is the official kaggle-benchmarks SDK model factory;
            # explicit credentials prevent global environment cross-talk.
            factory = getattr(getattr(self.sdk, "kaggle", None), "ModelProxy", None)
            if not callable(factory):
                raise ModelRequestError("Installed kaggle-benchmarks SDK lacks Kaggle ModelProxy factory.")
            try:
                model = factory(model=self.model, api=self.api, api_key=credentials.token,
                                base_url=credentials.base_url)
                self._check_temperature(model)
            except UnsupportedParameterError:
                raise
            except Exception as exc:
                raise _map_sdk_error(exc) from exc
            self._models[account.name] = (credentials.token, model)
            return model

    @contextmanager
    def _fresh_chat(self, system_prompt: str | None = None) -> Iterator[Any]:
        with self.sdk.chats.new(f"frontier-{uuid.uuid4().hex[:8]}", system_instructions=system_prompt, orphan=True) as thread:
            yield thread

    def _call_model(self, model: Any, prompt: str, kwargs: dict[str, Any],
                    session: "ChatSession | None") -> tuple[Any, dict[str, Any]]:
        if session is None:
            with self._fresh_chat() as thread:
                answer = model.prompt(prompt, **kwargs)
                message = next((m for m in reversed(thread.messages)
                                if getattr(getattr(m, "sender", None), "role", None) == "assistant"), None)
                return answer, _usage(getattr(message, "usage", None))
        with self.sdk.contexts.enter(chat=session._thread):
            before = len(session._thread.history)
            try:
                answer = model.prompt(prompt, **kwargs)
            except Exception:
                del session._thread.history[before:]
                raise
            message = next((m for m in reversed(session._thread.messages)
                            if getattr(getattr(m, "sender", None), "role", None) == "assistant"), None)
            return answer, _usage(getattr(message, "usage", None))

    def _invoke(self, prompt: str, image: str | Path | None,
                session: "ChatSession | None", schema: type | None) -> FrontierResponse:
        request_id = uuid.uuid4().hex
        timestamp = _timestamp()
        start = time.perf_counter()
        record: dict[str, Any] = {
            "request_id": request_id, "timestamp": timestamp,
            "session_id": session.session_id if session else None, "model": self.model,
            "account": None, "accounts_attempted": [],
            "input": {"type": "text_image" if image is not None else "text",
                      "prompt": prompt if self.log_prompts else None, "images": []},
            "output": {"text": None, "structured": None}, "usage": _usage(None),
            "latency_seconds": None, "status": "failed", "error": None,
        }
        try:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("prompt must be a non-empty string")
            kwargs: dict[str, Any] = {}
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            if self.reasoning is not None:
                kwargs["reasoning"] = self.reasoning
            if schema is not None:
                kwargs["schema"] = schema
            if image is not None:
                try:
                    payload, ref = prepare_image(image, self.sdk.content_types.images)
                except ImageInputError:
                    raise
                except Exception as exc:
                    raise _map_sdk_error(exc) from exc
                kwargs["image"] = payload
                record["input"]["images"] = [ref]

            excluded: set[str] = set()
            refreshed: set[str] = set()
            while True:
                account = None
                if self.account_pool is None:
                    model = self._model
                else:
                    account, credentials = self.account_pool.next_available(exclude=excluded)
                    record["account"] = account.name
                if account is not None:
                    record["accounts_attempted"].append(account.name)
                try:
                    if account is not None:
                        model = self._account_model(account, credentials)
                    answer, usage = self._call_model(model, prompt, kwargs, session)
                    text = response_text(answer)
                    structured = None if isinstance(answer, str) else answer
                    encoded = safe_json(structured)
                    record.update(status="success", output={"text": text, "structured": encoded}, usage=usage)
                    return FrontierResponse(text=text, structured=structured, model=self.model,
                                            usage=usage, latency_seconds=time.perf_counter() - start,
                                            request_id=request_id, timestamp=timestamp,
                                            metadata={"account": account.name if account is not None else None})
                except (UnsupportedParameterError, ImageInputError):
                    raise
                except Exception as exc:
                    mapped = _map_sdk_error(exc)
                    if self.account_pool is not None and account is not None:
                        if isinstance(mapped, QuotaExhaustedError):
                            self.account_pool.block_exhausted(account.name)
                            excluded.add(account.name)
                            # Only quota-specific failures may automatically fail over.
                            if len(excluded) < len(self.account_pool.accounts):
                                continue
                        elif isinstance(mapped, AuthenticationError) and account.name not in refreshed:
                            self.account_pool.invalidate_credentials(account.name)
                            refreshed.add(account.name)
                            continue  # Mint a fresh proxy token for this SAME account once.
                    raise mapped from exc
        except (ValueError, FrontierError) as exc:
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            record["latency_seconds"] = time.perf_counter() - start
            if self.logger is not None:
                self.logger.append(record)

    def prompt(self, prompt: str, *, image: str | Path | None = None,
               schema: type | None = None) -> FrontierResponse:
        """One independent inference request, optionally with PNG/JPEG image."""
        return self._invoke(prompt, image, None, schema)

    def chat(self, *, system_prompt: str | None = None) -> "ChatSession":
        """Create an isolated resettable session (use as a context manager)."""
        from .chat import ChatSession
        return ChatSession(self, system_prompt=system_prompt)
