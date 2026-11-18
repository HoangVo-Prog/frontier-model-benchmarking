"""Thin adapter over public Kaggle Benchmarks SDK primitives."""
from __future__ import annotations

import importlib
import time
from contextlib import contextmanager
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelResponse:
    text: str
    latency_seconds: float
    usage: dict[str, Any] = field(default_factory=dict)


class ModelRequestError(RuntimeError):
    """Actionable failure returned by Kaggle Model Proxy or the SDK."""


class KaggleModelClient:
    def __init__(self, model_id: str, *, system_prompt: str | None = None,
                 generation: dict[str, Any] | None = None, retries: int = 2,
                 backoff_seconds: float = 1.0, sdk: Any = None):
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.generation = generation or {}
        self.retries = max(0, retries)
        self.backoff_seconds = max(0.0, backoff_seconds)
        self.sdk = sdk or self._import_sdk()
        self.model = self._load_model()
        self._active_context = None

    @staticmethod
    def _import_sdk() -> Any:
        try:
            return importlib.import_module("kaggle_benchmarks")
        except ImportError as exc:
            raise RuntimeError("Kaggle Benchmarks SDK is missing. Install with: python -m pip install -e '.[kaggle]'") from exc

    def _load_model(self) -> Any:
        # Current SDK exposes model selection through the llms registry. Keep the legacy
        # public loader fallback for SDK releases that still provide it.
        llms = getattr(self.sdk, "llms", None)
        if llms is not None:
            try:
                return llms[self.model_id]
            except (KeyError, TypeError, AttributeError):
                pass
        try:
            loader = importlib.import_module("kaggle_benchmarks.kaggle.models").load_model
            return loader(self.model_id)
        except Exception as exc:
            hint = _error_hint(str(exc).lower())
            message = f"{hint} Original SDK error: {_redact(str(exc))}" if hint else str(exc)
            raise ModelRequestError(f"Model {self.model_id!r} could not be initialized: {message}") from exc

    def prompt(self, prompt: str, *, image: Any = None, audio: Any = None) -> ModelResponse:
        import time
        delay = self.backoff_seconds
        for attempt in range(self.retries + 1):
            start = time.perf_counter()
            try:
                if self._active_context is not None:
                    return self._invoke(prompt, image, audio, self._active_context, start)
                context = self._chat_context()
                with context:
                    return self._invoke(prompt, image, audio, context, start)
            except Exception as exc:
                message = str(exc).lower()
                category = _error_hint(message)
                if category:
                    exc = ModelRequestError(f"{category} Original SDK error: {_redact(str(exc))}")
                retryable = any(token in message for token in ("rate limit", "429", "temporarily unavailable", "timeout", "503"))
                if attempt >= self.retries or not retryable:
                    raise exc
                time.sleep(delay)
                delay = max(delay * 2, self.backoff_seconds)

    def _invoke(self, prompt: str, image: Any, audio: Any, context: Any, start: float) -> ModelResponse:
        kwargs = _generation_kwargs(self.generation)
        if image is not None:
            kwargs["image"] = self._content("images", image)
        if audio is not None:
            kwargs["audio"] = self._content("audios", audio)
        result = self.model.prompt(prompt, **kwargs)
        elapsed = time.perf_counter() - start
        return ModelResponse(str(result), elapsed, _usage(getattr(context, "usage", None)))

    @contextmanager
    def conversation(self):
        """Keep one SDK chat context active across multiple turns."""
        context = self._chat_context()
        with context:
            self._active_context = context
            try:
                yield self
            finally:
                self._active_context = None

    def _chat_context(self):
        chats = getattr(self.sdk, "chats", None)
        if chats is not None and callable(getattr(chats, "new", None)):
            return chats.new(f"frontier-{self.model_id}", system_instructions=self.system_prompt)
        # Model wrappers may provide their own multi-turn context in older SDK releases.
        return _NullContext()

    def _content(self, kind: str, value: Any) -> Any:
        if not isinstance(value, (str, Path)):
            return value
        factories = getattr(getattr(self.sdk, "content_types", None), kind, None)
        if factories is None:
            try:
                factories = getattr(importlib.import_module("kaggle_benchmarks.content_types"), kind)
            except (ImportError, AttributeError):
                raise RuntimeError(f"The installed Kaggle SDK does not expose {kind} content support")
        return factories.from_path(str(value))


def _usage(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    result = {}
    for source, target in (("input_tokens", "input_tokens"), ("output_tokens", "output_tokens"),
                           ("input_tokens_cost_nanodollars", "input_cost_nanodollars"),
                           ("output_tokens_cost_nanodollars", "output_cost_nanodollars"),
                           ("total_cost_nanodollars", "total_cost_nanodollars"),
                           ("total_backend_latency_ms", "backend_latency_ms")):
        value = getattr(obj, source, None)
        if value is not None:
            result[target] = value
    return result


def _redact(message: str) -> str:
    for key, value in os.environ.items():
        if value and ("TOKEN" in key or "KEY" in key or "SECRET" in key):
            message = message.replace(value, "[REDACTED]")
    return message


def _error_hint(message: str) -> str | None:
    if any(x in message for x in ("401", "403", "unauthorized", "expired credential", "invalid token", "permission denied")):
        return "Kaggle Model Proxy authentication failed or expired. Refresh credentials with `kaggle benchmarks auth -y --env-file .env`."
    if any(x in message for x in ("quota exhausted", "quota exceeded", "insufficient quota")):
        return "Kaggle Model Proxy quota is exhausted. Check `kaggle benchmarks quota` or wait for quota renewal."
    if any(x in message for x in ("rate limit", "too many requests", "429")):
        return "Kaggle Model Proxy rate-limited this request. Automatic retries were attempted; reduce request volume or try again later."
    if any(x in message for x in ("unsupported model", "model not found", "unknown model", "404")):
        return "The selected model is unavailable to this account. Run `frontier models list` and choose an available model."
    if any(x in message for x in ("invalid payload", "invalid argument", "unsupported parameter", "bad request", "400")):
        return "The request payload or generation parameters are unsupported by this model; check its supported inputs and options."
    if "model_proxy_api_key" in message or "model_proxy_url" in message:
        return "Kaggle Model Proxy credentials are missing. Run `kaggle benchmarks auth -y --env-file .env`, then load that .env before starting the CLI."
    return None


def _generation_kwargs(parameters: dict[str, Any]) -> dict[str, Any]:
    """Translate framework aliases to the documented SDK prompt options."""
    options = dict(parameters)
    if "reasoning_effort" in options:
        if "reasoning" in options:
            raise ValueError("Set only one of generation.reasoning_effort or generation.reasoning")
        options["reasoning"] = options.pop("reasoning_effort")
    accepted = {"seed", "temperature", "reasoning", "tools", "schema", "extra_api_params"}
    invalid = options.keys() - accepted
    if invalid:
        raise ValueError(f"Kaggle Benchmarks SDK prompt options are unsupported: {', '.join(sorted(invalid))}. Use extra_api_params for provider-specific parameters.")
    return options


class _NullContext:
    usage = None
    def __enter__(self): return self
    def __exit__(self, *_): return False
