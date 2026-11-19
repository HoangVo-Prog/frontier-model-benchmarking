"""Unified text, image, stateless and stateful Kaggle Model Proxy inference."""
from __future__ import annotations

import importlib
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .auth import REFRESH, load_project_env, require_proxy_credentials
from .errors import AuthenticationError, ModelRequestError, UnsupportedParameterError
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
    output = {name: getattr(obj, name, None) if obj is not None else None for name in fields}
    total = output["total_cost_nanodollars"]
    output["total_cost"] = total / 1_000_000_000 if total is not None else None
    return output


def _map_sdk_error(exc: Exception) -> ModelRequestError:
    """Map error categories without serializing untrusted exception text/tokens."""
    message = str(exc).lower()
    if any(x in message for x in ("401", "403", "unauthorized", "expired", "invalid api key", "authentication")):
        return AuthenticationError(f"Model Proxy authentication failed or expired. Run `{REFRESH}`.")
    if any(x in message for x in ("quota", "resource exhausted", "insufficient funds")):
        return ModelRequestError("Kaggle Model Proxy quota exhausted. Run `frontier quota`.")
    if any(x in message for x in ("unsupported image", "image not supported", "vision", "does not support image", "image input")):
        return ModelRequestError("The selected model does not support this image input. Choose a vision-capable model.")
    if any(x in message for x in ("parameter", "invalid argument", "bad request", "400")):
        return ModelRequestError("The model rejected the request or a generation parameter. Check model compatibility.")
    if any(x in message for x in ("model not found", "unknown model", "404", "not available")):
        return ModelRequestError("Model is unavailable to this Model Proxy credential. Check `frontier models list` and your local access restrictions.")
    if any(x in message for x in ("429", "rate limit")):
        return ModelRequestError("Model Proxy rate limit reached. Retry later.")
    return ModelRequestError(f"Model Proxy request failed ({type(exc).__name__}). See Kaggle service logs for details.")


class FrontierClient:
    """Small facade around the official Kaggle Benchmarks SDK.

    Pass `sdk=` only for unit testing. By default prompts are logged but their
    input text is omitted; opt in with `log_prompts=True` for reproducibility.
    """

    def __init__(
        self, model: str, *, temperature: float | None = None,
        reasoning: str | None = None, api: str = "openai",
        log_dir: str | Path | None = None, logging: bool = True,
        log_prompts: bool = False, env_file: str | Path | None = None,
        sdk: Any = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty Kaggle Model Proxy model ID")
        if temperature is not None and (not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2):
            raise UnsupportedParameterError("temperature must be between 0 and 2")
        if reasoning is not None and reasoning not in {"none", "low", "medium", "high"}:
            raise UnsupportedParameterError("reasoning must be one of none, low, medium, high")
        if api not in {"openai", "genai"}:
            raise UnsupportedParameterError("api must be 'openai' or 'genai' (both through Kaggle Model Proxy)")
        self.model = model.strip()
        self.temperature = temperature
        self.reasoning = reasoning
        self.api = api
        self.log_prompts = log_prompts
        self.logger = JSONLLogger(log_dir or os.getenv("FRONTIER_LOG_DIR") or "logs") if logging else None
        load_project_env(env_file)
        if sdk is None:
            require_proxy_credentials()
            try:
                sdk = importlib.import_module("kaggle_benchmarks")
            except ImportError as exc:
                raise ModelRequestError("Missing kaggle-benchmarks SDK; run `python -m pip install -e '.[dev]'`.") from exc
        self.sdk = sdk
        try:
            loader = getattr(getattr(sdk, "kaggle", None), "load_model", None)
            if callable(loader):
                self._model = loader(self.model, api=self.api)
            else:
                self._model = sdk.llms[self.model]
        except Exception as exc:
            raise _map_sdk_error(exc) from exc
        if self.temperature is not None and getattr(self._model, "support_temperature", True) is False:
            raise UnsupportedParameterError("Installed Kaggle SDK/backend does not support temperature for this model; omit it or choose a compatible Kaggle backend.")

    @contextmanager
    def _fresh_chat(self, system_prompt: str | None = None) -> Iterator[Any]:
        with self.sdk.chats.new(f"frontier-{uuid.uuid4().hex[:8]}", system_instructions=system_prompt, orphan=True) as thread:
            yield thread

    def _invoke(self, prompt: str, image: str | Path | None, session: "ChatSession | None", schema: type | None) -> FrontierResponse:
        request_id = uuid.uuid4().hex
        timestamp = _timestamp()
        start = time.perf_counter()
        kind = "text_image" if image is not None else "text"
        record: dict[str, Any] = {
            "request_id": request_id, "timestamp": timestamp,
            "session_id": session.session_id if session else None, "model": self.model,
            "input": {"type": kind, "prompt": prompt if self.log_prompts else None,
                      "images": []},
            "output": {"text": None, "structured": None},
            "usage": _usage(None), "latency_seconds": None,
            "status": "failed", "error": None,
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
                sdk_images = self.sdk.content_types.images
                payload, ref = prepare_image(image, sdk_images)
                kwargs["image"] = payload
                record["input"]["images"] = [ref]
            if session is None:
                with self._fresh_chat() as thread:
                    answer = self._model.prompt(prompt, **kwargs)
                    message = next((m for m in reversed(thread.messages) if getattr(getattr(m, "sender", None), "role", None) == "assistant"), None)
                    usage = _usage(getattr(message, "usage", None))
            else:
                # Re-enter this session only for the current call. This keeps
                # separate (even interleaved) sessions from sharing SDK history.
                with self.sdk.contexts.enter(chat=session._thread):
                    before = len(session._thread.history)
                    try:
                        answer = self._model.prompt(prompt, **kwargs)
                    except Exception:
                        del session._thread.history[before:]
                        raise
                    message = next((m for m in reversed(session._thread.messages) if getattr(getattr(m, "sender", None), "role", None) == "assistant"), None)
                    usage = _usage(getattr(message, "usage", None))
            text = response_text(answer)
            structured = None if isinstance(answer, str) else answer
            response = FrontierResponse(text=text, structured=structured, model=self.model,
                                        usage=usage, latency_seconds=time.perf_counter() - start,
                                        request_id=request_id, timestamp=timestamp)
            record.update(status="success", output={"text": text, "structured": safe_json(structured)})
            record["usage"] = usage
            return response
        except (ValueError, AuthenticationError, UnsupportedParameterError) as exc:
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        except Exception as exc:
            error = _map_sdk_error(exc)
            record["error"] = {"type": type(error).__name__, "message": str(error)}
            raise error from exc
        finally:
            record["latency_seconds"] = time.perf_counter() - start
            if self.logger is not None:
                self.logger.append(record)

    def prompt(self, prompt: str, *, image: str | Path | None = None, schema: type | None = None) -> FrontierResponse:
        """One independent inference request, optionally with PNG/JPEG image."""
        return self._invoke(prompt, image, None, schema)

    def chat(self, *, system_prompt: str | None = None) -> "ChatSession":
        """Create an isolated, resettable session (use as a context manager)."""
        from .chat import ChatSession
        return ChatSession(self, system_prompt=system_prompt)
