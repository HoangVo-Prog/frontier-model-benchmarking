"""Structured inference responses and safe JSON serialization."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any


def safe_json(value: Any) -> Any:
    """Only serialize explicitly supported primitives; never repr arbitrary SDK objects."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return safe_json(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items() if isinstance(k, (str, int, float, bool))}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if callable(getattr(value, "model_dump", None)):
        return safe_json(value.model_dump(mode="json"))
    raise TypeError(f"Unsupported structured output type: {type(value).__name__}")


@dataclass(frozen=True)
class FrontierResponse:
    text: str
    model: str
    usage: dict[str, Any]
    latency_seconds: float
    request_id: str
    timestamp: str
    structured: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return safe_json(dataclasses.asdict(self))


def response_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(safe_json(value), ensure_ascii=False)
