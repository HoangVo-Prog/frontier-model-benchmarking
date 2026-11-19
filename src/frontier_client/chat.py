"""Independent conversation sessions using SDK chat contexts."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from .schemas import FrontierResponse, safe_json
from .image import sanitized_image_ref


class ChatSession:
    def __init__(self, client: Any, *, system_prompt: str | None = None) -> None:
        self.client = client
        self.system_prompt = system_prompt
        self.session_id = uuid.uuid4().hex
        self.history: list[dict[str, Any]] = []
        self._thread: Any = None
        self._active = False

    def _new_thread(self) -> Any:
        # Generate an SDK Chat via the supported chats.new primitive, then
        # attach it to requests with SDK contexts.enter(chat=...).
        with self.client._fresh_chat(self.system_prompt) as thread:
            return thread

    def __enter__(self) -> "ChatSession":
        if self._active:
            raise RuntimeError("ChatSession is already active")
        self._thread = self._new_thread()
        self._active = True
        return self

    def __exit__(self, *_: Any) -> None:
        self._active = False
        self._thread = None

    def send(self, prompt: str, *, image: str | Path | None = None, schema: type | None = None) -> FrontierResponse:
        if not self._active:
            raise RuntimeError("Use `with client.chat() as chat:` before sending messages.")
        response = self.client._invoke(prompt, image, self, schema)
        self.history.extend([
            {"role": "user", "content": prompt, "image": sanitized_image_ref(image) if image is not None else None},
            {"role": "assistant", "content": response.text, "request_id": response.request_id,
             "usage": response.usage, "timestamp": response.timestamp},
        ])
        return response

    def reset(self) -> None:
        if not self._active:
            raise RuntimeError("Reset requires an active chat session")
        self._thread = self._new_thread()
        self.history.clear()
        self.session_id = uuid.uuid4().hex

    def export(self, destination: str | Path) -> Path:
        output = Path(destination).expanduser()
        if output.suffix.lower() != ".json":
            raise ValueError("Chat export currently supports .json files only")
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"session_id": self.session_id, "model": self.client.model,
                   "system_prompt": self.system_prompt, "messages": self.history}
        output.write_text(json.dumps(safe_json(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return output
