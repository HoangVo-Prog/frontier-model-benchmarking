from __future__ import annotations
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ChatSession:
    model_id: str
    client: Any
    system_prompt: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def ask(self, text: str, *, image: Any = None, audio: Any = None) -> Any:
        response = self.client.prompt(text, image=image, audio=audio)
        self.history.extend([
            {"role": "user", "content": text, "timestamp": datetime.now(timezone.utc).isoformat()},
            {"role": "assistant", "content": response.text, "usage": response.usage,
             "latency_seconds": response.latency_seconds, "timestamp": datetime.now(timezone.utc).isoformat()},
        ])
        return response

    def reset(self) -> None:
        self.history.clear()

    def export(self, path: str | Path) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix.lower() in {".md", ".markdown"}:
            text = f"# Chat: {self.model_id}\n\n"
            for item in self.history:
                text += f"### {item['role'].title()}\n\n{item['content']}\n\n"
            target.write_text(text, encoding="utf-8")
        else:
            target.write_text(json.dumps({"model_id": self.model_id, "system_prompt": self.system_prompt,
                                          "messages": self.history}, ensure_ascii=False, indent=2), encoding="utf-8")
        return target
