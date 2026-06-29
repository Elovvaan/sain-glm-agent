"""File-backed JSON conversation storage."""

from __future__ import annotations

import json
import re
from pathlib import Path
from threading import RLock

from sain_glm_agent.domain.entities import ConversationHistory, Message
from sain_glm_agent.infrastructure.memory.base import BaseMemoryStore


class FileMemoryStore(BaseMemoryStore):
    """Persists session histories as JSON files on disk."""

    def __init__(self, base_dir: str | Path = ".sain_glm_agent_memory") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def add_message(self, session_id: str, message: Message) -> None:
        """Append a message and persist the updated history."""
        with self._lock:
            history = self.get_history(session_id)
            history.messages.append(message)
            self._save(history)

    def get_history(self, session_id: str) -> ConversationHistory:
        """Load history for a session or return an empty history."""
        with self._lock:
            path = self._session_path(session_id)
            if not path.exists():
                return ConversationHistory(session_id=session_id, messages=[])
            data = json.loads(path.read_text(encoding="utf-8"))
            return ConversationHistory.model_validate(data)

    def clear(self, session_id: str) -> None:
        """Delete a session file if present."""
        with self._lock:
            path = self._session_path(session_id)
            if path.exists():
                path.unlink()

    def list_sessions(self) -> list[str]:
        """List session identifiers from all stored JSON documents."""
        with self._lock:
            sessions: list[str] = []
            for path in sorted(self.base_dir.glob("*.json")):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    sessions.append(data["session_id"])
                except (OSError, json.JSONDecodeError, KeyError):
                    continue
            return sessions

    def _save(self, history: ConversationHistory) -> None:
        path = self._session_path(history.session_id)
        payload = history.model_dump(mode="json")
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _session_path(self, session_id: str) -> Path:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id).strip("_") or "session"
        return self.base_dir / f"{safe_name}.json"
