"""Conversation memory and persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from .models import Message


@dataclass(slots=True)
class ConversationTurn:
    """One conversation turn persisted to disk."""

    role: str
    content: str
    timestamp: str


class ConversationMemory:
    """Persistent conversation memory stored as JSON."""

    def __init__(self, memory_file: Path) -> None:
        self.memory_file = memory_file
        self.turns: list[ConversationTurn] = []
        self.load()

    def load(self) -> None:
        """Load saved conversation turns if the file exists."""

        if not self.memory_file.exists():
            self.turns = []
            return
        payload = json.loads(self.memory_file.read_text(encoding="utf-8"))
        self.turns = [ConversationTurn(**item) for item in payload]

    def save(self) -> None:
        """Persist conversation turns to disk."""

        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(turn) for turn in self.turns]
        self.memory_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def add_turn(self, role: str, content: str) -> None:
        """Append and persist a new conversation turn."""

        self.turns.append(
            ConversationTurn(
                role=role,
                content=content,
                timestamp=datetime.now(tz=UTC).isoformat(),
            )
        )
        self.save()

    def recent_messages(self, limit: int = 6) -> list[Message]:
        """Return the most recent turns as normalized messages."""

        return [Message(role=turn.role, content=turn.content) for turn in self.turns[-limit:]]

    def render_recent_history(self, limit: int = 6) -> str:
        """Render recent history as plain text for prompts."""

        if not self.turns:
            return "No previous conversation history."
        return "\n".join(f"{turn.role}: {turn.content}" for turn in self.turns[-limit:])
