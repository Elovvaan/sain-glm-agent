from __future__ import annotations

from pathlib import Path

from sain_glm_agent.domain.entities import Message
from sain_glm_agent.infrastructure.memory.file_store import FileMemoryStore
from sain_glm_agent.infrastructure.memory.inmemory import InMemoryStore


def test_inmemory_store_roundtrip() -> None:
    store = InMemoryStore()
    store.add_message("session-1", Message(role="user", content="hello"))
    history = store.get_history("session-1")
    assert history.session_id == "session-1"
    assert len(history.messages) == 1
    assert store.list_sessions() == ["session-1"]
    store.clear("session-1")
    assert store.get_history("session-1").messages == []


def test_file_memory_store_roundtrip(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    store.add_message("session/1", Message(role="user", content="hello"))
    store.add_message("session/1", Message(role="assistant", content="world"))
    history = store.get_history("session/1")
    assert [message.content for message in history.messages] == ["hello", "world"]
    assert store.list_sessions() == ["session/1"]
    store.clear("session/1")
    assert store.list_sessions() == []
