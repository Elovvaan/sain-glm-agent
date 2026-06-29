"""Tests for sain_glm_agent.memory.conversation."""

from __future__ import annotations

import json

from sain_glm_agent.memory.conversation import ConversationMemory
from sain_glm_agent.providers.base import MessageRole


class TestConversationMemory:
    def test_empty_on_creation(self):
        mem = ConversationMemory()
        assert mem.message_count == 0
        assert mem.get_messages() == []

    def test_set_system(self):
        mem = ConversationMemory()
        mem.set_system("You are helpful.")
        msgs = mem.get_messages()
        assert len(msgs) == 1
        assert msgs[0].role == MessageRole.SYSTEM
        assert msgs[0].content == "You are helpful."

    def test_add_user(self):
        mem = ConversationMemory()
        mem.add_user("Hello!")
        assert mem.message_count == 1
        msgs = mem.get_messages()
        assert msgs[0].role == MessageRole.USER

    def test_add_assistant(self):
        mem = ConversationMemory()
        mem.add_assistant("Hi there!")
        assert mem.message_count == 1
        assert mem.get_messages()[0].role == MessageRole.ASSISTANT

    def test_add_tool_result(self):
        mem = ConversationMemory()
        mem.add_tool_result("42", tool_call_id="call_1")
        assert mem.message_count == 1
        msg = mem.get_messages()[0]
        assert msg.role == MessageRole.TOOL
        assert msg.tool_call_id == "call_1"

    def test_system_always_first(self):
        mem = ConversationMemory()
        mem.add_user("Hello")
        mem.set_system("System prompt")
        msgs = mem.get_messages()
        assert msgs[0].role == MessageRole.SYSTEM
        assert msgs[1].role == MessageRole.USER

    def test_system_replaced(self):
        mem = ConversationMemory()
        mem.set_system("First")
        mem.set_system("Second")
        assert mem.get_messages()[0].content == "Second"

    def test_sliding_window_eviction(self):
        mem = ConversationMemory(max_messages=4)
        for i in range(6):
            mem.add_user(f"user {i}")
        # Window should not exceed max_messages
        assert mem.message_count <= 4

    def test_clear(self):
        mem = ConversationMemory()
        mem.set_system("Sys")
        mem.add_user("hello")
        mem.clear()
        assert mem.message_count == 0
        # System should survive clear
        assert mem.get_messages()[0].role == MessageRole.SYSTEM

    def test_token_estimate_default(self):
        mem = ConversationMemory()
        mem.add_user("Hello, world!")
        # 13 chars / 4 ≈ 3
        assert mem.token_estimate >= 1

    def test_custom_token_counter(self):
        mem = ConversationMemory(token_counter=lambda t: len(t))
        mem.add_user("abc")
        assert mem.token_estimate == 3

    def test_to_json_round_trip(self):
        mem = ConversationMemory()
        mem.set_system("Sys")
        mem.add_user("Hello")
        mem.add_assistant("Hi")
        raw = mem.to_json()
        data = json.loads(raw)
        assert len(data) == 3

        restored = ConversationMemory.from_json(raw)
        msgs = restored.get_messages()
        assert msgs[0].role == MessageRole.SYSTEM
        assert msgs[1].role == MessageRole.USER
        assert msgs[2].role == MessageRole.ASSISTANT

    def test_save_and_load(self, tmp_path):
        mem = ConversationMemory()
        mem.set_system("Sys")
        mem.add_user("Question")
        mem.add_assistant("Answer")

        path = tmp_path / "convo.json"
        mem.save(path)
        assert path.exists()

        loaded = ConversationMemory.load(path)
        assert loaded.message_count == 2
        assert loaded.get_messages()[0].content == "Sys"
