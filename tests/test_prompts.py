"""Tests for sain_glm_agent.prompts.manager."""

from __future__ import annotations

import pytest

from sain_glm_agent.prompts.manager import PromptManager, PromptTemplate


class TestPromptTemplate:
    def test_variables_extracted(self):
        tmpl = PromptTemplate("t", "Hello {name}, your age is {age}.")
        assert set(tmpl.variables) == {"name", "age"}

    def test_render_happy_path(self):
        tmpl = PromptTemplate("t", "Hello {name}!")
        assert tmpl.render(name="World") == "Hello World!"

    def test_render_missing_variable(self):
        tmpl = PromptTemplate("t", "Hello {name}!")
        with pytest.raises(KeyError):
            tmpl.render()

    def test_explicit_variables(self):
        tmpl = PromptTemplate("t", "x", variables=["a", "b"])
        assert tmpl.variables == ["a", "b"]

    def test_no_variables(self):
        tmpl = PromptTemplate("t", "Static text")
        assert tmpl.variables == []
        assert tmpl.render() == "Static text"


class TestPromptManager:
    def test_builtin_templates_present(self):
        pm = PromptManager()
        for name in (
            "system_coding_agent",
            "repo_analysis",
            "code_review",
            "pr_description",
            "bug_fix",
            "unit_test",
        ):
            assert pm.get(name) is not None

    def test_register_new(self):
        pm = PromptManager()
        pm.register(PromptTemplate("custom", "Hi {who}!"))
        assert pm.get("custom") is not None

    def test_register_duplicate_raises(self):
        pm = PromptManager()
        pm.register(PromptTemplate("my_tmpl", "x"))
        with pytest.raises(ValueError, match="already registered"):
            pm.register(PromptTemplate("my_tmpl", "y"))

    def test_register_overwrite(self):
        pm = PromptManager()
        pm.register(PromptTemplate("dup", "original"))
        pm.register(PromptTemplate("dup", "replaced"), overwrite=True)
        assert pm.render("dup") == "replaced"

    def test_render_builtin(self):
        pm = PromptManager()
        result = pm.render("unit_test", file_path="x.py", framework="pytest", code="pass")
        assert "pytest" in result

    def test_render_unknown_raises(self):
        pm = PromptManager()
        with pytest.raises(KeyError):
            pm.render("does_not_exist")

    def test_template_names_sorted(self):
        pm = PromptManager()
        names = pm.template_names
        assert names == sorted(names)
