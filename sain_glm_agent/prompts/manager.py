"""Prompt-template management for SAIN GLM Agent.

Templates use Python's built-in ``str.format_map`` so they work without any
additional dependency.  Variables are referenced as ``{variable_name}`` inside
the template string.

Built-in templates:
    * ``system_coding_agent`` — system prompt for the core coding agent.
    * ``repo_analysis``       — prompt requesting repository analysis.
    * ``code_review``         — prompt requesting a code-review.
    * ``pr_description``      — prompt generating a pull-request body.
    * ``bug_fix``             — prompt requesting a bug-fix plan.
    * ``unit_test``           — prompt requesting unit tests.

Usage::

    from sain_glm_agent.prompts import PromptManager

    pm = PromptManager()
    text = pm.render("repo_analysis", repo_url="https://github.com/org/repo",
                     objective="Add pagination to the API")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in templates
# ---------------------------------------------------------------------------

_BUILTIN_TEMPLATES: dict[str, str] = {
    "system_coding_agent": """\
You are SAIN GLM Agent, an expert AI software engineer with deep knowledge of \
software architecture, best practices, and modern development workflows.

Your capabilities include:
- Analysing and understanding codebases
- Planning and implementing code changes
- Writing clean, well-tested, production-quality code
- Creating pull requests and documentation
- Debugging and fixing issues
- Reviewing code for quality and security

Always respond with structured, actionable output. When writing code, follow \
the project's existing style. Prefer incremental, reversible changes. \
Explain your reasoning clearly.

Current date: {current_date}
""",
    "repo_analysis": """\
Analyse the following GitHub repository and provide a structured report.

Repository URL: {repo_url}
Objective: {objective}

Please provide:
1. **Repository overview** — purpose, main language, key dependencies.
2. **Architecture summary** — modules, layers, main entry points.
3. **Relevant files** — list files most relevant to the objective.
4. **Change plan** — step-by-step plan to achieve the objective.
5. **Risks & considerations** — potential pitfalls or breaking changes.
""",
    "code_review": """\
Review the following code changes and provide actionable feedback.

Repository: {repo_url}
Pull request / branch: {branch}

Diff:
```
{diff}
```

Evaluate:
1. **Correctness** — does it do what was intended?
2. **Security** — any vulnerabilities or unsafe patterns?
3. **Performance** — obvious bottlenecks?
4. **Maintainability** — readability, naming, documentation.
5. **Test coverage** — are edge cases tested?

Provide specific line-level comments where appropriate.
""",
    "pr_description": """\
Generate a professional pull-request description for the following changes.

Repository: {repo_url}
Branch: {branch}
Summary of changes: {summary}

The PR description should include:
- A concise title (one line)
- **What** was changed and **why**
- **How** to test the changes
- Any **breaking changes** or migration notes
- References to related issues (if provided)
""",
    "bug_fix": """\
Diagnose and fix the following bug.

Repository: {repo_url}
Issue description: {issue_description}

Affected file(s):
```
{file_contents}
```

Please:
1. Identify the root cause.
2. Propose a minimal, targeted fix.
3. Write the corrected code.
4. Explain the fix and why it resolves the issue.
5. Suggest a regression test.
""",
    "unit_test": """\
Generate comprehensive unit tests for the following code.

File: {file_path}
Testing framework: {framework}

Code:
```python
{code}
```

Requirements:
- Cover all public methods and functions.
- Include happy-path, edge-case, and error-case tests.
- Use mocks/stubs where external dependencies are involved.
- Follow {framework} conventions and best practices.
""",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PromptTemplate:
    """A named prompt template with optional metadata.

    Attributes:
        name: Unique identifier.
        template: Template string using ``{variable}`` placeholders.
        description: Human-readable description of the template's purpose.
        variables: Names of all required substitution variables.
    """

    name: str
    template: str
    description: str = ""
    variables: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.variables:
            self.variables = self._extract_variables()

    def _extract_variables(self) -> list[str]:
        """Return the list of ``{variable}`` names found in the template."""
        return list({m.group(1) for m in re.finditer(r"\{(\w+)\}", self.template)})

    def render(self, **kwargs: str) -> str:
        """Substitute variables and return the rendered prompt.

        Args:
            **kwargs: Variable values; unknown keys are silently ignored.

        Returns:
            The fully rendered prompt string.

        Raises:
            KeyError: If a required variable is missing from *kwargs*.
        """
        try:
            return self.template.format_map(kwargs)
        except KeyError as exc:
            raise KeyError(
                f"Missing variable {exc} in template '{self.name}'. "
                f"Required: {self.variables}"
            ) from exc


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class PromptManager:
    """Registry and renderer for named prompt templates.

    On construction the built-in templates are registered automatically.
    Custom templates can be added with :meth:`register`.

    Usage::

        pm = PromptManager()
        pm.register(PromptTemplate("custom", "Hello {name}!"))
        text = pm.render("custom", name="World")
    """

    def __init__(self) -> None:
        self._templates: dict[str, PromptTemplate] = {}
        self._register_builtins()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, template: PromptTemplate, *, overwrite: bool = False) -> None:
        """Register a new prompt template.

        Args:
            template: The :class:`PromptTemplate` to add.
            overwrite: If ``False`` (default), raise on duplicate name.

        Raises:
            ValueError: If the name is already registered and *overwrite* is
                ``False``.
        """
        if template.name in self._templates and not overwrite:
            raise ValueError(
                f"Template '{template.name}' is already registered. "
                "Pass overwrite=True to replace it."
            )
        self._templates[template.name] = template
        logger.debug("Registered prompt template: %s", template.name)

    def get(self, name: str) -> PromptTemplate | None:
        """Return a template by name, or ``None`` if not found."""
        return self._templates.get(name)

    def render(self, name: str, **kwargs: str) -> str:
        """Look up a template and render it with the given variables.

        Args:
            name: Registered template name.
            **kwargs: Variable substitutions.

        Returns:
            Rendered prompt string.

        Raises:
            KeyError: If *name* is not registered.
        """
        tmpl = self._templates.get(name)
        if tmpl is None:
            raise KeyError(f"No prompt template named '{name}' is registered.")
        return tmpl.render(**kwargs)

    @property
    def template_names(self) -> list[str]:
        """Sorted list of all registered template names."""
        return sorted(self._templates)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _register_builtins(self) -> None:
        for name, tpl_str in _BUILTIN_TEMPLATES.items():
            self._templates[name] = PromptTemplate(name=name, template=tpl_str)
