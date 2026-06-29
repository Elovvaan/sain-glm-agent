# SAIN GLM Agent

**SAIN Enterprise AI coding agent using GLM for intelligent software engineering, automation, and repository management.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)

---

## Overview

SAIN GLM Agent is a production-ready, modular AI coding-agent framework built around [ZhipuAI GLM](https://open.bigmodel.cn/) models. It implements a **ReAct** (Reason + Act) reasoning loop and provides a clean abstraction layer so you can swap in any supported model provider—OpenAI, Claude, Gemini, DeepSeek, Qwen, or a local model—without touching the agent core.

Out of the box it ships with:
- A **GitHub repository assistant** that can read repos, plan changes, generate code, and prepare pull requests.
- A **provider architecture** ready for GLM (primary), OpenAI, Claude, Gemini, DeepSeek, Qwen, and local models.
- A **sliding-window conversation memory** with JSON persistence.
- A **prompt-template manager** with six built-in coding-agent templates.
- A **tool registry and executor** for extending the agent with custom actuators.
- A **structured logging system** supporting rich, plain, and JSON formats.
- A **Click-based CLI** with `chat`, `run`, `repo analyse`, `repo pr`, and `info` commands.
- **90 unit tests** covering all modules.

---

## Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [CLI](#cli)
  - [Python API](#python-api)
- [Modules](#modules)
- [Extending the Framework](#extending-the-framework)
- [Development](#development)
- [Roadmap](#roadmap)
- [License](#license)

---

## Architecture

```
sain-glm-agent/
├── sain_glm_agent/
│   ├── config/           # Pydantic-settings configuration
│   ├── logging_/         # Structured logging (rich / plain / JSON)
│   ├── providers/        # Model provider abstraction
│   │   ├── base.py       #   BaseProvider, Message, ModelResponse
│   │   ├── glm.py        #   ZhipuAI GLM implementation
│   │   ├── stubs.py      #   Stub providers (OpenAI, Claude, Gemini, …)
│   │   └── factory.py    #   ProviderFactory
│   ├── memory/           # Sliding-window conversation history
│   ├── prompts/          # Prompt-template manager
│   ├── tools/            # Tool registry & executor
│   ├── repository/       # GitHub client & repository analyser
│   ├── agent/            # ReAct orchestrator & state machine
│   └── cli.py            # Click CLI entry point
├── tests/                # 90 unit tests (pytest)
├── pyproject.toml
├── .env.example
└── README.md
```

**Key design principles:**
- **Clean separation of concerns** — each module has one responsibility.
- **Dependency inversion** — the agent core depends on `BaseProvider`, not on any specific model SDK.
- **Immutable configuration** — settings are loaded once from the environment and cached.
- **Fail-fast** — missing API keys and unknown providers raise `ValueError` at startup, not mid-run.

---

## Installation

### Prerequisites

- Python 3.10 or newer
- A [ZhipuAI API key](https://open.bigmodel.cn/) (or another supported provider key)
- A [GitHub personal-access token](https://github.com/settings/tokens) (for repository operations)

### From source

```bash
git clone https://github.com/Elovvaan/sain-glm-agent.git
cd sain-glm-agent

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install the package (production)
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

---

## Configuration

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

All settings can also be provided as plain environment variables. The `SAIN_` prefix is used for project-specific options; third-party API keys keep their conventional names.

| Environment variable | Default | Description |
|---|---|---|
| `SAIN_PROVIDER` | `glm` | Active model provider (`glm`, `openai`, `claude`, `gemini`, `deepseek`, `qwen`, `local`) |
| `ZHIPUAI_API_KEY` | — | ZhipuAI / GLM API key |
| `SAIN_GLM_MODEL` | `glm-4-flash` | GLM model name |
| `OPENAI_API_KEY` | — | OpenAI API key (when provider = openai) |
| `ANTHROPIC_API_KEY` | — | Anthropic API key (when provider = claude) |
| `GOOGLE_API_KEY` | — | Google API key (when provider = gemini) |
| `DEEPSEEK_API_KEY` | — | DeepSeek API key |
| `DASHSCOPE_API_KEY` | — | Alibaba DashScope API key (Qwen) |
| `GITHUB_TOKEN` | — | GitHub personal-access token |
| `SAIN_MAX_TOKENS` | `4096` | Max completion tokens per request |
| `SAIN_TEMPERATURE` | `0.1` | Sampling temperature (0 – 2) |
| `SAIN_MAX_ITERATIONS` | `10` | Max ReAct loop iterations |
| `SAIN_MEMORY_MAX_MESSAGES` | `50` | Sliding-window history size |
| `SAIN_LOG_LEVEL` | `INFO` | Log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `SAIN_LOG_FORMAT` | `rich` | Log output format (`rich`, `plain`, `json`) |
| `SAIN_LOG_FILE` | — | Optional path to a log file |

---

## Usage

### CLI

After installation the `sain-agent` command is available in your path.

**Interactive chat session**
```bash
sain-agent chat
sain-agent chat --stream   # stream tokens as they arrive
```

**Single-task execution**
```bash
sain-agent run "Explain the difference between async and threading in Python"
sain-agent run "Write a merge-sort implementation in Go" --verbose
```

**Repository analysis**
```bash
sain-agent repo analyse owner/repo
sain-agent repo analyse owner/repo --objective "Add pagination to the REST API"
```

**Open a pull request**
```bash
sain-agent repo pr owner/repo \
  --task "Fix the off-by-one error in the pagination helper" \
  --branch "fix/pagination-off-by-one"

# Dry-run (plan only — nothing is pushed)
sain-agent repo pr owner/repo --task "Refactor auth module" --dry-run
```

**Print configuration**
```bash
sain-agent info
```

### Python API

```python
from sain_glm_agent.providers.glm import GLMProvider
from sain_glm_agent.agent import AgentOrchestrator
from sain_glm_agent.tools.registry import ToolRegistry, Tool

# 1. Create a provider
provider = GLMProvider(api_key="your-key", model="glm-4-flash")

# 2. Register custom tools
registry = ToolRegistry()
registry.register(
    Tool(
        name="read_file",
        description="Read a local file.",
        parameters={"path": {"type": "string", "description": "File path"}},
        fn=lambda path: open(path).read(),
    )
)

# 3. Create and run the agent
agent = AgentOrchestrator(
    provider=provider,
    tool_registry=registry,
    max_iterations=8,
)
state = agent.run("Summarise the content of README.md")
print(state.final_answer)
```

**Conversation memory**
```python
from sain_glm_agent.memory import ConversationMemory
from pathlib import Path

mem = ConversationMemory(max_messages=20)
mem.set_system("You are a senior Python developer.")
mem.add_user("What is a metaclass?")
mem.add_assistant("A metaclass is the class of a class …")

# Persist and restore
mem.save(Path("session.json"))
restored = ConversationMemory.load(Path("session.json"))
```

**Prompt templates**
```python
from sain_glm_agent.prompts import PromptManager, PromptTemplate

pm = PromptManager()
# Use a built-in template
text = pm.render("unit_test", file_path="utils.py", framework="pytest", code="def add(a, b): ...")

# Register your own
pm.register(PromptTemplate("deploy", "Deploy {service} to {env}."))
pm.render("deploy", service="api", env="production")
```

---

## Modules

| Module | Responsibility |
|---|---|
| `config` | Pydantic-settings singleton; all env vars validated at startup |
| `logging_` | Root logger setup; rich / plain / JSON output; optional file sink |
| `providers` | `BaseProvider` ABC + `GLMProvider` + stubs + `ProviderFactory` |
| `memory` | `ConversationMemory` — sliding window, JSON serialisation, file I/O |
| `prompts` | `PromptTemplate` + `PromptManager` registry with 6 built-in templates |
| `tools` | `ToolRegistry` + `ToolExecutor` (sync & async) |
| `repository` | `GitHubClient` (PyGithub wrapper) + `RepositoryAnalyzer` |
| `agent` | `AgentState` (lifecycle / step tracking) + `AgentOrchestrator` (ReAct loop) |
| `cli` | Click-based CLI: `chat`, `run`, `repo analyse`, `repo pr`, `info` |

---

## Extending the Framework

### Add a new model provider

1. Create `sain_glm_agent/providers/my_provider.py` and subclass `BaseProvider`.
2. Implement `chat()` and `stream_chat()`.
3. Add a `ProviderName.MY_PROVIDER` entry to `config/settings.py`.
4. Add a factory branch to `providers/factory.py`.

```python
from sain_glm_agent.providers.base import BaseProvider, Message, ModelResponse

class MyProvider(BaseProvider):
    PROVIDER_NAME = "myprovider"

    def chat(self, messages, **kwargs):
        # call your API here
        return ModelResponse(content="…", model=self.model, provider=self.PROVIDER_NAME)

    def stream_chat(self, messages, **kwargs):
        yield "token1"
        yield "token2"
```

### Add a custom tool

```python
from sain_glm_agent.tools.registry import ToolRegistry

registry = ToolRegistry()

@registry.register_fn(
    name="web_search",
    description="Search the web for up-to-date information.",
    parameters={"query": {"type": "string", "description": "Search query"}},
)
def web_search(query: str) -> str:
    import requests
    r = requests.get(f"https://search-api.example.com/?q={query}")
    return r.text
```

---

## Development

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=sain_glm_agent --cov-report=term-missing

# Lint
ruff check .
ruff check . --fix

# Type check
mypy sain_glm_agent/
```

---

## Roadmap

- [ ] Full OpenAI / Claude / Gemini provider implementations
- [ ] Local model support via Ollama / vLLM
- [ ] Vector-store memory (FAISS / ChromaDB)
- [ ] Multi-agent collaboration support
- [ ] Desktop GUI (Tkinter / PyQt)
- [ ] Web interface (FastAPI + React)
- [ ] GitHub Actions integration
- [ ] Docker image & Helm chart

---

## License

MIT — see [LICENSE](LICENSE).

