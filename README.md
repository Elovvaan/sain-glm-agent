# SAIN GLM Agent

SAIN GLM Agent is a production-ready Python framework for building repository-aware AI coding agents around **GLM-5.2**. It provides a clean architecture, a pluggable provider layer, secure environment-based configuration, repository analysis utilities, prompt management, memory, tool execution, and a CLI that is ready to grow into desktop or web interfaces.

## Features

- GLM-5.2 integration through a provider abstraction layer
- Provider registry designed for future OpenAI, Claude, Gemini, DeepSeek, Qwen, and local model support
- Repository assistant workflows for analysis, planning, code generation guidance, and pull request drafting
- Environment-variable driven configuration with secret redaction
- Modular architecture for providers, orchestration, repository analysis, prompts, tools, memory, logging, and configuration
- CLI entrypoint for local usage and automation
- Unit tests using Python's standard library
- Type hints, docstrings, and lint/type-checker configuration

## Installation

```bash
python -m pip install -e .
```

## Configuration

Set credentials and runtime settings with environment variables.

```bash
export SAIN_API_KEY="your-glm-api-key"
export SAIN_PROVIDER="glm"
export SAIN_MODEL="glm-5.2"
export SAIN_BASE_URL="https://open.bigmodel.cn/api/paas/v4"
export SAIN_LOG_LEVEL="INFO"
```

### Supported environment variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `SAIN_API_KEY` / `GLM_API_KEY` | API key for the active remote provider | _required for remote inference_ |
| `SAIN_PROVIDER` | Provider name in the registry | `glm` |
| `SAIN_MODEL` | Model name sent to the provider | `glm-5.2` |
| `SAIN_BASE_URL` | Base URL for the provider API | `https://open.bigmodel.cn/api/paas/v4` |
| `SAIN_TIMEOUT_SECONDS` | Network and git command timeout | `60` |
| `SAIN_MAX_TOKENS` | Response token limit | `3000` |
| `SAIN_TEMPERATURE` | Sampling temperature | `0.2` |
| `SAIN_LOG_LEVEL` | Python logging level | `INFO` |
| `SAIN_DATA_DIR` | Local runtime state directory | `~/.sain_glm_agent` |
| `SAIN_MEMORY_FILE` | Conversation history file override | `<data_dir>/memory.json` |
| `SAIN_MAX_CONTEXT_FILES` | Number of repository files added to prompts | `6` |
| `SAIN_MAX_FILE_BYTES` | Maximum bytes read per file excerpt | `16000` |
| `SAIN_ALLOWED_COMMANDS` | Comma-separated command allow-list for local tool execution | `git,python,pytest,ruff,mypy` |

## CLI Usage

```bash
sain-glm-agent analyze "Summarize the repository architecture" --repo /path/to/repo
sain-glm-agent plan "Describe how to add a pull request workflow" --repo /path/to/repo
sain-glm-agent generate "Propose code changes for adding a provider registry" --repo /path/to/repo
sain-glm-agent prepare-pr "Draft a PR description for the current branch" --repo /path/to/repo
sain-glm-agent config
```

Use `--json` to return machine-readable results.

## Architecture

The framework is organized into focused modules:

- `sain_glm_agent.config` — secure runtime configuration
- `sain_glm_agent.logging_utils` — logging setup
- `sain_glm_agent.providers` — model provider interfaces and GLM implementation
- `sain_glm_agent.agent` — orchestration for repository workflows
- `sain_glm_agent.repository` — repository scanning, file selection, and Git context
- `sain_glm_agent.prompts` — task-specific prompt rendering
- `sain_glm_agent.tools` — local command execution with allow-listing
- `sain_glm_agent.memory` — conversation persistence

Additional architectural notes are available in `/home/runner/work/sain-glm-agent/sain-glm-agent/docs/architecture.md`.

## Extending Providers

Add a new provider by implementing `ModelProvider` and registering it in the provider registry. The core assistant only depends on the provider interface, so new backends can be introduced without changing orchestration logic.

## Testing

```bash
python -m unittest discover -s tests -v
```

## Linting and Type Checking

Configuration for Ruff and mypy is included in `pyproject.toml`.

Typical commands:

```bash
ruff check src tests
mypy src
```

## Project Layout

```text
src/sain_glm_agent/
├── agent.py
├── cli.py
├── config.py
├── exceptions.py
├── logging_utils.py
├── memory.py
├── models.py
├── prompts.py
├── repository.py
├── tools.py
└── providers/
    ├── base.py
    ├── glm.py
    └── registry.py
tests/
docs/
```

## Production Notes

- Secrets are never stored in source control and are redacted in config output.
- Tool execution is restricted to explicitly allowed binaries.
- Repository reads are rooted to the target workspace to avoid path traversal.
- Conversation history is persisted locally for repeatable agent sessions.
