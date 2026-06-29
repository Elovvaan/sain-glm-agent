# SAIN GLM Agent

SAIN GLM Agent is a production-ready Python framework for building repository-aware coding agents powered by GLM-compatible chat models. It combines clean architecture, provider abstraction, GitHub repository analysis, memory backends, prompt management, structured logging, and a Typer CLI.

## Features

- GLM provider integration over `httpx`
- Clean architecture split into application, domain, infrastructure, interfaces, prompts, and tools
- GitHub REST adapter for repository metadata, file browsing, and file content retrieval
- Conversation memory with in-memory and file-backed implementations
- Prompt template manager for ask, analyze, plan, and generate workflows
- Safe tool execution engine with repository read tools
- Config management with `pydantic-settings`
- Rich CLI output with actionable error handling
- Test suite covering settings, providers, memory, prompts, agent orchestration, and CLI behavior

## Architecture

```text
+---------------------------+
|        CLI / Typer        |
|  ask · analyze · plan     |
+-------------+-------------+
              |
              v
+---------------------------+
|      CodingAgent          |
|  orchestration layer      |
+------+------+-------------+
       |      |
       |      +--------------------+
       |                           |
       v                           v
+-------------+             +-------------+
| PromptManager|             | MemoryStore |
+------+------+             +------+------+
       |                           |
       v                           v
+---------------------------+  +------------------+
| Provider Registry / GLM   |  | InMemory / File  |
+-------------+-------------+  +------------------+
              |
              v
+---------------------------+
| GitHub Repository Service |
| Repo tools + PR drafts    |
+---------------------------+
```

## Installation

### Standard install

```bash
python -m pip install .
```

### Development install

```bash
python -m pip install -e .[dev]
```

## Configuration

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Minimum required values for GLM-backed commands:

```env
GLM_API_KEY=your-real-key
GITHUB_TOKEN=optional-for-public-repos
ACTIVE_PROVIDER=glm
```

The framework loads configuration from environment variables and `.env` automatically.

## CLI Usage

Analyze a repository:

```bash
sain-agent analyze elovvaan/sain-glm-agent
```

Ask a coding question:

```bash
sain-agent ask elovvaan/sain-glm-agent "What does the architecture look like?"
```

Plan a change:

```bash
sain-agent plan elovvaan/sain-glm-agent "Add caching to repository lookups"
```

You can also run the CLI module directly:

```bash
PYTHONPATH=src python -m sain_glm_agent.interfaces.cli.main analyze owner/repo
```

## Development Workflow

1. Install dependencies with the development extras.
2. Configure `.env`.
3. Run tests, Ruff, and optional type checks.
4. Extend providers, prompts, tools, or agent workflows as needed.

Example commands:

```bash
python -m pytest tests/ -v
python -m ruff check src tests
python -m mypy src
```

## Testing

Run the full test suite:

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

## Package Layout

```text
src/sain_glm_agent/
├── application/      # Orchestration and workflows
├── domain/           # Models and core interfaces
├── infrastructure/   # Providers, GitHub, memory, config, logging
├── interfaces/       # CLI entrypoints
├── prompts/          # Prompt templates and message builders
└── tools/            # Safe repository tools
```

## Roadmap

- Add streaming provider support
- Add richer tool calling and structured outputs
- Add repository write operations with approval workflows
- Add first-class support for more LLM providers
- Add long-term vector memory and embeddings
- Add GitHub pull request creation and review automation

## License

MIT
