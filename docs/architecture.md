# SAIN GLM Agent Architecture

SAIN GLM Agent follows a clean, modular architecture designed for extension.

## Layers

1. **Configuration and Logging**
   - `sain_glm_agent.config` loads environment-based settings.
   - `sain_glm_agent.logging_utils` centralizes runtime logging.
2. **Core Models and Exceptions**
   - `sain_glm_agent.models` contains provider-agnostic request, response, and orchestration models.
   - `sain_glm_agent.exceptions` defines bounded failure types.
3. **Infrastructure Adapters**
   - `sain_glm_agent.providers` contains provider implementations and a registry.
   - `sain_glm_agent.repository` handles repository reading and Git metadata collection.
   - `sain_glm_agent.tools` executes local commands with an allow-list.
   - `sain_glm_agent.memory` persists conversation history.
4. **Application Layer**
   - `sain_glm_agent.prompts` renders task-specific prompts.
   - `sain_glm_agent.agent` orchestrates repository analysis, prompt building, provider calls, and memory updates.
5. **Interface Layer**
   - `sain_glm_agent.cli` exposes the framework through a command-line interface and is designed to be replaced or supplemented by desktop/web frontends later.

## Extending Providers

Add a provider by implementing `ModelProvider.generate()` and registering it in `build_default_registry()` or a custom registry. The orchestration layer only depends on `ModelProvider`, so new backends do not require changes in `RepositoryAssistant`.

## Security Defaults

- API secrets are read only from environment variables.
- Tool execution is restricted to an allow-list.
- Repository reads are rooted to the configured workspace.
- Conversation memory is stored locally under the configured data directory.
