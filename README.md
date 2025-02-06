# kb

A personal knowledge base

## Getting Started

This project uses `uv`:

- `uv venv`
- `source .venv/bin/activate`
- `uv pip install -e .` - make an editable kb project
    - Note: pytorch 2.2 is used to support macOS x86_64
- `uvicorn kb.api:app`
    - Use `uvicorn kb.api:app --reload` for development
- Generate API bindings for the cli: `openapi-python-client generate --url http://localhost:8000/openapi.json --output-path src/kb_client`
- Run the cli: `python -m kb.cli`

### Configuration

Make sure the following environment variables are specified in `config.toml`

### Anthropic

```
[obsidian]
vault="/path/to/vault"

[llm]
type="anthropic"
api_key="secret"
```

### Ollama

See [docs](./docs) for instructions on how to set up ollama.

```
[obsidian]
vault="/path/to/vault"

[llm]
llm_type="ollama"
model="deepseek-r1:32b"
api_key="secret"
base_url="https://ollama.example.com/"
```

## Tests

Run tests using `pytest tests/test_api.py`.

Tests will use the local chroma database, but will not call the LLM.
