# kb

A personal knowledge base

## Getting Started

This project uses `uv`:

- `source .venv/bin/activate`
- `uv pip install -e .` - make an editable kb project
    - Note: pytorch 2.2 is used to support macOS x86_64
- `uvicorn kb.api:app --reload`

### .env

Make sure the following environment variables are specified

```
ANTHROPIC_API_KEY=secret
OBSIDIAN_VAULT_PATH=/path/to/vault
```
