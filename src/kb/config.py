from pathlib import Path
from enum import Enum
import tomli
from pydantic import BaseModel, SecretStr


class LLMType(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    LOCAL = "local"


class LLMConfig(BaseModel):
    llm_type: LLMType
    api_key: SecretStr


class ObsidianConfig(BaseModel):
    vault: Path


class Config(BaseModel):
    llm: LLMConfig
    obsidian: ObsidianConfig

    @classmethod
    def from_toml(cls, path: str | Path) -> "Config":
        with open(path, "rb") as f:
            data = tomli.load(f)
        return cls.model_validate(data)
