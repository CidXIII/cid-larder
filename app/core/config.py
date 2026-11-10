"""Configuration: users write data/config.json; env vars override for development."""

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CONFIG_PATH = DATA_DIR / "config.json"

PRESETS_PATH = Path(__file__).with_name("providers.json")
PRESETS: dict = json.loads(PRESETS_PATH.read_text(encoding="utf-8"))
# Keys starting with "_" are file metadata, not providers.
PROVIDERS: dict = {k: v for k, v in PRESETS.items() if not k.startswith("_")}


class ConfigError(RuntimeError):
    """A configured value is missing or inconsistent."""


@dataclass(frozen=True)
class Endpoint:
    provider: str
    base_url: str
    api_key: str
    model: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = ""
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""

    embedding_provider: str = ""
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = ""

    # Dev-only credentials. Read from .env; never written to config.json.
    bili_sessdata: str = ""
    bili_csrf: str = ""
    bili_uid: str = ""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: list[PydanticBaseSettingsSource] = [
            init_settings,
            env_settings,
            dotenv_settings,
        ]
        if CONFIG_PATH.exists():
            sources.append(JsonConfigSettingsSource(settings_cls, json_file=CONFIG_PATH))
        sources.append(file_secret_settings)
        return tuple(sources)

    def _resolve(
        self,
        kind: str,
        provider: str,
        api_key: str,
        base_override: str,
        model_override: str,
    ) -> Endpoint | None:
        if not provider and not api_key:
            return None
        if not provider:
            raise ConfigError(f"{kind}: provider is empty but a key was given")
        preset = PROVIDERS.get(provider)
        if preset is None:
            known = ", ".join(sorted(PROVIDERS))
            raise ConfigError(f"{kind}: unknown provider {provider!r}. Known: {known}")
        if not api_key:
            raise ConfigError(f"{kind}: provider {provider!r} has no api key")

        model = model_override or (preset.get(f"{kind}_model") or "")
        base_url = base_override or preset.get("base_url", "")
        if not model:
            raise ConfigError(f"{kind}: no model. Set it explicitly for {provider!r}")
        if not base_url:
            raise ConfigError(f"{kind}: no base_url for {provider!r}")
        return Endpoint(provider, base_url, api_key, model)

    @property
    def chat(self) -> Endpoint | None:
        return self._resolve(
            "chat", self.llm_provider, self.llm_api_key,
            self.llm_base_url, self.llm_model,
        )

    @property
    def embedding(self) -> Endpoint | None:
        """Falls back to the chat provider when it also serves embeddings."""
        if not self.embedding_provider and not self.embedding_api_key:
            chat = self.chat
            if chat and PROVIDERS.get(chat.provider, {}).get("embedding_model"):
                return Endpoint(
                    chat.provider, chat.base_url, chat.api_key,
                    PROVIDERS[chat.provider]["embedding_model"],
                )
            return None
        return self._resolve(
            "embedding", self.embedding_provider, self.embedding_api_key,
            self.embedding_base_url, self.embedding_model,
        )

    @property
    def ready(self) -> bool:
        """Status check: never raises, even when a provider is half configured."""
        try:
            return self.chat is not None and self.embedding is not None
        except ConfigError:
            return False


settings = Settings()


def reload_settings() -> Settings:
    """Re-read config.json and env. Call after the wizard saves a config."""
    global settings
    settings = Settings()
    return settings
