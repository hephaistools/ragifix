"""Modèle de configuration de ragifix.

ragifix ne connaît aucune notion de "source" : il ne fait qu'exposer une
API pour ajouter/supprimer/interroger des documents. Qui lui envoie ces
documents (ragifix-collector, n8n, un script maison...) est hors de son
périmètre.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ConfigError(Exception):
    """Erreur de configuration (fichier invalide ou secret manquant)."""


# ------------------------------------------------------------------------- #
# Chunking
# ------------------------------------------------------------------------- #

class ChunkingConfig(BaseModel):
    strategy: Literal["token"] = "token"
    chunk_size: int = 512
    chunk_overlap: int = 64

    @model_validator(mode="after")
    def _check_overlap(self) -> "ChunkingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunking.chunk_overlap doit être strictement inférieur à chunk_size")
        if self.chunk_size <= 0:
            raise ValueError("chunking.chunk_size doit être positif")
        return self


# ------------------------------------------------------------------------- #
# Embedding
# ------------------------------------------------------------------------- #

class FastEmbedConfig(BaseModel):
    model: str = "BAAI/bge-small-en-v1.5"


class OpenAICompatConfig(BaseModel):
    base_url: str
    api_key_env: str
    model: str

    @property
    def api_key(self) -> str:
        value = os.environ.get(self.api_key_env)
        if not value:
            raise ConfigError(f"Variable d'environnement manquante: {self.api_key_env}")
        return value


class EmbeddingConfig(BaseModel):
    backend: Literal["fastembed", "openai_compatible"] = "openai_compatible"
    fastembed: FastEmbedConfig | None = None
    openai_compatible: OpenAICompatConfig | None = None

    @model_validator(mode="after")
    def _check_backend_config(self) -> "EmbeddingConfig":
        if self.backend == "fastembed" and self.fastembed is None:
            self.fastembed = FastEmbedConfig()
        if self.backend == "openai_compatible" and self.openai_compatible is None:
            raise ValueError("embedding.openai_compatible est requis quand embedding.backend=openai_compatible")
        return self


# ------------------------------------------------------------------------- #
# Base vectorielle
# ------------------------------------------------------------------------- #

class MilvusConfig(BaseModel):
    mode: Literal["lite", "server"] = "lite"
    lite_path: str = "/var/lib/ragifix/milvus_lite.db"
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "ragifix_documents"


class VectorStoreConfig(BaseModel):
    backend: Literal["milvus"] = "milvus"
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)


# ------------------------------------------------------------------------- #
# Registre des documents indexés
# ------------------------------------------------------------------------- #

class SqliteRegistryConfig(BaseModel):
    path: str = "/var/lib/ragifix/registry.db"


class RegistryConfig(BaseModel):
    backend: Literal["sqlite"] = "sqlite"
    sqlite: SqliteRegistryConfig = Field(default_factory=SqliteRegistryConfig)


# ------------------------------------------------------------------------- #
# API
# ------------------------------------------------------------------------- #

class ApiConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8421
    auth_token_env: str
    default_top_k: int = 5
    max_document_size_mb: int = 50

    @field_validator("host")
    @classmethod
    def _check_local_only(cls, v: str) -> str:
        if v not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError(
                "api.host doit rester strictement local (127.0.0.1) — "
                "l'API ne doit jamais être exposée sur le réseau"
            )
        return v

    @property
    def auth_token(self) -> str:
        value = os.environ.get(self.auth_token_env)
        if not value:
            raise ConfigError(f"Variable d'environnement manquante: {self.auth_token_env}")
        return value


# ------------------------------------------------------------------------- #
# Logging
# ------------------------------------------------------------------------- #

class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


# ------------------------------------------------------------------------- #
# Racine
# ------------------------------------------------------------------------- #

class AppConfig(BaseModel):
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
    api: ApiConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Fichier de configuration introuvable: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ConfigError(f"Fichier de configuration vide: {path}")
    try:
        return AppConfig.model_validate(raw)
    except Exception as exc:
        raise ConfigError(f"Configuration invalide ({path}): {exc}") from exc
