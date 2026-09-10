"""Tests des modèles de configuration."""

from __future__ import annotations

import pytest

from ragifix.config import (
    ApiConfig,
    AppConfig,
    ChunkingConfig,
    ConfigError,
    EmbeddingConfig,
    OpenAICompatConfig,
    ParsingConfig,
    load_config,
)


# -- Chunking -------------------------------------------------------------

def test_chunking_valid():
    ChunkingConfig(chunk_size=512, chunk_overlap=64)


def test_chunking_overlap_equal_raises():
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=10, chunk_overlap=10)


def test_chunking_overlap_greater_raises():
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=10, chunk_overlap=20)


def test_chunking_size_zero_raises():
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=0)


# -- Choix du backend de parsing ------------------------------------------

def test_parsing_backend_defaults_to_docling():
    cfg = ParsingConfig()
    assert cfg.backend == "docling"


def test_parsing_backend_markitdown_defaults():
    cfg = ParsingConfig.model_validate({"backend": "markitdown"})
    assert cfg.markitdown.enable_plugins is False


def test_parsing_backend_unknown_raises():
    with pytest.raises(ValueError):
        ParsingConfig.model_validate({"backend": "unknown"})


# -- Api host -------------------------------------------------------------

def test_api_host_local_ok():
    ApiConfig(host="127.0.0.1", auth_token_env="RAGIFIX_API_TOKEN")


def test_api_host_non_local_raises():
    with pytest.raises(ValueError):
        ApiConfig(host="0.0.0.0", auth_token_env="RAGIFIX_API_TOKEN")


def test_api_host_external_ip_raises():
    with pytest.raises(ValueError):
        ApiConfig(host="192.168.1.10", auth_token_env="RAGIFIX_API_TOKEN")


# -- Secrets via env ------------------------------------------------------

def test_auth_token_missing(monkeypatch):
    monkeypatch.delenv("RAGIFIX_API_TOKEN", raising=False)
    cfg = ApiConfig(host="127.0.0.1", auth_token_env="RAGIFIX_API_TOKEN")
    with pytest.raises(ConfigError):
        _ = cfg.auth_token


def test_auth_token_present(monkeypatch):
    monkeypatch.setenv("RAGIFIX_API_TOKEN", "secret")
    cfg = ApiConfig(host="127.0.0.1", auth_token_env="RAGIFIX_API_TOKEN")
    assert cfg.auth_token == "secret"


def test_openai_api_key_missing(monkeypatch):
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    cfg = OpenAICompatConfig(base_url="http://x", api_key_env="EMBEDDING_API_KEY", model="m")
    with pytest.raises(ConfigError):
        _ = cfg.api_key


def test_openai_api_key_present(monkeypatch):
    monkeypatch.setenv("EMBEDDING_API_KEY", "key")
    cfg = OpenAICompatConfig(base_url="http://x", api_key_env="EMBEDDING_API_KEY", model="m")
    assert cfg.api_key == "key"


# -- Choix du backend d'embedding ----------------------------------------

def test_fastembed_backend_sets_default():
    cfg = EmbeddingConfig.model_validate({"backend": "fastembed"})
    assert cfg.fastembed is not None
    assert cfg.fastembed.model == "BAAI/bge-small-en-v1.5"


def test_openai_compatible_requires_config():
    with pytest.raises(ValueError):
        EmbeddingConfig.model_validate({"backend": "openai_compatible"})


def test_openai_compatible_ok():
    cfg = EmbeddingConfig.model_validate(
        {
            "backend": "openai_compatible",
            "openai_compatible": {"base_url": "http://x", "api_key_env": "K", "model": "m"},
        }
    )
    assert cfg.openai_compatible is not None


# -- AppConfig + load_config ----------------------------------------------

def test_app_config_valid():
    cfg = AppConfig.model_validate(
        {"embedding": {"backend": "fastembed"}, "api": {"host": "127.0.0.1", "auth_token_env": "X"}}
    )
    assert cfg.embedding.backend == "fastembed"
    assert cfg.api.host == "127.0.0.1"
    assert cfg.parsing.backend == "docling"


def test_load_config_missing(tmp_path):
    with pytest.raises(ConfigError):
        load_config(str(tmp_path / "inexistant.yaml"))


def test_load_config_empty(tmp_path):
    f = tmp_path / "empty.yaml"
    f.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(str(f))


def test_load_config_invalid(tmp_path):
    f = tmp_path / "invalid.yaml"
    f.write_text("api:\n  host: 0.0.0.0\n  auth_token_env: X\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(str(f))
