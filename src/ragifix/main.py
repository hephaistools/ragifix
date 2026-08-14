"""Point d'entrée de ragifix.

IMPORTANT : ne jamais lancer plusieurs instances de ce process (ni
plusieurs workers uvicorn/gunicorn) pointant vers le même fichier Milvus
Lite (vectorstore.milvus.mode=lite) — ce backend ne supporte qu'un seul
process ouvrant le fichier à la fois. Pour monter en charge, passer
vectorstore.milvus.mode=server (Milvus distant, qui supporte nativement
plusieurs clients concurrents) plutôt que de dupliquer ce process.
"""

from __future__ import annotations

import argparse
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .auth import TokenAuth
from .config import AppConfig, ConfigError, load_config
from .embedding.base import EmbeddingBackend
from .embedding.fastembed_backend import FastEmbedBackend
from .embedding.openai_compat_backend import OpenAICompatibleBackend
from .processing import build_chunker
from .registry import build_registry
from .routes import build_router
from .service import RagifixService
from .vectorstore.base import VectorStore
from .vectorstore.milvus_backend import MilvusVectorStore

logger = logging.getLogger(__name__)


def build_embedding_backend(config: AppConfig) -> EmbeddingBackend:
    if config.embedding.backend == "fastembed":
        return FastEmbedBackend(model_name=config.embedding.fastembed.model)
    if config.embedding.backend == "openai_compatible":
        oc = config.embedding.openai_compatible
        return OpenAICompatibleBackend(base_url=oc.base_url, api_key=oc.api_key, model=oc.model)
    raise ValueError(f"Backend d'embedding inconnu: {config.embedding.backend}")


def build_vector_store(config: AppConfig, dimension: int) -> VectorStore:
    vs = config.vectorstore
    if vs.backend == "milvus":
        return MilvusVectorStore(
            collection_name=vs.milvus.collection_name,
            dimension=dimension,
            mode=vs.milvus.mode,
            lite_path=vs.milvus.lite_path,
            host=vs.milvus.host,
            port=vs.milvus.port,
        )
    raise ValueError(f"Backend de base vectorielle inconnu: {vs.backend}")


def create_app(config_path: str) -> FastAPI:
    config = load_config(config_path)

    logging.basicConfig(
        level=config.logging.level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    embedding_backend = build_embedding_backend(config)
    vector_store = build_vector_store(config, dimension=embedding_backend.dimension)
    registry = build_registry(config.registry.backend, config.registry.sqlite.path)
    chunker = build_chunker(config.chunking.strategy, config.chunking.chunk_size, config.chunking.chunk_overlap)

    service = RagifixService(embedding_backend, vector_store, registry, chunker)
    auth = TokenAuth(expected_token=config.api.auth_token)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        service.shutdown()

    app = FastAPI(
        title="ragifix",
        description="Service RAG auto-porté : ingestion, suppression et interrogation via API (127.0.0.1 uniquement).",
        lifespan=lifespan,
    )
    app.include_router(build_router(service, auth, config.api.default_top_k, config.api.max_document_size_mb))
    app.state.config = config
    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="ragifix — service RAG auto-porté (ingestion + interrogation)")
    parser.add_argument("--config", required=True, help="Chemin vers le fichier config.yaml")
    args = parser.parse_args()

    try:
        app = create_app(args.config)
    except ConfigError as exc:
        print(f"Erreur de configuration: {exc}", file=sys.stderr)
        sys.exit(1)

    config: AppConfig = app.state.config
    assert config.api.host in ("127.0.0.1", "localhost", "::1")

    uvicorn.run(app, host=config.api.host, port=config.api.port, log_level=config.logging.level.lower())


if __name__ == "__main__":
    main()
