"""Orchestration métier de ragifix.

Toute la logique RAG vit ici : chunker un document, l'embedder, l'écrire
dans la base vectorielle, répondre à une requête. C'est l'unique point
d'entrée utilisé par les routes HTTP (ragifix.routes) — elles ne portent
aucune logique métier propre.

Sérialisation : toutes les opérations touchant l'embedding, la base
vectorielle et le registre sont exécutées sur un unique thread dédié
(ThreadPoolExecutor à un seul worker). C'est ce qui garantit qu'un backend
Milvus Lite (mono-écrivain par nature) n'est jamais sollicité par deux
appels concurrents au sein du même process, même si plusieurs requêtes HTTP
arrivent en parallèle. ragifix reste par ailleurs le SEUL process censé
ouvrir la base vectorielle (voir README) : cette sérialisation est une
défense en profondeur, pas la protection principale.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from concurrent.futures import ThreadPoolExecutor

from .embedding.base import EmbeddingBackend
from .parsing.base import DocumentParserBackend
from .processing import parse_document
from .registry import DocumentRecord, DocumentRegistry
from .vectorstore.base import SearchResult, VectorChunk, VectorStore
from .vectorstore.base import make_chunk_id  # noqa: F401  (ré-export, voir vectorstore.base)

logger = logging.getLogger(__name__)


class EmptyDocumentError(Exception):
    """Le document ne produit aucun chunk exploitable après parsing."""


class RagifixService:
    def __init__(
        self,
        embedding_backend: EmbeddingBackend,
        vector_store: VectorStore,
        registry: DocumentRegistry,
        chunker,
        parser: DocumentParserBackend,
    ):
        self._embedding = embedding_backend
        self._vector_store = vector_store
        self._registry = registry
        self._chunker = chunker
        self._parser = parser
        # Un seul worker : sérialise strictement tous les accès Milvus/embedding.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ragifix-worker")

    async def _run(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn, *args)

    # -- Ingestion ------------------------------------------------------

    async def ingest_document(self, doc_id: str, content: bytes, extension: str, metadata: dict) -> DocumentRecord:
        return await self._run(self._ingest_document_sync, doc_id, content, extension, metadata)

    def _ingest_document_sync(self, doc_id: str, content: bytes, extension: str, metadata: dict) -> DocumentRecord:
        text = parse_document(content, extension, filename=doc_id, parser=self._parser)
        text_chunks = self._chunker.chunk(text)

        if not text_chunks:
            self._vector_store.delete_document(doc_id)
            self._registry.delete(doc_id)
            raise EmptyDocumentError(f"Document '{doc_id}' vide après parsing/chunking")

        vectors = self._embedding.embed_documents(text_chunks)

        chunks = [
            VectorChunk(
                chunk_id=make_chunk_id(doc_id, i),
                doc_id=doc_id,
                text=text_chunk,
                vector=vector,
                metadata=metadata,
            )
            for i, (text_chunk, vector) in enumerate(zip(text_chunks, vectors))
        ]
        new_chunk_ids = [c.chunk_id for c in chunks]

        self._vector_store.upsert(chunks)
        self._vector_store.delete_by_doc_id(doc_id, keep_chunk_ids=new_chunk_ids)

        record = self._registry.upsert(doc_id, chunk_ids=new_chunk_ids)
        return dataclasses.replace(record, metadata=metadata)

    # -- Suppression ------------------------------------------------------

    async def delete_document(self, doc_id: str) -> bool:
        return await self._run(self._delete_document_sync, doc_id)

    def _delete_document_sync(self, doc_id: str) -> bool:
        self._vector_store.delete_document(doc_id)
        return self._registry.delete(doc_id)

    # -- Interrogation ------------------------------------------------------

    async def query(self, text: str, top_k: int, filters: dict | None = None) -> list[SearchResult]:
        return await self._run(self._query_sync, text, top_k, filters)

    def _query_sync(self, text: str, top_k: int, filters: dict | None) -> list[SearchResult]:
        vector = self._embedding.embed_query(text)
        return self._vector_store.search(vector, top_k=top_k, filters=filters)

    # -- Listing ------------------------------------------------------

    async def get_document(self, doc_id: str) -> DocumentRecord | None:
        return await self._run(self._get_document_sync, doc_id)

    def _get_document_sync(self, doc_id: str) -> DocumentRecord | None:
        record = self._registry.get(doc_id)
        if record is None:
            return None
        metadata = self._vector_store.get_document_metadata([doc_id]).get(doc_id, {})
        return dataclasses.replace(record, metadata=metadata)

    async def list_documents(self) -> list[DocumentRecord]:
        return await self._run(self._list_documents_sync)

    def _list_documents_sync(self) -> list[DocumentRecord]:
        records = self._registry.list()
        metadata_by_doc_id = self._vector_store.get_document_metadata([r.doc_id for r in records])
        return [dataclasses.replace(r, metadata=metadata_by_doc_id.get(r.doc_id, {})) for r in records]

    # -- Sources ------------------------------------------------------

    async def set_source(self, name: str, description: str, enabled: bool) -> None:
        await self._run(self._registry.set_source, name, description, enabled)

    async def get_sources(self) -> list[dict]:
        return await self._run(self._registry.get_sources)

    # -- Cycle de vie ------------------------------------------------------

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
        self._registry.close()
