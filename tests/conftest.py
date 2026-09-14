"""Fixtures et utilitaires partagés pour la suite de tests de ragifix.

Les backends lourds (embedding, base vectorielle, chunker tokenisé) sont
remplacés par des versions factices légères : la logique métier et les routes
peuvent ainsi être testées sans dépendre du réseau ni des modèles.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Rend le package importable sans installation (lancement direct de pytest).
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ragifix.registry import SqliteDocumentRegistry  # noqa: E402
from ragifix.service import RagifixService  # noqa: E402
from ragifix.vectorstore.base import SearchResult, VectorChunk, make_chunk_id  # noqa: E402


class FakeEmbeddingBackend:
    """Backend d'embedding fake : vecteurs constants, dimension fixée."""

    def __init__(self, dimension: int = 4):
        self._dimension = dimension

    def embed_documents(self, texts):
        return [[0.0] * self._dimension for _ in texts]

    def embed_query(self, text):
        return [0.0] * self._dimension

    @property
    def dimension(self) -> int:
        return self._dimension


class InMemoryVectorStore:
    """Base vectorielle en mémoire (respecte le contrat de VectorStore)."""

    def __init__(self):
        self._chunks: dict[str, VectorChunk] = {}

    def upsert(self, chunks):
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

    def delete_by_doc_id(self, doc_id, keep_chunk_ids):
        for chunk_id in [
            cid for cid, c in self._chunks.items()
            if c.doc_id == doc_id and cid not in keep_chunk_ids
        ]:
            del self._chunks[chunk_id]

    def delete_document(self, doc_id):
        for chunk_id in [
            cid for cid, c in self._chunks.items() if c.doc_id == doc_id
        ]:
            del self._chunks[chunk_id]

    def search(self, vector, top_k, filters=None):
        results = [
            SearchResult(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                score=0.0,
                metadata=dict(chunk.metadata),
            )
            for chunk in self._chunks.values()
        ]
        return results[:top_k]

    def get_document_metadata(self, doc_ids):
        result = {}
        for doc_id in doc_ids:
            chunk = self._chunks.get(make_chunk_id(doc_id, 0))
            if chunk is not None:
                result[doc_id] = dict(chunk.metadata)
        return result


class FakeParserBackend:
    """Backend de parsing fake : retourne le contenu décodé tel quel,
    préfixé pour signaler qu'il est passé par ce backend (utile pour
    distinguer ce chemin de TEXT_EXTENSIONS dans les assertions)."""

    def parse(self, content: bytes, extension: str, filename: str) -> str:
        return f"[fake-parsed:{extension}] {content.decode('utf-8', errors='replace')}"


class SimpleChunker:
    """Chunker simple (découpage en morceaux de taille fixe), sans tiktoken.

    Utilisé dans les tests d'intégration (service, routes) où la tokenisation
    n'est pas l'objet du test. La tokenisation réelle est couverte par
    ``test_processing.py``.
    """

    def __init__(self, chunk_size: int = 10):
        self.chunk_size = chunk_size

    def chunk(self, text):
        if not text or not text.strip():
            return []
        return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]


@pytest.fixture
def fake_embedding():
    return FakeEmbeddingBackend(dimension=4)


@pytest.fixture
def in_memory_vector_store():
    return InMemoryVectorStore()


@pytest.fixture
def simple_chunker():
    return SimpleChunker(chunk_size=8)


@pytest.fixture
def fake_parser():
    return FakeParserBackend()


@pytest.fixture
def registry(tmp_path):
    return SqliteDocumentRegistry(str(tmp_path / "registry.db"))


@pytest.fixture
def build_service(fake_embedding, in_memory_vector_store, registry, simple_chunker, fake_parser):
    def _build() -> RagifixService:
        return RagifixService(fake_embedding, in_memory_vector_store, registry, simple_chunker, fake_parser)

    return _build
