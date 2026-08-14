from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class VectorChunk:
    """Un chunk prêt à être inséré dans la base vectorielle.

    chunk_id est déterministe (dérivé de doc_id + index du chunk) : cela
    permet un upsert idempotent plutôt qu'un delete-then-insert, qui
    créerait une fenêtre d'incohérence où le document n'existerait plus du
    tout dans la base pendant une mise à jour.
    """

    chunk_id: str
    doc_id: str
    text: str
    vector: list[float]
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class VectorStore(Protocol):
    """Interface commune des backends de base vectorielle (upsert/delete/search).

    IMPORTANT (mode Milvus Lite) : un seul process doit détenir cette
    instance à la fois — voir ragifix.service.RagifixService, qui sérialise
    tous les appels sur un unique thread dédié. Ne jamais lancer plusieurs
    instances de ragifix (ni plusieurs workers) pointant vers le même
    fichier Milvus Lite.
    """

    def upsert(self, chunks: list[VectorChunk]) -> None: ...

    def delete_by_doc_id(self, doc_id: str, keep_chunk_ids: list[str]) -> None:
        """Supprime les chunks d'un document qui ne sont plus d'actualité
        (ceux dont le chunk_id n'apparaît pas dans `keep_chunk_ids`),
        typiquement appelé juste après un upsert des nouveaux chunks."""
        ...

    def delete_document(self, doc_id: str) -> None:
        """Supprime tous les chunks d'un document."""
        ...

    def search(
        self, vector: list[float], top_k: int, filters: dict | None = None
    ) -> list[SearchResult]: ...
