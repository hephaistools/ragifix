from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Interface commune des backends d'embedding (local ou distant).

    Le choix du backend est piloté uniquement par configuration
    (embedding.backend) — aucun changement de code applicatif requis pour
    basculer de l'un à l'autre.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    @property
    def dimension(self) -> int: ...
