from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class DocumentParserBackend(Protocol):
    """Backend de parsing des documents « riches » (docx, pptx, xlsx, html...).

    Le choix du backend est piloté uniquement par configuration
    (parsing.backend) — même principe que EmbeddingBackend/VectorStore :
    un seul backend actif gère l'intégralité de ce lot d'extensions, pas de
    négociation par extension entre plusieurs backends.

    Le texte brut (.txt/.md) et le PDF (via pymupdf4llm) restent hors de
    cette abstraction dans ragifix.processing : ils n'ont pas besoin d'un
    moteur pluggable.
    """

    def parse(self, content: bytes, extension: str, filename: str) -> str: ...
