"""Parsing et chunking des documents.

Regroupés volontairement dans un seul fichier : le rôle de ragifix ici est
de mettre en relation des outils déjà existants (pymupdf4llm,
tiktoken), pas de recréer une architecture modulaire pour des briques qui
n'en ont pas besoin.

Le parsing prend en entrée un flux d'octets déjà entièrement en mémoire
(jamais un chemin de fichier ni un objet spoolé sur disque) — cohérent avec
la contrainte de sécurité "zéro écriture temporaire", qui vaut aussi bien
pour un contenu SharePoint relayé par ragifix-collector que pour tout
autre appelant de l'API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = {"txt", "md"}
PDF_EXTENSIONS = {"pdf"}

ALL_SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | PDF_EXTENSIONS


class UnsupportedFileTypeError(Exception):
    """Levée pour un type de fichier non supporté par ce pipeline de parsing.

    Note : ce projet ne fait volontairement ni OCR ni speech-to-text. Pour
    ingérer des images ou de l'audio, la voie recommandée est un modèle
    d'embedding multimodal, pas une brique de parsing supplémentaire ici.
    """


def parse_document(content: bytes, extension: str, filename: str = "document") -> str:
    """Parse un document en mémoire et retourne son texte (ou markdown)."""
    ext = extension.lower().lstrip(".")

    if ext in TEXT_EXTENSIONS:
        return content.decode("utf-8", errors="replace")

    if ext in PDF_EXTENSIONS:
        return _parse_pdf(content)

    raise UnsupportedFileTypeError(
        f"Extension '.{ext}' non supportée par ce pipeline. "
        "Ce projet ne fait ni OCR ni speech-to-text : pour des images ou de "
        "l'audio, utilisez un modèle d'embedding multimodal plutôt que "
        "d'ajouter ces briques ici."
    )


def _parse_pdf(content: bytes) -> str:
    import fitz  # PyMuPDF
    import pymupdf4llm

    doc = fitz.open(stream=content, filetype="pdf")
    try:
        return pymupdf4llm.to_markdown(doc)
    finally:
        doc.close()


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #

@dataclass
class ChunkerConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64


class TokenChunker:
    """Découpe un texte en chunks de taille fixe (en tokens) avec
    chevauchement, via tiktoken pour le comptage.

    Note opérationnelle : le fichier d'encodage BPE est téléchargé au
    premier usage puis mis en cache localement. En environnement sans accès
    sortant à internet, prévoir de pré-peupler le cache (variable
    d'environnement TIKTOKEN_CACHE_DIR) — voir le README.
    """

    def __init__(self, config: ChunkerConfig):
        if config.chunk_overlap >= config.chunk_size:
            raise ValueError("chunk_overlap doit être strictement inférieur à chunk_size")
        self._config = config
        import tiktoken

        self._encoding = tiktoken.get_encoding("cl100k_base")

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        tokens = self._encoding.encode(text, disallowed_special=())
        size = self._config.chunk_size
        overlap = self._config.chunk_overlap
        step = size - overlap

        chunks: list[str] = []
        start = 0
        while start < len(tokens):
            window = tokens[start : start + size]
            if not window:
                break
            chunks.append(self._encoding.decode(window))
            if start + size >= len(tokens):
                break
            start += step
        return chunks


def build_chunker(strategy: str, chunk_size: int, chunk_overlap: int) -> TokenChunker:
    if strategy == "token":
        return TokenChunker(ChunkerConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    raise ValueError(f"Stratégie de chunking inconnue: {strategy}")
