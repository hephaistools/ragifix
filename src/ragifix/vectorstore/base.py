from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Protocol, runtime_checkable


def make_chunk_id(doc_id: str, index: int) -> str:
    """ID de chunk déterministe (doc_id + index) — permet un upsert
    idempotent plutôt qu'un delete-then-insert (qui laisserait une fenêtre
    de temps où le document n'existerait plus du tout dans la base).

    Le fait que ce soit déterministe permet aussi de retrouver le chunk
    d'index 0 d'un doc_id sans requête préalable (point-lookup), utile
    pour lire les métadonnées d'un document sans recherche vectorielle."""
    digest = hashlib.sha256(f"{doc_id}::{index}".encode("utf-8")).hexdigest()
    return digest[:32]


def _as_list(value) -> list:
    return value if isinstance(value, list) else [value]


def matches_filters(metadata: dict, filters: dict | None) -> bool:
    """Vrai si `metadata` satisfait tous les filtres (combinés en AND).

    Clés reconnues dans `filters` :
    - `source`, `extension` : valeur scalaire (égalité) ou liste de valeurs
      (vrai si `metadata[clé]` est dans la liste — OR intra-clé).
    - `filename_glob` : motif `fnmatch` (ex. `*rapport*.pdf`) ou liste de
      motifs (OR intra-clé), comparé à `metadata["filename"]`.
    - `modified_after` / `modified_before` : bornes (incluses) sur
      `metadata["modified_at"]`, comparées lexicographiquement (chaînes
      ISO 8601). Valeur scalaire uniquement (pas de liste).
    """
    if not filters:
        return True

    for key in ("source", "extension"):
        value = filters.get(key)
        if value is not None and metadata.get(key) not in _as_list(value):
            return False

    filename_glob = filters.get("filename_glob")
    if filename_glob is not None:
        filename = metadata.get("filename", "")
        if not any(fnmatch(filename, pattern) for pattern in _as_list(filename_glob)):
            return False

    modified_at = metadata.get("modified_at")
    modified_after = filters.get("modified_after")
    if modified_after is not None and (modified_at is None or modified_at < modified_after):
        return False
    modified_before = filters.get("modified_before")
    if modified_before is not None and (modified_at is None or modified_at > modified_before):
        return False

    return True


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

    def get_document_metadata(self, doc_ids: list[str]) -> dict[str, dict]:
        """Métadonnées de chaque doc_id connu, lues depuis son chunk d'index 0
        (point-lookup, sans recherche vectorielle). Les doc_id absents de la
        base ne figurent pas dans le résultat."""
        ...
