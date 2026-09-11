from __future__ import annotations

import json
from pathlib import Path

from .base import SearchResult, VectorChunk

# Facteur de sur-échantillonnage appliqué quand des `filters` sont fournis à
# `search` : FAISS ne sait pas filtrer nativement sur des métadonnées, donc
# on récupère top_k * OVERFETCH_FACTOR candidats puis on filtre en Python
# avant de tronquer à top_k. Best-effort assumé (voir docstring de la classe).
OVERFETCH_FACTOR = 10


class FaissVectorStore:
    """Backend de base vectorielle basé sur faiss-cpu, entièrement local et
    persisté sur disque (pas de mode "server" : FAISS n'a pas cette notion).

    FAISS n'indexe que des vecteurs (pas de payload, pas d'ID string natif) :
    un sidecar JSON (metadata.json) tient la correspondance chunk_id <->
    id entier FAISS ainsi que doc_id/text/metadata. Les deux fichiers sont
    réécrits intégralement à chaque mutation (upsert/delete) — pas de
    méthode flush exposée ailleurs dans le code.

    Les vecteurs sont normalisés (L2) avant indexation/recherche et l'index
    utilise le produit scalaire (IndexFlatIP) : produit scalaire sur
    vecteurs normalisés = similarité cosinus, pour rester cohérent avec le
    metric_type="COSINE" utilisé par MilvusVectorStore.

    Limite connue : `search(..., filters=...)` est du best-effort (voir
    OVERFETCH_FACTOR), pas un vrai filtre pré-recherche comme avec Milvus.

    ATTENTION : comme pour Milvus en mode "lite", un seul process doit
    détenir cette instance à la fois — ne jamais lancer deux instances de
    ragifix pointant vers le même index_path.
    """

    _INDEX_FILENAME = "index.faiss"
    _METADATA_FILENAME = "metadata.json"

    def __init__(self, dimension: int, index_path: str):
        import faiss

        self._faiss = faiss
        self._dimension = dimension
        self._dir = Path(index_path)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self._dir / self._INDEX_FILENAME
        self._metadata_file = self._dir / self._METADATA_FILENAME

        if self._index_file.exists() and self._metadata_file.exists():
            self._index = faiss.read_index(str(self._index_file))
            self._metadata: dict[str, dict] = json.loads(self._metadata_file.read_text(encoding="utf-8"))
        else:
            self._index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
            self._metadata = {}

    # -- Persistance ------------------------------------------------------

    def _save(self) -> None:
        self._faiss.write_index(self._index, str(self._index_file))
        self._metadata_file.write_text(json.dumps(self._metadata, ensure_ascii=False), encoding="utf-8")

    def _next_faiss_id(self) -> int:
        if not self._metadata:
            return 0
        return max(entry["faiss_id"] for entry in self._metadata.values()) + 1

    def _normalize(self, vectors: list[list[float]]):
        import numpy as np

        array = np.asarray(vectors, dtype="float32")
        self._faiss.normalize_L2(array)
        return array

    # -- VectorStore --------------------------------------------------------

    def upsert(self, chunks: list[VectorChunk]) -> None:
        if not chunks:
            return

        import numpy as np

        existing_ids = [
            self._metadata[c.chunk_id]["faiss_id"] for c in chunks if c.chunk_id in self._metadata
        ]
        if existing_ids:
            self._index.remove_ids(np.asarray(existing_ids, dtype="int64"))

        next_id = self._next_faiss_id()
        ids: list[int] = []
        for i, chunk in enumerate(chunks):
            faiss_id = next_id + i
            ids.append(faiss_id)
            self._metadata[chunk.chunk_id] = {
                "faiss_id": faiss_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }

        vectors = self._normalize([c.vector for c in chunks])
        self._index.add_with_ids(vectors, np.asarray(ids, dtype="int64"))
        self._save()

    def delete_by_doc_id(self, doc_id: str, keep_chunk_ids: list[str]) -> None:
        keep = set(keep_chunk_ids)
        orphan_chunk_ids = [
            chunk_id
            for chunk_id, entry in self._metadata.items()
            if entry["doc_id"] == doc_id and chunk_id not in keep
        ]
        self._remove_chunk_ids(orphan_chunk_ids)

    def delete_document(self, doc_id: str) -> None:
        chunk_ids = [chunk_id for chunk_id, entry in self._metadata.items() if entry["doc_id"] == doc_id]
        self._remove_chunk_ids(chunk_ids)

    def _remove_chunk_ids(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return

        import numpy as np

        faiss_ids = [self._metadata[chunk_id]["faiss_id"] for chunk_id in chunk_ids]
        self._index.remove_ids(np.asarray(faiss_ids, dtype="int64"))
        for chunk_id in chunk_ids:
            del self._metadata[chunk_id]
        self._save()

    def search(self, vector: list[float], top_k: int, filters: dict | None = None) -> list[SearchResult]:
        if self._index.ntotal == 0:
            return []

        query = self._normalize([vector])
        fetch_k = min(self._index.ntotal, top_k * OVERFETCH_FACTOR if filters else top_k)
        scores, faiss_ids = self._index.search(query, fetch_k)

        by_faiss_id = {entry["faiss_id"]: (chunk_id, entry) for chunk_id, entry in self._metadata.items()}

        results: list[SearchResult] = []
        for score, faiss_id in zip(scores[0], faiss_ids[0]):
            if faiss_id == -1:
                continue
            chunk_id, entry = by_faiss_id[int(faiss_id)]
            if filters and any(entry["metadata"].get(key) != value for key, value in filters.items()):
                continue
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    doc_id=entry["doc_id"],
                    text=entry["text"],
                    score=float(score),
                    metadata=entry["metadata"],
                )
            )
            if len(results) >= top_k:
                break

        return results
