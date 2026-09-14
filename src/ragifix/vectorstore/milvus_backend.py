from __future__ import annotations

from pathlib import Path

from .base import SearchResult, VectorChunk, _as_list, make_chunk_id, matches_filters


def _escape(value: str) -> str:
    """Échappement minimal pour les valeurs insérées dans une expression de
    filtre Milvus (guillemets doubles)."""
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


class MilvusVectorStore:
    """Backend de base vectorielle basé sur pymilvus (MilvusClient).

    Fonctionne aussi bien en mode "lite" (fichier local, aucun service à
    opérer — par défaut) qu'en mode "server" (Milvus distant, host/port).

    ATTENTION : en mode "lite", le fichier ne peut être ouvert que par un
    seul process à la fois. ragifix est conçu pour être le SEUL process qui
    instancie cette classe — ne jamais lancer deux instances de ragifix
    pointant vers le même fichier lite_path.
    """

    def __init__(
        self,
        collection_name: str,
        dimension: int,
        mode: str = "lite",
        lite_path: str = "./milvus_lite.db",
        host: str = "localhost",
        port: int = 19530,
    ):
        from pymilvus import DataType, MilvusClient

        if mode == "lite":
            # MilvusClient crée le fichier/dossier lite_path lui-même, mais pas
            # son arborescence parente — même précaution que registry.py.
            Path(lite_path).parent.mkdir(parents=True, exist_ok=True)

        uri = lite_path if mode == "lite" else f"http://{host}:{port}"
        self._client = MilvusClient(uri=uri)
        self._collection_name = collection_name

        if not self._client.has_collection(collection_name):
            schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, is_primary=True, max_length=128)
            schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=2048)
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="metadata", datatype=DataType.JSON)
            # Dupliqué depuis metadata["modified_at"] : un champ JSON ne permet pas
            # de comparaisons >=/<= fiables, un champ scalaire VARCHAR si (comparaison
            # lexicographique correcte pour des dates ISO 8601). Détail interne à ce
            # backend, jamais exposé au reste du programme.
            schema.add_field(field_name="modified_at", datatype=DataType.VARCHAR, max_length=64)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)

            index_params = self._client.prepare_index_params()
            index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")

            self._client.create_collection(
                collection_name=collection_name, schema=schema, index_params=index_params
            )

        self._client.load_collection(collection_name=collection_name)

    def upsert(self, chunks: list[VectorChunk]) -> None:
        if not chunks:
            return
        rows = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "text": c.text,
                "metadata": c.metadata,
                "modified_at": c.metadata.get("modified_at") or "",
                "vector": c.vector,
            }
            for c in chunks
        ]
        self._client.upsert(collection_name=self._collection_name, data=rows)

    def delete_by_doc_id(self, doc_id: str, keep_chunk_ids: list[str]) -> None:
        expr = f'doc_id == "{_escape(doc_id)}"'
        existing = self._client.query(
            collection_name=self._collection_name, filter=expr, output_fields=["chunk_id"]
        )
        existing_ids = {row["chunk_id"] for row in existing}
        orphan_ids = list(existing_ids - set(keep_chunk_ids))
        if orphan_ids:
            self._client.delete(collection_name=self._collection_name, ids=orphan_ids)

    def delete_document(self, doc_id: str) -> None:
        expr = f'doc_id == "{_escape(doc_id)}"'
        self._client.delete(collection_name=self._collection_name, filter=expr)

    def search(
        self, vector: list[float], top_k: int, filters: dict | None = None
    ) -> list[SearchResult]:
        filename_glob = filters.get("filename_glob") if filters else None
        expr = self._build_expr(filters) if filters else None
        # `filename_glob` n'est pas exprimable de façon fiable en filtre Milvus
        # (JSON path + wildcard) : on sur-échantillonne puis on post-filtre en
        # Python. Limite : peut renvoyer moins de `top_k` résultats si le(s)
        # motif(s) sont très restrictifs.
        limit = min(top_k * 5, 200) if filename_glob else top_k

        results = self._client.search(
            collection_name=self._collection_name,
            data=[vector],
            limit=limit,
            filter=expr,
            output_fields=["doc_id", "text", "metadata"],
        )
        hits = results[0] if results else []

        # Le champ primaire (chunk_id) est exposé directement sous son
        # propre nom dans le hit, pas sous "id" — voir pymilvus MilvusClient.search().
        output: list[SearchResult] = []
        for hit in hits:
            entity = hit.get("entity", hit)
            metadata = entity.get("metadata") or {}
            if filename_glob is not None and not matches_filters(metadata, {"filename_glob": filename_glob}):
                continue
            output.append(
                SearchResult(
                    chunk_id=hit["chunk_id"],
                    doc_id=entity["doc_id"],
                    text=entity["text"],
                    score=hit["distance"],
                    metadata=metadata,
                )
            )
            if len(output) >= top_k:
                break
        return output

    @staticmethod
    def _build_expr(filters: dict) -> str | None:
        clauses = []
        for key in ("source", "extension"):
            value = filters.get(key)
            if value is None:
                continue
            values = _as_list(value)
            if len(values) == 1:
                clauses.append(f'metadata["{key}"] == "{_escape(values[0])}"')
            else:
                # `in` n'est pas supporté par Milvus sur un JSON path (seulement
                # sur une vraie référence de champ) : on exprime le OR explicitement.
                ored = " or ".join(f'metadata["{key}"] == "{_escape(v)}"' for v in values)
                clauses.append(f"({ored})")

        modified_after = filters.get("modified_after")
        if modified_after is not None:
            clauses.append(f'modified_at >= "{_escape(modified_after)}"')
        modified_before = filters.get("modified_before")
        if modified_before is not None:
            clauses.append(f'modified_at <= "{_escape(modified_before)}"')

        return " and ".join(clauses) if clauses else None

    def get_document_metadata(self, doc_ids: list[str]) -> dict[str, dict]:
        if not doc_ids:
            return {}
        chunk_ids = [make_chunk_id(doc_id, 0) for doc_id in doc_ids]
        rows = self._client.get(
            collection_name=self._collection_name, ids=chunk_ids, output_fields=["chunk_id", "metadata"]
        )
        metadata_by_chunk_id = {row["chunk_id"]: (row.get("metadata") or {}) for row in rows}
        return {
            doc_id: metadata_by_chunk_id[chunk_id]
            for doc_id, chunk_id in zip(doc_ids, chunk_ids)
            if chunk_id in metadata_by_chunk_id
        }
