"""Registre des documents indexés par ragifix.

Aucune notion de cursor de synchronisation ici (c'est entièrement l'affaire
de ragifix-collector, côté client de l'API). Ce registre retient, pour
chaque doc_id connu, uniquement les chunk_ids qui lui appartiennent dans
la base vectorielle (nécessaire pour nettoyer les chunks orphelins lors
d'une mise à jour) et la date de dernière mise à jour — ce qui permet
aussi de répondre à "quels documents sont indexés ?" sans avoir à
interroger la base vectorielle elle-même.

`metadata` (qui inclut désormais `extension`) ne vit plus ici : elle est
stockée uniquement dans la base vectorielle (dupliquée par chunk), lue via
`VectorStore.get_document_metadata`. `DocumentRecord.metadata` reste un
champ du dataclass mais il est rempli par `RagifixService`, jamais par ce
registre (toujours `{}` en sortie d'ici).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class DocumentRecord:
    doc_id: str
    chunk_ids: list[str]
    updated_at: str  # ISO 8601 UTC
    metadata: dict = field(default_factory=dict)

    @property
    def chunk_count(self) -> int:
        return len(self.chunk_ids)


@runtime_checkable
class DocumentRegistry(Protocol):
    def get(self, doc_id: str) -> DocumentRecord | None: ...

    def upsert(self, doc_id: str, chunk_ids: list[str]) -> DocumentRecord: ...

    def delete(self, doc_id: str) -> bool:
        """Retourne True si le document existait (et a été supprimé)."""
        ...

    def list(self) -> list[DocumentRecord]: ...

    def set_source(self, name: str, description: str, enabled: bool) -> None: ...

    def get_sources(self) -> list[dict]: ...

    def close(self) -> None: ...


class SqliteDocumentRegistry:
    """Implémentation SQLite (backend par défaut, aucun service à opérer).

    ragifix étant le seul process à manipuler l'état d'indexation (voir
    ragifix.service.RagifixService, qui sérialise tous les accès sur un
    unique thread), une simple connexion protégée par un verrou est
    amplement suffisante.
    """

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    chunk_ids TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sources (
                    name TEXT PRIMARY KEY,
                    description TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _row_to_record(row: tuple) -> DocumentRecord:
        doc_id, chunk_ids, updated_at = row
        return DocumentRecord(
            doc_id=doc_id,
            chunk_ids=json.loads(chunk_ids),
            updated_at=updated_at,
        )

    def get(self, doc_id: str) -> DocumentRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT doc_id, chunk_ids, updated_at FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            return self._row_to_record(row) if row else None

    def upsert(self, doc_id: str, chunk_ids: list[str]) -> DocumentRecord:
        updated_at = datetime.now(timezone.utc).isoformat()
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO documents (doc_id, chunk_ids, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    chunk_ids = excluded.chunk_ids,
                    updated_at = excluded.updated_at
                """,
                (doc_id, json.dumps(chunk_ids), updated_at),
            )
        return DocumentRecord(doc_id=doc_id, chunk_ids=chunk_ids, updated_at=updated_at)

    def delete(self, doc_id: str) -> bool:
        with self._lock, self._conn:
            cursor = self._conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            return cursor.rowcount > 0

    def list(self) -> list[DocumentRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT doc_id, chunk_ids, updated_at FROM documents ORDER BY doc_id"
            ).fetchall()
            return [self._row_to_record(row) for row in rows]

    def close(self) -> None:
        self._conn.close()

    def set_source(self, name: str, description: str, enabled: bool) -> None:
        updated_at = datetime.now(timezone.utc).isoformat()
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO sources (name, description, enabled, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    description = excluded.description,
                    enabled = excluded.enabled,
                    updated_at = excluded.updated_at
                """,
                (name, description, 1 if enabled else 0, updated_at),
            )

    def get_sources(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT name, description, enabled, updated_at FROM sources ORDER BY name"
            ).fetchall()
            return [
                {
                    "name": row[0],
                    "description": row[1],
                    "enabled": bool(row[2]),
                    "updated_at": row[3],
                }
                for row in rows
            ]


def build_registry(backend: str, sqlite_path: str) -> DocumentRegistry:
    if backend == "sqlite":
        return SqliteDocumentRegistry(sqlite_path)
    raise ValueError(f"Backend de registre inconnu: {backend}")
