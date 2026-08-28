"""Tests de la logique métier de ragifix (RagifixService).

Le service est testé avec de vrais fakes (embedding, vector store, registre,
chunker) : toute la logique d'ingestion/suppression/interrogation est couverte
sans dépendre des backends lourds.
"""

from __future__ import annotations

import asyncio

import pytest

from ragifix.service import EmptyDocumentError, make_chunk_id, RagifixService


def test_make_chunk_id_deterministic():
    assert make_chunk_id("doc1", 0) == make_chunk_id("doc1", 0)
    assert make_chunk_id("doc1", 0) != make_chunk_id("doc1", 1)
    assert make_chunk_id("doc1", 0) != make_chunk_id("doc2", 0)


def test_ingest_document(build_service):
    service = build_service()
    try:
        content = b"this is a sufficiently long test document to be split into several chunks"
        record = asyncio.run(service.ingest_document("doc1", content, "txt", {"src": "test"}))
        assert record.doc_id == "doc1"
        assert record.extension == "txt"
        assert record.chunk_count >= 1
        assert record.metadata == {"src": "test"}
    finally:
        service.shutdown()


def test_ingest_is_idempotent(build_service):
    service = build_service()
    try:
        content = b"aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii jjjj kkkk llll mmmm nnnn oooo"
        r1 = asyncio.run(service.ingest_document("doc1", content, "txt", {}))
        r2 = asyncio.run(service.ingest_document("doc1", content, "txt", {}))
        assert list(r1.chunk_ids) == list(r2.chunk_ids)
        got = asyncio.run(service.get_document("doc1"))
        assert got is not None
        assert got.chunk_count == r1.chunk_count
    finally:
        service.shutdown()


def test_ingest_empty_raises(build_service):
    service = build_service()
    try:
        with pytest.raises(EmptyDocumentError):
            asyncio.run(service.ingest_document("empty", b"", "txt", {}))
    finally:
        service.shutdown()


def test_delete_document(build_service):
    service = build_service()
    try:
        asyncio.run(service.ingest_document("doc1", b"some content here to chunk", "txt", {}))
        assert asyncio.run(service.delete_document("doc1")) is True
        assert asyncio.run(service.delete_document("doc1")) is False
        assert asyncio.run(service.get_document("doc1")) is None
    finally:
        service.shutdown()


def test_query(build_service):
    service = build_service()
    try:
        asyncio.run(
            service.ingest_document("doc1", b"this document talks about apples and oranges", "txt", {"src": "x"})
        )
        results = asyncio.run(service.query("apples", top_k=5))
        assert len(results) >= 1
        assert all(r.doc_id == "doc1" for r in results)
        assert results[0].metadata == {"src": "x"}
    finally:
        service.shutdown()


def test_list_documents(build_service):
    service = build_service()
    try:
        asyncio.run(service.ingest_document("notes/a", b"content one here", "txt", {}))
        asyncio.run(service.ingest_document("notes/b", b"content two here", "txt", {}))
        asyncio.run(service.ingest_document("other/c", b"content three here", "txt", {}))
        all_docs = asyncio.run(service.list_documents())
        assert {r.doc_id for r in all_docs} == {"notes/a", "notes/b", "other/c"}
        prefix_docs = asyncio.run(service.list_documents(prefix="notes/"))
        assert {r.doc_id for r in prefix_docs} == {"notes/a", "notes/b"}
    finally:
        service.shutdown()


def test_sources(build_service):
    service = build_service()
    try:
        asyncio.run(service.set_source("collector", "une description", True))
        sources = asyncio.run(service.get_sources())
        assert sources[0]["name"] == "collector"
        assert sources[0]["description"] == "une description"
        assert sources[0]["enabled"] is True
    finally:
        service.shutdown()
