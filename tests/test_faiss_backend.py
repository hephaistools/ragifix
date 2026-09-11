"""Tests de FaissVectorStore avec le véritable faiss-cpu (léger, pas de
modèle à télécharger ni de service externe — contrairement à
MilvusVectorStore, volontairement non testé avec un vrai backend)."""

from __future__ import annotations

from ragifix.vectorstore.base import VectorChunk
from ragifix.vectorstore.faiss_backend import FaissVectorStore


def _store(tmp_path, dimension: int = 3) -> FaissVectorStore:
    return FaissVectorStore(dimension=dimension, index_path=str(tmp_path / "faiss_index"))


def test_upsert_then_search_finds_chunk(tmp_path):
    store = _store(tmp_path)
    chunk = VectorChunk(chunk_id="c1", doc_id="d1", text="hello", vector=[1.0, 0.0, 0.0])
    store.upsert([chunk])

    results = store.search([1.0, 0.0, 0.0], top_k=5)

    assert len(results) == 1
    assert results[0].chunk_id == "c1"
    assert results[0].doc_id == "d1"
    assert results[0].text == "hello"


def test_search_orders_by_similarity(tmp_path):
    store = _store(tmp_path)
    store.upsert(
        [
            VectorChunk(chunk_id="close", doc_id="d1", text="close", vector=[1.0, 0.0, 0.0]),
            VectorChunk(chunk_id="far", doc_id="d1", text="far", vector=[0.0, 1.0, 0.0]),
        ]
    )

    results = store.search([0.9, 0.1, 0.0], top_k=2)

    assert [r.chunk_id for r in results] == ["close", "far"]


def test_upsert_overwrites_existing_chunk_id(tmp_path):
    store = _store(tmp_path)
    store.upsert([VectorChunk(chunk_id="c1", doc_id="d1", text="v1", vector=[1.0, 0.0, 0.0])])
    store.upsert([VectorChunk(chunk_id="c1", doc_id="d1", text="v2", vector=[0.0, 1.0, 0.0])])

    results = store.search([0.0, 1.0, 0.0], top_k=5)

    assert len(results) == 1
    assert results[0].text == "v2"


def test_delete_by_doc_id_removes_orphans_keeps_listed(tmp_path):
    store = _store(tmp_path)
    store.upsert(
        [
            VectorChunk(chunk_id="c1", doc_id="d1", text="keep", vector=[1.0, 0.0, 0.0]),
            VectorChunk(chunk_id="c2", doc_id="d1", text="drop", vector=[0.0, 1.0, 0.0]),
        ]
    )

    store.delete_by_doc_id("d1", keep_chunk_ids=["c1"])
    results = store.search([1.0, 0.0, 0.0], top_k=5)

    assert {r.chunk_id for r in results} == {"c1"}


def test_delete_document_removes_all_its_chunks(tmp_path):
    store = _store(tmp_path)
    store.upsert(
        [
            VectorChunk(chunk_id="c1", doc_id="d1", text="a", vector=[1.0, 0.0, 0.0]),
            VectorChunk(chunk_id="c2", doc_id="d2", text="b", vector=[0.0, 1.0, 0.0]),
        ]
    )

    store.delete_document("d1")
    results = store.search([1.0, 0.0, 0.0], top_k=5)

    assert {r.chunk_id for r in results} == {"c2"}


def test_search_applies_metadata_filter(tmp_path):
    store = _store(tmp_path)
    store.upsert(
        [
            VectorChunk(chunk_id="c1", doc_id="d1", text="a", vector=[1.0, 0.0, 0.0], metadata={"lang": "fr"}),
            VectorChunk(chunk_id="c2", doc_id="d2", text="b", vector=[1.0, 0.0, 0.0], metadata={"lang": "en"}),
        ]
    )

    results = store.search([1.0, 0.0, 0.0], top_k=5, filters={"lang": "en"})

    assert {r.chunk_id for r in results} == {"c2"}


def test_search_on_empty_store_returns_empty_list(tmp_path):
    store = _store(tmp_path)
    assert store.search([1.0, 0.0, 0.0], top_k=5) == []


def test_persistence_reloads_from_disk(tmp_path):
    index_path = str(tmp_path / "faiss_index")
    store = FaissVectorStore(dimension=3, index_path=index_path)
    store.upsert([VectorChunk(chunk_id="c1", doc_id="d1", text="hello", vector=[1.0, 0.0, 0.0])])

    reloaded = FaissVectorStore(dimension=3, index_path=index_path)
    results = reloaded.search([1.0, 0.0, 0.0], top_k=5)

    assert len(results) == 1
    assert results[0].chunk_id == "c1"
    assert results[0].text == "hello"
