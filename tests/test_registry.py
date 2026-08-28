"""Tests du registre SQLite des documents indexés."""

from __future__ import annotations

from ragifix.registry import SqliteDocumentRegistry


def test_get_on_empty_returns_none(tmp_path):
    reg = SqliteDocumentRegistry(str(tmp_path / "registry.db"))
    assert reg.get("missing") is None
    reg.close()


def test_upsert_then_get(tmp_path):
    reg = SqliteDocumentRegistry(str(tmp_path / "registry.db"))
    record = reg.upsert("doc1", ["c1", "c2"], "txt", {"k": "v"})
    assert record.doc_id == "doc1"
    assert record.chunk_count == 2
    found = reg.get("doc1")
    assert found is not None
    assert found.chunk_ids == ["c1", "c2"]
    assert found.extension == "txt"
    assert found.metadata == {"k": "v"}
    reg.close()


def test_upsert_is_idempotent(tmp_path):
    reg = SqliteDocumentRegistry(str(tmp_path / "registry.db"))
    reg.upsert("doc1", ["c1"], "txt", {})
    reg.upsert("doc1", ["c2", "c3"], "md", {"k": "v"})
    found = reg.get("doc1")
    assert found.chunk_ids == ["c2", "c3"]
    assert found.extension == "md"
    assert found.metadata == {"k": "v"}
    reg.close()


def test_delete(tmp_path):
    reg = SqliteDocumentRegistry(str(tmp_path / "registry.db"))
    reg.upsert("doc1", ["c1"], "txt", {})
    assert reg.delete("doc1") is True
    assert reg.delete("doc1") is False
    assert reg.get("doc1") is None
    reg.close()


def test_list(tmp_path):
    reg = SqliteDocumentRegistry(str(tmp_path / "registry.db"))
    reg.upsert("a/1", ["c"], "txt", {})
    reg.upsert("b/2", ["c"], "txt", {})
    all_docs = reg.list()
    assert {r.doc_id for r in all_docs} == {"a/1", "b/2"}
    reg.close()


def test_list_with_prefix(tmp_path):
    reg = SqliteDocumentRegistry(str(tmp_path / "registry.db"))
    reg.upsert("notes/readme", ["c"], "md", {})
    reg.upsert("notes/other", ["c"], "md", {})
    reg.upsert("other/file", ["c"], "txt", {})
    prefix_docs = reg.list(prefix="notes/")
    assert {r.doc_id for r in prefix_docs} == {"notes/readme", "notes/other"}
    reg.close()


def test_list_prefix_escapes_like_chars(tmp_path):
    reg = SqliteDocumentRegistry(str(tmp_path / "registry.db"))
    reg.upsert("a%b", ["c"], "txt", {})
    reg.upsert("axyz", ["c"], "txt", {})
    matched = reg.list(prefix="a%b")
    assert [r.doc_id for r in matched] == ["a%b"]
    reg.close()


def test_metadata_unicode_roundtrip(tmp_path):
    reg = SqliteDocumentRegistry(str(tmp_path / "registry.db"))
    reg.upsert("doc1", ["c"], "txt", {"nom": "café", "rue": "déjà→vu"})
    found = reg.get("doc1")
    assert found.metadata["nom"] == "café"
    assert found.metadata["rue"] == "déjà→vu"
    reg.close()


def test_sources(tmp_path):
    reg = SqliteDocumentRegistry(str(tmp_path / "registry.db"))
    reg.set_source("collector", "une description", True)
    reg.set_source("web", "", False)
    sources = reg.get_sources()
    by_name = {s["name"]: s for s in sources}
    assert by_name["collector"]["description"] == "une description"
    assert by_name["collector"]["enabled"] is True
    assert by_name["web"]["enabled"] is False
    # Mise à jour idempotente d'une source existante.
    reg.set_source("collector", "nouvelle", False)
    assert reg.get_sources()[0]["description"] == "nouvelle"
    reg.close()
