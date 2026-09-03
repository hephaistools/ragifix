"""Tests des routes HTTP via FastAPI TestClient.

Le vrai `RagifixService` est utilisé avec de faux backends (voir conftest) :
les routes ne portent aucune logique métier, on teste donc ici leur façonnage
(autorisation, validation, codes de réponse) sur un service fonctionnel.
"""

from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ragifix.auth import TokenAuth
from ragifix.routes import build_router

AUTH = {"Authorization": "Bearer test-token"}
BAD_AUTH = {"Authorization": "Bearer wrong"}


def _build_client(service, max_mb=50) -> TestClient:
    auth = TokenAuth(expected_token="test-token")
    router = build_router(service, auth, default_top_k=5, max_document_size_mb=max_mb)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_health_no_auth(build_service):
    service = build_service()
    try:
        client = _build_client(service)
        resp = client.get("/health")
    finally:
        service.shutdown()
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_query_missing_token(build_service):
    service = build_service()
    try:
        client = _build_client(service)
        resp = client.post("/query", json={"query": "x"})
    finally:
        service.shutdown()
    assert resp.status_code == 401


def test_query_bad_token(build_service):
    service = build_service()
    try:
        client = _build_client(service)
        resp = client.post("/query", json={"query": "x"}, headers=BAD_AUTH)
    finally:
        service.shutdown()
    assert resp.status_code == 401


def test_put_metadata_invalid_json(build_service):
    service = build_service()
    try:
        client = _build_client(service)
        resp = client.put(
            "/documents/doc1", params={"extension": "txt", "metadata": "not-json"}, headers=AUTH
        )
    finally:
        service.shutdown()
    assert resp.status_code == 400


def test_put_unsupported_type(build_service):
    service = build_service()
    try:
        client = _build_client(service)
        resp = client.put("/documents/doc1", params={"extension": "zip"}, content=b"data", headers=AUTH)
    finally:
        service.shutdown()
    assert resp.status_code == 415


def test_put_empty_body(build_service):
    service = build_service()
    try:
        client = _build_client(service)
        resp = client.put("/documents/doc1", params={"extension": "txt"}, content=b"", headers=AUTH)
    finally:
        service.shutdown()
    assert resp.status_code == 400


def test_put_too_large(build_service):
    service = build_service()
    try:
        client = _build_client(service, max_mb=0.00001)
        resp = client.put("/documents/doc1", params={"extension": "txt"}, content=b"x" * 40, headers=AUTH)
    finally:
        service.shutdown()
    assert resp.status_code == 413


def test_put_success(build_service):
    service = build_service()
    try:
        client = _build_client(service)
        resp = client.put(
            "/documents/doc1",
            params={"extension": "txt"},
            content=b"this is a test document content that will be chunked",
            headers=AUTH,
        )
    finally:
        service.shutdown()
    assert resp.status_code == 200
    body = resp.json()
    assert body["doc_id"] == "doc1"
    assert body["extension"] == "txt"
    assert body["chunk_count"] >= 1
    assert body["origin"] is None


def test_put_with_origin_metadata(build_service):
    service = build_service()
    origin = {"kind": "https", "uri": "https://contoso.sharepoint.com/doc.pdf", "label": "doc.pdf"}
    try:
        client = _build_client(service)
        resp = client.put(
            "/documents/doc1",
            params={"extension": "txt", "metadata": json.dumps({"origin": origin})},
            content=b"this is a test document content that will be chunked",
            headers=AUTH,
        )
    finally:
        service.shutdown()
    assert resp.status_code == 200
    assert resp.json()["origin"] == origin


def test_delete_missing(build_service):
    service = build_service()
    try:
        client = _build_client(service)
        resp = client.delete("/documents/unknown", headers=AUTH)
    finally:
        service.shutdown()
    assert resp.status_code == 404


def test_delete_success(build_service):
    service = build_service()
    try:
        client = _build_client(service)
        client.put(
            "/documents/doc1",
            params={"extension": "txt"},
            content=b"content to delete afterwards",
            headers=AUTH,
        )
        resp = client.delete("/documents/doc1", headers=AUTH)
    finally:
        service.shutdown()
    assert resp.status_code == 204


def test_list_documents(build_service):
    service = build_service()
    try:
        client = _build_client(service)
        client.put("/documents/notes/a", params={"extension": "md"}, content=b"one", headers=AUTH)
        client.put("/documents/notes/b", params={"extension": "md"}, content=b"two", headers=AUTH)
        resp = client.get("/documents", headers=AUTH)
    finally:
        service.shutdown()
    assert resp.status_code == 200
    doc_ids = {d["doc_id"] for d in resp.json()["documents"]}
    assert doc_ids == {"notes/a", "notes/b"}


def test_sources_flow(build_service):
    service = build_service()
    try:
        client = _build_client(service)
        created = client.post(
            "/sources", json=[{"name": "collector", "description": "desc", "enabled": True}], headers=AUTH
        )
        assert created.status_code == 200
        listed = client.get("/sources", headers=AUTH)
        assert listed.status_code == 200
        assert listed.json()["sources"][0]["name"] == "collector"
    finally:
        service.shutdown()


def test_query_success(build_service):
    service = build_service()
    try:
        client = _build_client(service)
        client.put(
            "/documents/doc1",
            params={"extension": "txt"},
            content=b"this document is about apples and oranges and fruits",
            headers=AUTH,
        )
        resp = client.post("/query", json={"query": "apples", "top_k": 5}, headers=AUTH)
    finally:
        service.shutdown()
    assert resp.status_code == 200
    assert len(resp.json()["results"]) >= 1


def test_query_returns_origin(build_service):
    service = build_service()
    origin = {"kind": "file", "uri": "file:///srv/docs/apples.txt", "label": "apples.txt"}
    try:
        client = _build_client(service)
        client.put(
            "/documents/doc1",
            params={"extension": "txt", "metadata": json.dumps({"origin": origin})},
            content=b"this document is about apples and oranges and fruits",
            headers=AUTH,
        )
        resp = client.post("/query", json={"query": "apples", "top_k": 5}, headers=AUTH)
    finally:
        service.shutdown()
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) >= 1
    assert results[0]["origin"] == origin
