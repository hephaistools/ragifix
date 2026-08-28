"""Tests du backend d'embedding OpenAI-compatible.

Aucun appel réseau : on remplace le client httpx par un fake retournant des
réponses factices. Cela couvre le tri des embeddings par index, la dimension
(probe à l'initialisation), le corps vide et la propagation des erreurs.
"""

from __future__ import annotations

import httpx
import pytest

from ragifix.embedding.openai_compat_backend import OpenAICompatibleBackend


class _FakeResponse:
    def __init__(self, data, status=200):
        self._data = data
        self._status = status

    def raise_for_status(self):
        if self._status >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)

    def json(self):
        return {"data": self._data, "model": "m", "object": "list"}


class _FakeClient:
    def __init__(self, data, status=200):
        self._data = data
        self._status = status
        self.closed = False

    def post(self, path, json):
        return _FakeResponse(self._data, self._status)

    def close(self):
        self.closed = True


def _build(data, status=200):
    backend = object.__new__(OpenAICompatibleBackend)
    backend._model = "m"
    backend._client = _FakeClient(data, status)
    backend._dimension = 4
    return backend


def test_init_probes_dimension(monkeypatch):
    # L'initialisation appelle self._request([" "]) : on le patche au niveau classe.
    monkeypatch.setattr(OpenAICompatibleBackend, "_request", lambda self, texts: [[0.0] * 4])
    backend = OpenAICompatibleBackend(base_url="http://localhost:1234/v1/", api_key="k", model="m")
    assert backend.dimension == 4


def test_embed_documents_tries_by_index():
    # « server » renverse les index ; le backend doit les retrier par index croissant.
    backend = _build([{"index": 1, "embedding": [7.0] * 4}, {"index": 0, "embedding": [8.0] * 4}])
    result = backend.embed_documents(["a", "b"])
    assert result == [[8.0, 8.0, 8.0, 8.0], [7.0, 7.0, 7.0, 7.0]]


def test_embed_query():
    backend = _build([{"index": 0, "embedding": [9.0] * 4}])
    assert backend.embed_query("x") == [9.0, 9.0, 9.0, 9.0]


def test_embed_documents_empty():
    backend = _build([{"index": 0, "embedding": [1.0] * 4}])
    assert backend.embed_documents([]) == []


def test_embed_documents_raises_on_server_error():
    backend = _build([{"index": 0, "embedding": [1.0] * 4}], status=500)
    with pytest.raises(httpx.HTTPStatusError):
        backend.embed_documents(["x"])


def test_close():
    backend = _build([{"index": 0, "embedding": [1.0] * 4}])
    backend.close()
    assert backend._client.closed is True
