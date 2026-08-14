from __future__ import annotations

import httpx


class OpenAICompatibleBackend:
    """Backend d'embedding distant, via une API compatible OpenAI
    (endpoint POST {base_url}/embeddings)."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 30.0):
        self._model = model
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        self._dimension = len(self._request([" "])[0])

    def _request(self, texts: list[str]) -> list[list[float]]:
        response = self._client.post("/embeddings", json={"model": self._model, "input": texts})
        response.raise_for_status()
        data = response.json()["data"]
        ordered = sorted(data, key=lambda item: item["index"])
        return [item["embedding"] for item in ordered]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._request(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._request([text])[0]

    @property
    def dimension(self) -> int:
        return self._dimension

    def close(self) -> None:
        self._client.close()
