from __future__ import annotations


class FastEmbedBackend:
    """Backend d'embedding local, basé sur fastembed. Le modèle est
    téléchargé (une fois, mis en cache) et chargé en mémoire dans le
    process ragifix."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        probe = list(self._model.embed(["dimension probe"]))
        self._dimension = len(probe[0])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(self._model.embed([text]))[0].tolist()

    @property
    def dimension(self) -> int:
        return self._dimension
