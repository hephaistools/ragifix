from __future__ import annotations

import io


class DoclingBackend:
    """Backend de parsing basé sur docling (extra optionnel `ragifix[docling]`)."""

    def __init__(self):
        import docling  # noqa: F401  — probe d'import : échoue vite si l'extra n'est pas installé

    def parse(self, content: bytes, extension: str, filename: str) -> str:
        from docling.datamodel.base_models import DocumentStream
        from docling.document_converter import DocumentConverter

        name = filename if filename.lower().endswith(f".{extension}") else f"{filename}.{extension}"
        stream = DocumentStream(name=name, stream=io.BytesIO(content))
        converter = DocumentConverter()
        result = converter.convert(stream)
        return result.document.export_to_markdown()
