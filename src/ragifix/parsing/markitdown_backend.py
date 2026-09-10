from __future__ import annotations

import io


class MarkitdownBackend:
    """Backend de parsing basé sur markitdown (extra optionnel `ragifix[markitdown]`).

    `convert_stream` travaille entièrement en mémoire (aucun fichier
    temporaire créé) — cohérent avec la contrainte "zéro écriture disque"
    de ragifix.processing.
    """

    def __init__(self, enable_plugins: bool = False):
        from markitdown import MarkItDown

        self._md = MarkItDown(enable_plugins=enable_plugins)

    def parse(self, content: bytes, extension: str, filename: str) -> str:
        from markitdown import StreamInfo

        result = self._md.convert_stream(
            io.BytesIO(content),
            stream_info=StreamInfo(extension=f".{extension}", filename=filename),
        )
        return result.markdown
