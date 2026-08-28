"""Tests du parsing et du chunking des documents."""

from __future__ import annotations

import pytest

from ragifix.processing import (
    ChunkerConfig,
    TokenChunker,
    UnsupportedFileTypeError,
    build_chunker,
    parse_document,
)


# -- parse_document -------------------------------------------------------

def test_parse_document_txt():
    text = parse_document(b"hello world", "txt")
    assert text == "hello world"


def test_parse_document_md():
    content = b"# Titre\nLigne 1"
    assert parse_document(content, "md") == "# Titre\nLigne 1"


def test_parse_document_replaces_bad_utf8():
    text = parse_document(b"\xff\xfe bad", "txt")
    assert isinstance(text, str)
    assert len(text) > 0


def test_parse_document_unsupported_raises():
    with pytest.raises(UnsupportedFileTypeError):
        parse_document(b"data", "zip")


# -- TokenChunker ---------------------------------------------------------

def test_chunker_empty_text():
    chunker = TokenChunker(ChunkerConfig(chunk_size=512, chunk_overlap=64))
    assert chunker.chunk("") == []
    assert chunker.chunk("   ") == []


def test_chunker_small_text_single_chunk():
    chunker = TokenChunker(ChunkerConfig(chunk_size=512, chunk_overlap=64))
    chunks = chunker.chunk("hello world")
    assert len(chunks) == 1
    assert chunks[0] == "hello world"


def test_chunker_covers_full_text():
    text = " ".join(f"word{i}" for i in range(100))
    # overlap=0 : les fenêtres sont contiguës, leur concaténation reforme le texte.
    chunker = TokenChunker(ChunkerConfig(chunk_size=8, chunk_overlap=0))
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2
    assert "".join(chunks) == text


def test_chunker_overlap_must_be_below_size():
    with pytest.raises(ValueError):
        TokenChunker(ChunkerConfig(chunk_size=10, chunk_overlap=10))


# -- build_chunker --------------------------------------------------------

def test_build_chunker_token():
    chunker = build_chunker(strategy="token", chunk_size=512, chunk_overlap=64)
    assert isinstance(chunker, TokenChunker)


def test_build_chunker_unknown_strategy():
    with pytest.raises(ValueError):
        build_chunker(strategy="rule", chunk_size=512, chunk_overlap=64)
