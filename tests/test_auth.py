"""Tests de l'authentification par token."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException
from starlette.status import HTTP_401_UNAUTHORIZED

from ragifix.auth import TokenAuth


def test_empty_token_raises():
    with pytest.raises(ValueError):
        TokenAuth(expected_token="")


def test_valid_bearer_ok():
    auth = TokenAuth(expected_token="secret")
    # Ne doit pas lever.
    asyncio.run(auth("Bearer secret"))


def test_wrong_token_raises_401():
    auth = TokenAuth(expected_token="secret")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(auth("Bearer wrong"))
    assert exc.value.status_code == HTTP_401_UNAUTHORIZED


def test_missing_bearer_prefix_raises_401():
    auth = TokenAuth(expected_token="secret")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(auth("secret"))
    assert exc.value.status_code == HTTP_401_UNAUTHORIZED


def test_malformed_header_raises_401():
    auth = TokenAuth(expected_token="secret")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(auth("Basic abc"))
    assert exc.value.status_code == HTTP_401_UNAUTHORIZED


def test_empty_header_raises_401():
    auth = TokenAuth(expected_token="secret")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(auth(""))
    assert exc.value.status_code == HTTP_401_UNAUTHORIZED


def test_bearer_with_extra_ws_stripped():
    auth = TokenAuth(expected_token="secret")
    # « Bearer   secret » → le strip retire les espaces superflus.
    asyncio.run(auth("Bearer   secret"))
