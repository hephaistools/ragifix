"""Authentification de l'API ragifix.

Un token bearer simple, comparé en temps constant (hmac.compare_digest).
Ce token protège autant l'intégrité de l'index (ajout/suppression) que sa
confidentialité (interrogation).
"""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, status


class TokenAuth:
    def __init__(self, expected_token: str):
        if not expected_token:
            raise ValueError("TokenAuth: le token attendu ne peut pas être vide")
        self._expected_token = expected_token

    async def __call__(self, authorization: str = Header(default="")) -> None:
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="En-tête Authorization manquant ou mal formé (attendu: 'Bearer <token>')",
            )
        provided = authorization.removeprefix("Bearer ").strip()
        if not hmac.compare_digest(provided, self._expected_token):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalide")
