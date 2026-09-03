"""Routes HTTP de ragifix.

Le contenu d'un document est reçu en corps de requête BRUT
(Content-Type: application/octet-stream), jamais en multipart/form-data :
FastAPI/Starlette bufferise les fichiers multipart au-delà d'un certain
seuil sur un fichier temporaire disque (SpooledTemporaryFile), ce qui
casserait silencieusement la contrainte "zéro écriture temporaire sur
disque" pour un contenu SharePoint relayé par ragifix-collector. Le corps
brut, lui, reste entièrement en mémoire (bytes) du début à la fin.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from .auth import TokenAuth
from .processing import UnsupportedFileTypeError
from .service import EmptyDocumentError, RagifixService

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Schémas
# --------------------------------------------------------------------------- #

class OriginInfo(BaseModel):
    """Localisateur le plus rapide pour ouvrir le document source (lien
    SharePoint, chemin local, etc.) — voir metadata["origin"], normalisé ici
    en champ de premier niveau pour ne pas obliger les clients à connaître la
    structure interne du blob metadata."""

    kind: str
    uri: str
    label: str = ""


class DocumentResponse(BaseModel):
    doc_id: str
    extension: str
    chunk_count: int
    metadata: dict
    updated_at: str
    origin: OriginInfo | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=100)
    filters: dict | None = None


class QueryResultItem(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: dict
    origin: OriginInfo | None = None


class QueryResponse(BaseModel):
    results: list[QueryResultItem]


class SourceResponse(BaseModel):
    name: str
    description: str
    enabled: bool
    updated_at: str


class SourcesListResponse(BaseModel):
    sources: list[SourceResponse]


class SetSourceRequest(BaseModel):
    name: str
    description: str = ""
    enabled: bool = True


def _extract_origin(metadata: dict) -> OriginInfo | None:
    raw = metadata.get("origin")
    if not isinstance(raw, dict):
        return None
    try:
        return OriginInfo(**raw)
    except (TypeError, ValueError):
        return None


def _to_response(record) -> DocumentResponse:
    return DocumentResponse(
        doc_id=record.doc_id,
        extension=record.extension,
        chunk_count=record.chunk_count,
        metadata=record.metadata,
        updated_at=record.updated_at,
        origin=_extract_origin(record.metadata),
    )


def _parse_metadata_param(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Paramètre 'metadata' invalide (JSON attendu): {exc}")
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="Paramètre 'metadata' doit être un objet JSON")
    return value


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #

def build_router(service: RagifixService, auth: TokenAuth, default_top_k: int, max_document_size_mb: int) -> APIRouter:
    router = APIRouter()
    max_bytes = max_document_size_mb * 1024 * 1024

    @router.put("/documents/{doc_id:path}", response_model=DocumentResponse, dependencies=[Depends(auth)])
    async def put_document(
        doc_id: str,
        request: Request,
        extension: str = Query(..., description="Extension du fichier (sans le point), ex: pdf, docx, txt"),
        metadata: str | None = Query(default=None, description="Métadonnées, encodées en JSON"),
    ) -> DocumentResponse:
        content_length = request.headers.get("content-length")
        if content_length is not None and int(content_length) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Document trop volumineux (max {max_document_size_mb} Mo)",
            )

        content = await request.body()
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Document trop volumineux (max {max_document_size_mb} Mo)",
            )
        if not content:
            raise HTTPException(status_code=400, detail="Corps de requête vide")

        meta_dict = _parse_metadata_param(metadata)

        try:
            record = await service.ingest_document(doc_id, content, extension, meta_dict)
        except UnsupportedFileTypeError as exc:
            raise HTTPException(status_code=415, detail=str(exc))
        except EmptyDocumentError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception:
            logger.exception("Échec de l'ingestion du document '%s'", doc_id)
            raise HTTPException(status_code=500, detail="Erreur interne lors de l'ingestion du document")

        return _to_response(record)

    @router.delete("/documents/{doc_id:path}", status_code=204, dependencies=[Depends(auth)])
    async def delete_document(doc_id: str) -> None:
        try:
            existed = await service.delete_document(doc_id)
        except Exception:
            logger.exception("Échec de la suppression du document '%s'", doc_id)
            raise HTTPException(status_code=500, detail="Erreur interne lors de la suppression du document")
        if not existed:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' inconnu")

    @router.get("/documents/{doc_id:path}", response_model=DocumentResponse, dependencies=[Depends(auth)])
    async def get_document(doc_id: str) -> DocumentResponse:
        record = await service.get_document(doc_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' inconnu")
        return _to_response(record)

    @router.get("/documents", response_model=DocumentListResponse, dependencies=[Depends(auth)])
    async def list_documents(
        prefix: str | None = Query(default=None, description="Filtre les doc_id commençant par ce préfixe"),
    ) -> DocumentListResponse:
        records = await service.list_documents(prefix=prefix)
        return DocumentListResponse(documents=[_to_response(r) for r in records])

    @router.post("/query", response_model=QueryResponse, dependencies=[Depends(auth)])
    async def query(body: QueryRequest) -> QueryResponse:
        top_k = body.top_k or default_top_k
        try:
            results = await service.query(body.query, top_k=top_k, filters=body.filters)
        except Exception:
            logger.exception("Échec de la requête d'interrogation")
            raise HTTPException(status_code=500, detail="Erreur interne lors de la recherche")

        return QueryResponse(
            results=[
                QueryResultItem(
                    chunk_id=r.chunk_id,
                    doc_id=r.doc_id,
                    text=r.text,
                    score=r.score,
                    metadata=r.metadata,
                    origin=_extract_origin(r.metadata),
                )
                for r in results
            ]
        )

    @router.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @router.post("/sources", dependencies=[Depends(auth)])
    async def set_sources(body: list[SetSourceRequest]) -> list[SourceResponse]:
        try:
            results = []
            for src in body:
                await service.set_source(src.name, src.description, src.enabled)
                results.append(SourceResponse(name=src.name, description=src.description, enabled=src.enabled, updated_at=datetime.now(timezone.utc).isoformat()))
            return results
        except Exception:
            logger.exception("Échec de la mise à jour des sources")
            raise HTTPException(status_code=500, detail="Erreur interne lors de la mise à jour des sources")

    @router.get("/sources", response_model=SourcesListResponse, dependencies=[Depends(auth)])
    async def get_sources() -> SourcesListResponse:
        try:
            sources = await service.get_sources()
            return SourcesListResponse(sources=[SourceResponse(**s) for s in sources])
        except Exception:
            logger.exception("Échec de la récupération des sources")
            raise HTTPException(status_code=500, detail="Erreur interne lors de la récupération des sources")

    return router
