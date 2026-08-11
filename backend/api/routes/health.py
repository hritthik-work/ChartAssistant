from __future__ import annotations

from collections import Counter
from typing import Any

from fastapi import APIRouter, Depends, Query

from ...core.config import Settings
from ...services.health import (
    check_chat,
    check_embeddings,
    check_search,
    check_search_ingestion,
    check_search_query,
    check_services,
)
from ...services.ingestion import ArtifactStore
from ...services.retrieval import RetrievalService
from ..dependencies import get_artifacts, get_retrieval, get_settings

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health(
    settings: Settings = Depends(get_settings),
    artifacts: ArtifactStore = Depends(get_artifacts),
    retrieval: RetrievalService = Depends(get_retrieval),
) -> dict[str, Any]:
    chunks = artifacts.load_chunks()
    roles = Counter(chunk.source_role for chunk in chunks)
    return {
        "status": "ok",
        "application": settings.app_name,
        "prompt_version": settings.system_prompt_version,
        "synthetic_only": True,
        "model_configured": settings.chat_ready,
        "embeddings_configured": settings.embeddings_ready,
        "hosted_search_configured": settings.search_query_ready,
        "search_admin_configured": settings.search_admin_ready,
        "search_index_name": settings.azure_search_index_name,
        "retrieval_backend": retrieval.last_backend,
        "retrieval_warning": retrieval.last_warning,
        "documents": len({chunk.document_id for chunk in chunks}),
        "patients": len({chunk.patient_id for chunk in chunks if chunk.patient_id}),
        "chunks": len(chunks),
        "source_roles": dict(roles),
        "health_endpoints": {
            "aggregate": "/health/services?deep=true",
            "chat": "/health/azure-openai/chat?deep=true",
            "embeddings": "/health/azure-openai/embeddings?deep=true",
            "search": "/health/azure-search?deep=true",
            "search_query": "/health/azure-search/query?deep=true",
            "search_ingestion": "/health/azure-search/ingestion?deep=true",
        },
    }


@router.get("/azure-openai/chat")
async def chat_health(
    deep: bool = Query(False),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    return await check_chat(settings, deep=deep)


@router.get("/azure-openai/embeddings")
async def embeddings_health(
    deep: bool = Query(False),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    return await check_embeddings(settings, deep=deep)


@router.get("/azure-search")
async def search_health(
    deep: bool = Query(False),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    return await check_search(settings, deep=deep)


@router.get("/azure-search/query")
async def search_query_health(
    deep: bool = Query(False),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    return await check_search_query(settings, deep=deep)


@router.get("/azure-search/ingestion")
async def search_ingestion_health(
    deep: bool = Query(False),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    return await check_search_ingestion(settings, deep=deep)


@router.get("/services")
async def services_health(
    deep: bool = Query(False),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    return await check_services(settings, deep=deep)
