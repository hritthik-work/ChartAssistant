from __future__ import annotations

import asyncio
import time
from typing import Any

from openai import AsyncAzureOpenAI

from ..core.config import Settings
from .azure import AzureEmbeddingService, AzureSearchService


def _configuration(service: str, configured: bool, **metadata: Any) -> dict[str, Any]:
    return {
        "service": service,
        "status": "configured" if configured else "not_configured",
        "configured": configured,
        "reachable": None,
        "check": "configuration_only",
        **metadata,
    }


async def check_chat(settings: Settings, *, deep: bool = False) -> dict[str, Any]:
    if not settings.chat_ready:
        return _configuration("azure_openai_chat", False)
    if not deep:
        return _configuration(
            "azure_openai_chat", True, deployment=settings.azure_openai_chat_deployment
        )
    started = time.perf_counter()
    try:
        client = AsyncAzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            timeout=15.0,
            max_retries=0,
        )
        response = await client.chat.completions.create(
            model=str(settings.azure_openai_chat_deployment),
            messages=[{"role": "user", "content": "Reply with only OK."}],
            max_completion_tokens=16,
        )
        if not response.choices:
            raise RuntimeError("empty response")
        return {
            "service": "azure_openai_chat",
            "status": "ok",
            "configured": True,
            "reachable": True,
            "check": "live_completion",
            "deployment": settings.azure_openai_chat_deployment,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    except Exception as exc:
        return {
            "service": "azure_openai_chat",
            "status": "error",
            "configured": True,
            "reachable": False,
            "error_code": type(exc).__name__,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }


async def check_embeddings(settings: Settings, *, deep: bool = False) -> dict[str, Any]:
    if not settings.embeddings_ready:
        return _configuration("azure_openai_embeddings", False)
    if not deep:
        return _configuration(
            "azure_openai_embeddings",
            True,
            deployment=settings.azure_openai_embedding_deployment,
        )
    started = time.perf_counter()
    try:
        service = AzureEmbeddingService(settings)
        vectors, tokens = await service.embed(["HealthChatBot health probe"])
        if len(vectors) != 1 or not vectors[0]:
            raise RuntimeError("empty response")
        return {
            "service": "azure_openai_embeddings",
            "status": "ok",
            "configured": True,
            "reachable": True,
            "check": "live_embedding",
            "deployment": service.deployment,
            "vector_dimensions": len(vectors[0]),
            "input_tokens": tokens,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    except Exception as exc:
        return {
            "service": "azure_openai_embeddings",
            "status": "error",
            "configured": True,
            "reachable": False,
            "error_code": type(exc).__name__,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }


async def check_search_query(settings: Settings, *, deep: bool = False) -> dict[str, Any]:
    if not settings.search_query_ready:
        return _configuration(
            "azure_ai_search_query",
            False,
            credential="query_key",
            index_name=settings.azure_search_index_name,
        )
    if not deep:
        return _configuration(
            "azure_ai_search_query",
            True,
            credential="query_key",
            index_name=settings.azure_search_index_name,
        )
    started = time.perf_counter()
    try:
        service = AzureSearchService(settings)
        metadata = await service.probe()
        return {
            "service": "azure_ai_search_query",
            "status": "ok",
            "configured": True,
            "reachable": True,
            "check": "live_query",
            "credential": "query_key",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            **metadata,
        }
    except Exception as exc:
        return {
            "service": "azure_ai_search_query",
            "status": "error",
            "configured": True,
            "reachable": False,
            "credential": "query_key",
            "index_name": settings.azure_search_index_name,
            "error_code": type(exc).__name__,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }


async def check_search_ingestion(settings: Settings, *, deep: bool = False) -> dict[str, Any]:
    if not settings.search_admin_ready:
        return _configuration(
            "azure_ai_search_ingestion",
            False,
            credential="admin_key",
            index_name=settings.azure_search_index_name,
        )
    if not deep:
        return _configuration(
            "azure_ai_search_ingestion",
            True,
            credential="admin_key",
            index_name=settings.azure_search_index_name,
        )
    started = time.perf_counter()
    try:
        service = AzureSearchService(settings, require_admin=True)
        metadata = await service.probe_admin()
        return {
            "service": "azure_ai_search_ingestion",
            "status": "ok",
            "configured": True,
            "reachable": True,
            "check": "live_admin_access",
            "credential": "admin_key",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            **metadata,
        }
    except Exception as exc:
        return {
            "service": "azure_ai_search_ingestion",
            "status": "error",
            "configured": True,
            "reachable": False,
            "credential": "admin_key",
            "index_name": settings.azure_search_index_name,
            "error_code": type(exc).__name__,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }


async def check_search(settings: Settings, *, deep: bool = False) -> dict[str, Any]:
    query, ingestion = await asyncio.gather(
        check_search_query(settings, deep=deep),
        check_search_ingestion(settings, deep=deep),
    )
    failed = [
        check["service"]
        for check in (query, ingestion)
        if check["status"] in {"error", "not_configured"}
    ]
    return {
        "service": "azure_ai_search",
        "status": "ok" if not failed else "degraded",
        "configured": query["configured"] and ingestion["configured"],
        "reachable": (query["reachable"] and ingestion["reachable"] if deep else None),
        "check": "live" if deep else "configuration_only",
        "index_name": settings.azure_search_index_name,
        "operations": {"query": query, "ingestion": ingestion},
        "failed_operations": failed,
    }


async def check_services(settings: Settings, *, deep: bool = False) -> dict[str, Any]:
    results = await asyncio.gather(
        check_chat(settings, deep=deep),
        check_embeddings(settings, deep=deep),
        check_search_query(settings, deep=deep),
        check_search_ingestion(settings, deep=deep),
    )
    services = {item["service"]: item for item in results}
    failed = [
        name for name, item in services.items() if item["status"] in {"error", "not_configured"}
    ]
    return {
        "status": "ok" if not failed else "degraded",
        "check": "live" if deep else "configuration_only",
        "services": services,
        "failed_services": failed,
    }
