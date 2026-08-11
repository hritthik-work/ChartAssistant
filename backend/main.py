from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import documents, health, ingestion, query
from .core.config import ROOT, Settings, get_settings
from .core.diagnostics import bind_request_id, configure_logging, log_event, reset_request_id
from .core.errors import AppError
from .models import ErrorResponse
from .services.agent import AgentProvider, GroundedPipeline, MafAgentProvider
from .services.azure import AzureEmbeddingService, AzureSearchService
from .services.ingestion import (
    ArtifactStore,
    EmbeddingService,
    IngestionService,
    SearchIndexService,
)
from .services.jobs import IngestionJobStore
from .services.retrieval import HostedSearch, RetrievalService

FRONTEND_DIR = ROOT / "frontend"


def _error_payload(exc: AppError, request_id: str | None = None) -> dict[str, Any]:
    return ErrorResponse(
        error="request_error" if exc.status_code < 500 else "service_error",
        request_id=request_id,
        category=exc.category,
        message=str(exc),
        retryable=exc.retryable,
        human_review_required=True,
    ).model_dump(mode="json")


def create_app(
    *,
    settings: Settings | None = None,
    embeddings: EmbeddingService | None = None,
    search_admin: SearchIndexService | None = None,
    search_query: HostedSearch | AzureSearchService | None = None,
    provider: AgentProvider | None = None,
    artifacts: ArtifactStore | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    logger = configure_logging(settings.log_level)
    artifacts = artifacts or ArtifactStore(settings.artifact_dir)
    if embeddings is None and settings.embeddings_ready:
        embeddings = AzureEmbeddingService(settings)
    if search_admin is None and settings.search_admin_ready:
        search_admin = AzureSearchService(settings, require_admin=True)
    if search_query is None and settings.search_query_ready:
        search_query = AzureSearchService(settings)
    if provider is None and settings.chat_ready:
        provider = MafAgentProvider(settings)

    ingestion_service = IngestionService(settings, embeddings, search_admin, artifacts)
    retrieval = RetrievalService(settings, artifacts, embeddings, search_query)
    pipeline = GroundedPipeline(
        retrieval,
        provider,
        settings.prompt_dir,
        prompt_version=settings.system_prompt_version,
    )
    jobs = IngestionJobStore()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        log_event(
            logger,
            logging.INFO,
            "application.startup",
            app_name=settings.app_name,
            chat_configured=settings.chat_ready,
            embeddings_configured=settings.embeddings_ready,
            search_query_configured=settings.search_query_ready,
            search_admin_configured=settings.search_admin_ready,
            search_index=settings.azure_search_index_name,
            prompt_version=settings.system_prompt_version,
            artifact_dir=str(settings.artifact_dir),
        )
        retrieval.refresh()
        try:
            yield
        finally:
            await jobs.shutdown()
            log_event(logger, logging.INFO, "application.shutdown")

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description=(
            "Question answering over synthetic patient charts using retrieval-augmented "
            "generation and Microsoft Agent Framework. Demonstration only."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.artifacts = artifacts
    app.state.ingestion = ingestion_service
    app.state.retrieval = retrieval
    app.state.pipeline = pipeline
    app.state.jobs = jobs

    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")
    app.include_router(query.router)
    app.include_router(ingestion.router)
    app.include_router(documents.router)
    app.include_router(health.router)

    @app.middleware("http")
    async def request_diagnostics(request: Request, call_next: Any):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        token = bind_request_id(request_id)
        started = time.perf_counter()
        log_event(
            logger,
            logging.INFO,
            "http.request.started",
            method=request.method,
            path=request.url.path,
            query_keys=sorted(request.query_params.keys()),
            content_type=request.headers.get("content-type"),
            content_length=request.headers.get("content-length"),
        )
        try:
            response = await call_next(request)
        except Exception as exc:
            log_event(
                logger,
                logging.ERROR,
                "http.request.unhandled_exception",
                error=exc,
                include_traceback=True,
                method=request.method,
                path=request.url.path,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            raise
        else:
            response.headers["X-Request-ID"] = request_id
            log_event(
                logger,
                logging.WARNING if response.status_code >= 400 else logging.INFO,
                "http.request.completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            return response
        finally:
            reset_request_id(token)

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        log_event(
            logger,
            logging.WARNING if exc.status_code < 500 else logging.ERROR,
            "application.error",
            error=exc,
            include_traceback=exc.status_code >= 500,
            category=exc.category,
            status_code=exc.status_code,
            retryable=exc.retryable,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(exc, request_id=request_id),
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        error = AppError(
            f"Request validation failed: {exc.errors()}",
            category="validation",
            status_code=422,
        )
        log_event(
            logger,
            logging.WARNING,
            "http.request.validation_failed",
            path=request.url.path,
            error_count=len(exc.errors()),
            errors=[
                {"type": item.get("type"), "location": item.get("loc")}
                for item in exc.errors()
            ],
        )
        return JSONResponse(
            status_code=422,
            content=_error_payload(error, request_id=getattr(request.state, "request_id", None)),
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        log_event(
            logger,
            logging.ERROR,
            "application.unexpected_error",
            error=exc,
            include_traceback=True,
            path=request.url.path,
        )
        error = AppError(
            "An unexpected internal error occurred",
            category="internal_error",
            status_code=500,
        )
        return JSONResponse(
            status_code=500,
            content=_error_payload(error, request_id=request_id),
        )

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "assets" / "favicon.svg", media_type="image/svg+xml")

    return app


app = create_app()
