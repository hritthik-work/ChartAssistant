from __future__ import annotations

from fastapi import Request

from ..core.config import Settings
from ..services.agent import GroundedPipeline
from ..services.ingestion import ArtifactStore, IngestionService
from ..services.jobs import IngestionJobStore
from ..services.retrieval import RetrievalService


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_artifacts(request: Request) -> ArtifactStore:
    return request.app.state.artifacts


def get_ingestion(request: Request) -> IngestionService:
    return request.app.state.ingestion


def get_retrieval(request: Request) -> RetrievalService:
    return request.app.state.retrieval


def get_pipeline(request: Request) -> GroundedPipeline:
    return request.app.state.pipeline


def get_jobs(request: Request) -> IngestionJobStore:
    return request.app.state.jobs
