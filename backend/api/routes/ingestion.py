from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, UploadFile

from ...core.config import Settings
from ...core.diagnostics import log_event
from ...core.errors import AppError, ValidationAppError
from ...models import IngestionJobCreated, IngestionJobStatus, IngestionResponse
from ...services.ingestion import IngestionService, UploadPayload
from ...services.jobs import IngestionJobStore
from ...services.retrieval import RetrievalService
from ..dependencies import get_ingestion, get_jobs, get_retrieval, get_settings

LOGGER = logging.getLogger("healthchat.api.ingestion")
router = APIRouter(prefix="/ingestion", tags=["ingestion"])


async def _read_uploads(files: list[UploadFile], settings: Settings) -> list[UploadPayload]:
    payloads: list[UploadPayload] = []
    total_bytes = 0
    for file in files:
        content = await file.read(settings.max_file_bytes + 1)
        if len(content) > settings.max_file_bytes:
            raise ValidationAppError(
                f"{file.filename or 'unnamed'}: file exceeds the configured size limit"
            )
        total_bytes += len(content)
        if total_bytes > settings.max_total_upload_bytes:
            raise ValidationAppError("Total upload size exceeds the configured limit")
        payloads.append(
            UploadPayload(
                filename=file.filename or "unnamed",
                content=content,
                content_type=file.content_type,
            )
        )
    log_event(
        LOGGER,
        logging.INFO,
        "ingestion.upload.read",
        file_count=len(payloads),
        total_bytes=total_bytes,
    )
    return payloads


@router.post("", response_model=IngestionResponse)
async def ingest(
    files: list[UploadFile] = File(..., description="1-25 synthetic TXT, MD, or PDF files"),
    settings: Settings = Depends(get_settings),
    ingestion: IngestionService = Depends(get_ingestion),
    retrieval: RetrievalService = Depends(get_retrieval),
) -> IngestionResponse:
    log_event(
        LOGGER,
        logging.INFO,
        "ingestion.upload.started",
        file_count=len(files),
        filenames=[file.filename or "unnamed" for file in files],
    )
    response = await ingestion.ingest(await _read_uploads(files, settings))
    retrieval.refresh()
    log_event(
        LOGGER,
        logging.INFO,
        "ingestion.upload.completed",
        ingestion_id=response.ingestion_id,
        status=response.status,
        documents_indexed=response.documents_indexed,
        chunks_indexed=response.chunks_indexed,
        latency_ms=response.latency_ms,
    )
    return response


@router.post("/jobs", response_model=IngestionJobCreated, status_code=202)
async def create_ingestion_job(
    patient_reference: str = Form(..., min_length=2, max_length=80),
    synthetic_confirmed: bool = Form(...),
    files: list[UploadFile] = File(
        ..., description="1-25 synthetic PDF, DOCX, TXT, or Markdown patient charts"
    ),
    settings: Settings = Depends(get_settings),
    ingestion: IngestionService = Depends(get_ingestion),
    retrieval: RetrievalService = Depends(get_retrieval),
    jobs: IngestionJobStore = Depends(get_jobs),
) -> IngestionJobCreated:
    if not synthetic_confirmed:
        raise ValidationAppError("Confirm that these files contain synthetic demo data")
    return await jobs.create(
        ingestion,
        await _read_uploads(files, settings),
        patient_reference=patient_reference,
        synthetic_confirmed=synthetic_confirmed,
        on_complete=retrieval.refresh,
    )


@router.get("/jobs/{job_id}", response_model=IngestionJobStatus)
async def ingestion_job(
    job_id: str,
    jobs: IngestionJobStore = Depends(get_jobs),
) -> IngestionJobStatus:
    status = await jobs.get(job_id)
    if status is None:
        raise AppError("Upload job was not found", category="validation", status_code=404)
    return status
