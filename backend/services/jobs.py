from __future__ import annotations

import asyncio
import logging
import uuid
from collections import OrderedDict
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from ..core.diagnostics import log_event
from ..core.errors import AppError
from ..models import (
    IngestionJobCreated,
    IngestionJobError,
    IngestionJobStatus,
)
from .ingestion import IngestionService, UploadPayload

LOGGER = logging.getLogger("healthchat.jobs")


def _now() -> str:
    return datetime.now(UTC).isoformat()


class IngestionJobStore:
    """Small in-memory job registry for the single-process demonstration app."""

    def __init__(self, *, limit: int = 50) -> None:
        self.limit = limit
        self._jobs: OrderedDict[str, IngestionJobStatus] = OrderedDict()
        self._tasks: set[asyncio.Task[None]] = set()
        self._lock = asyncio.Lock()

    async def create(
        self,
        ingestion: IngestionService,
        uploads: Sequence[UploadPayload],
        *,
        patient_reference: str,
        synthetic_confirmed: bool,
        on_complete: Callable[[], None],
    ) -> IngestionJobCreated:
        job_id = str(uuid.uuid4())
        created = _now()
        status = IngestionJobStatus(
            job_id=job_id,
            status="queued",
            stage="queued",
            progress=0,
            message="Upload received and waiting to start",
            created_at=created,
            updated_at=created,
        )
        async with self._lock:
            while len(self._jobs) >= self.limit:
                removable = next(
                    (
                        existing_id
                        for existing_id, existing in self._jobs.items()
                        if existing.status in {"completed", "failed"}
                    ),
                    None,
                )
                if removable is None:
                    raise AppError(
                        "Too many chart uploads are already processing",
                        category="rate_limit",
                        status_code=429,
                        retryable=True,
                    )
                self._jobs.pop(removable)
            self._jobs[job_id] = status
        task = asyncio.create_task(
            self._run(
                job_id,
                ingestion,
                list(uploads),
                patient_reference=patient_reference,
                synthetic_confirmed=synthetic_confirmed,
                on_complete=on_complete,
            ),
            name=f"ingestion-{job_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return IngestionJobCreated(
            job_id=job_id,
            status_url=f"/ingestion/jobs/{job_id}",
        )

    async def get(self, job_id: str) -> IngestionJobStatus | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def shutdown(self) -> None:
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _update(self, job_id: str, **changes: object) -> None:
        async with self._lock:
            current = self._jobs[job_id]
            self._jobs[job_id] = current.model_copy(
                update={**changes, "updated_at": _now()}
            )

    async def _run(
        self,
        job_id: str,
        ingestion: IngestionService,
        uploads: list[UploadPayload],
        *,
        patient_reference: str,
        synthetic_confirmed: bool,
        on_complete: Callable[[], None],
    ) -> None:
        async def report(stage: str, progress: int, message: str) -> None:
            await self._update(
                job_id,
                status="processing" if stage != "completed" else "completed",
                stage=stage,
                progress=progress,
                message=message,
            )

        try:
            result = await ingestion.ingest(
                uploads,
                patient_reference=patient_reference,
                synthetic_confirmed=synthetic_confirmed,
                progress=report,
            )
            on_complete()
            await self._update(
                job_id,
                status="completed",
                stage="completed",
                progress=100,
                message="Charts are ready for questions",
                result=result,
            )
        except AppError as exc:
            await self._update(
                job_id,
                status="failed",
                stage="failed",
                message=str(exc),
                error=IngestionJobError(
                    category=exc.category,
                    message=str(exc),
                    retryable=exc.retryable,
                ),
            )
            log_event(
                LOGGER,
                logging.WARNING if exc.status_code < 500 else logging.ERROR,
                "ingestion_job.failed",
                job_id=job_id,
                category=exc.category,
                retryable=exc.retryable,
                error=exc,
            )
        except Exception as exc:
            await self._update(
                job_id,
                status="failed",
                stage="failed",
                message="The chart upload failed unexpectedly",
                error=IngestionJobError(
                    category="internal_error",
                    message="The chart upload failed unexpectedly",
                    retryable=False,
                ),
            )
            log_event(
                LOGGER,
                logging.ERROR,
                "ingestion_job.failed",
                job_id=job_id,
                error=exc,
                include_traceback=True,
            )
        finally:
            uploads.clear()
