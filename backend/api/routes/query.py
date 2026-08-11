from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from ...core.diagnostics import log_event
from ...models import QueryRequest, QueryResponse
from ...services.agent import GroundedPipeline
from ..dependencies import get_pipeline

LOGGER = logging.getLogger("healthchat.api.query")
router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    pipeline: GroundedPipeline = Depends(get_pipeline),
) -> QueryResponse:
    log_event(LOGGER, logging.INFO, "query.started", query_length=len(request.query))
    return await pipeline.run(request.query)
