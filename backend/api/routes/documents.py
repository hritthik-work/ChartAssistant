from __future__ import annotations

from fastapi import APIRouter, Depends

from ...models import DocumentSummary
from ...services.ingestion import ArtifactStore
from ..dependencies import get_artifacts

router = APIRouter(tags=["documents"])


@router.get("/documents", response_model=list[DocumentSummary])
async def documents(
    artifacts: ArtifactStore = Depends(get_artifacts),
) -> list[DocumentSummary]:
    groups: dict[str, list] = {}
    for chunk in artifacts.load_chunks():
        groups.setdefault(chunk.document_id, []).append(chunk)
    return [
        DocumentSummary(
            document_id=items[0].document_id,
            title=items[0].title,
            source_role=items[0].source_role,
            document_type=items[0].document_type,
            patient_id=items[0].patient_id,
            encounter_date=items[0].encounter_date,
            effective_date=items[0].effective_date,
            chunks=len(items),
        )
        for _, items in sorted(groups.items())
    ]
