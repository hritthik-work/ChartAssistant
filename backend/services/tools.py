from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from ..models import RetrievedChunk, ToolResult

ICD_TABLE = {
    "chronic kidney disease stage 3b": ("N18.32", "Chronic kidney disease, stage 3b"),
    "type 2 diabetes mellitus without complications": (
        "E11.9",
        "Type 2 diabetes mellitus without complications",
    ),
    "chronic obstructive pulmonary disease": (
        "J44.9",
        "Chronic obstructive pulmonary disease, unspecified",
    ),
    "essential hypertension": ("I10", "Essential (primary) hypertension"),
}
ALIASES = {
    "ckd stage 3b": "chronic kidney disease stage 3b",
    "stage 3b chronic kidney disease": "chronic kidney disease stage 3b",
    "copd": "chronic obstructive pulmonary disease",
    "type 2 diabetes": "type 2 diabetes mellitus without complications",
    "type 2 diabetes mellitus without documented complication": (
        "type 2 diabetes mellitus without complications"
    ),
    "hypertension": "essential hypertension",
}


class IcdLookupArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condition: str = Field(min_length=2, max_length=160)
    evidence_chunk_id: str = Field(min_length=3, max_length=160)


def _canonical(condition: str) -> str:
    normalized = " ".join(re.findall(r"[a-z0-9]+", condition.lower()))
    exact = ALIASES.get(normalized, normalized)
    if exact in ICD_TABLE:
        return exact
    for alias, canonical in sorted(ALIASES.items(), key=lambda item: -len(item[0])):
        if alias in normalized:
            return canonical
    for canonical in ICD_TABLE:
        if canonical in normalized:
            return canonical
    return normalized


def execute_icd_lookup(
    condition: str,
    evidence_chunk_id: str,
    retrieved: list[RetrievedChunk],
    *,
    simulate_timeout: bool = False,
) -> ToolResult:
    warning = (
        "Synthetic lookup only. A match does not establish HCC eligibility, coverage, "
        "medical necessity, submission validity, or a final coding decision; human review "
        "is required."
    )
    allowed = {chunk.chunk_id: chunk for chunk in retrieved}
    evidence = allowed.get(evidence_chunk_id)
    if evidence is None or evidence.source_role != "clinical_note":
        return ToolResult(
            tool="mock_icd10_lookup",
            query=condition,
            evidence_chunk_id=evidence_chunk_id,
            status="rejected",
            warning=(
                "Tool request rejected because its evidence was not a retrieved clinical chunk."
            ),
        )
    canonical = _canonical(condition)
    source = " ".join(evidence.content.lower().split())
    terms = canonical.split()
    if canonical not in source and not all(term in source for term in terms):
        return ToolResult(
            tool="mock_icd10_lookup",
            query=condition,
            evidence_chunk_id=evidence_chunk_id,
            status="rejected",
            warning="Tool request rejected because the condition is not supported by the chunk.",
        )
    if simulate_timeout:
        return ToolResult(
            tool="mock_icd10_lookup",
            query=condition,
            evidence_chunk_id=evidence_chunk_id,
            status="timeout",
            warning="Simulated tool timeout; no code was fabricated.",
        )
    match = ICD_TABLE.get(canonical)
    if not match:
        return ToolResult(
            tool="mock_icd10_lookup",
            query=condition,
            evidence_chunk_id=evidence_chunk_id,
            status="no_match",
            warning=warning,
        )
    return ToolResult(
        tool="mock_icd10_lookup",
        query=condition,
        evidence_chunk_id=evidence_chunk_id,
        status="matched",
        code=match[0],
        description=match[1],
        warning=warning,
    )
