from __future__ import annotations

import re

from ..models import ModelCandidate, RetrievedChunk, ToolResult

CODE_PATTERN = re.compile(r"\b[A-Z]\d{2}(?:\.\d{1,4})?\b")
INLINE_REFERENCE = re.compile(r"\[([A-Za-z0-9_-]+-chunk-\d{3})\]")


def validate_grounding(
    candidate: ModelCandidate,
    retrieved: list[RetrievedChunk],
    tool_result: ToolResult | None,
) -> None:
    allowed = {chunk.chunk_id: chunk for chunk in retrieved}
    if candidate.status == "answered" and not candidate.citations:
        raise ValueError("answered output requires at least one citation")
    if candidate.status in {"partial", "refused"} and not candidate.missing_information:
        raise ValueError("partial/refused output requires missing information")
    if candidate.citations and not retrieved:
        raise ValueError("output cited evidence when no chunks were retrieved")
    citation_ids = {citation.chunk_id for citation in candidate.citations}
    inline_ids = set(INLINE_REFERENCE.findall(candidate.answer))
    if inline_ids != citation_ids:
        missing_inline = sorted(citation_ids - inline_ids)
        uncited_inline = sorted(inline_ids - citation_ids)
        raise ValueError(
            "inline references must match citation objects exactly: "
            f"missing inline {missing_inline}, uncited inline {uncited_inline}"
        )
    for citation in candidate.citations:
        chunk = allowed.get(citation.chunk_id)
        if chunk is None:
            raise ValueError(f"citation {citation.chunk_id!r} was not retrieved")
        if citation.document_id != chunk.document_id:
            raise ValueError(f"citation {citation.chunk_id!r} has the wrong document ID")
        quote = re.sub(r"\s+", " ", citation.quote).strip()
        source = re.sub(r"\s+", " ", chunk.content).strip()
        if quote not in source:
            raise ValueError(f"quote for {citation.chunk_id!r} is not exact source text")
    codes = set(CODE_PATTERN.findall(candidate.answer))
    if codes and (tool_result is None or tool_result.status != "matched"):
        raise ValueError("answer contains a code without a successful lookup result")
    if codes and tool_result and tool_result.code not in codes:
        raise ValueError("answer contains a code not returned by the lookup tool")


def workflow_confidence(
    candidate: ModelCandidate,
    retrieved: list[RetrievedChunk],
    tool_result: ToolResult | None,
    *,
    repaired: bool,
    fallback: bool,
) -> tuple[float, str, str]:
    by_id = {chunk.chunk_id: chunk for chunk in retrieved}
    cited_scores = [
        by_id[item.chunk_id].score for item in candidate.citations if item.chunk_id in by_id
    ]
    score = min(cited_scores) if cited_scores else 0.0
    reasons: list[str] = []
    if candidate.status == "partial":
        score *= 0.7
        reasons.append("the answer is partial")
    elif candidate.status == "refused":
        score = 0.0
        reasons.append("the evidence did not support an answer")
    if repaired:
        score -= 0.15
        reasons.append("the model output required repair")
    if tool_result and tool_result.status != "matched":
        score -= 0.15
        reasons.append(f"the lookup ended with {tool_result.status}")
    if fallback:
        score -= 0.15
        reasons.append("retrieval used a visible local fallback")
    score = round(max(0.0, min(1.0, score)), 2)
    level = "high" if score >= 0.75 else "medium" if score >= 0.45 else "low"
    rationale = (
        "Workflow confidence is based on validated cited retrieval scores"
        + ("; " + "; ".join(reasons) if reasons else " with no repair or fallback penalty")
        + ". It is not a probability of clinical correctness."
    )
    return score, level, rationale
