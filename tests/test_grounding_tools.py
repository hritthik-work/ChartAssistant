from __future__ import annotations

import pytest

from backend.models import Citation, ModelCandidate, RetrievedChunk
from backend.services.grounding import validate_grounding, workflow_confidence
from backend.services.tools import execute_icd_lookup


def _retrieved(chunk) -> RetrievedChunk:
    return RetrievedChunk(
        **chunk.model_dump(),
        score=0.9,
        lexical_score=2.0,
        reranker_score=3.6,
    )


def test_exact_grounding_and_inline_reference(curated_corpus):
    _, chunks = curated_corpus
    chunk = _retrieved(next(item for item in chunks if item.patient_id == "SYN-P004"))
    quote = chunk.content.split("\n", 1)[1].split(".", 1)[0] + "."
    candidate = ModelCandidate(
        status="answered",
        intent="clinical_evidence",
        answer=f"The source states this directly [{chunk.chunk_id}].",
        citations=[
            Citation(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                quote=quote,
                supports="The cited patient fact.",
            )
        ],
        missing_information=[],
    )
    validate_grounding(candidate, [chunk], None)
    invalid = candidate.model_copy(
        update={"citations": [candidate.citations[0].model_copy(update={"quote": "invented"})]}
    )
    with pytest.raises(ValueError, match="not exact"):
        validate_grounding(invalid, [chunk], None)

    extra_reference = candidate.model_copy(
        update={"answer": f"{candidate.answer} [DOC-CLIN-P999-999-chunk-001]"}
    )
    with pytest.raises(ValueError, match="uncited inline"):
        validate_grounding(extra_reference, [chunk], None)


def test_icd_lookup_requires_retrieved_clinical_evidence(curated_corpus):
    _, chunks = curated_corpus
    chunk = _retrieved(
        next(
            item
            for item in chunks
            if "chronic kidney disease" in item.content.lower()
            and "stage 3b" in item.content.lower()
        )
    )
    matched = execute_icd_lookup("CKD stage 3b", chunk.chunk_id, [chunk])
    punctuation_variant = execute_icd_lookup(
        "chronic kidney disease, stage 3b", chunk.chunk_id, [chunk]
    )
    rejected = execute_icd_lookup("COPD", chunk.chunk_id, [chunk])
    assert matched.status == "matched" and matched.code == "N18.32"
    assert punctuation_variant.status == "matched" and punctuation_variant.code == "N18.32"
    assert rejected.status == "rejected" and rejected.code is None


def test_icd_lookup_timeout_never_fabricates_a_code(curated_corpus):
    _, chunks = curated_corpus
    chunk = _retrieved(
        next(
            item
            for item in chunks
            if "chronic kidney disease" in item.content.lower()
            and "stage 3b" in item.content.lower()
        )
    )

    result = execute_icd_lookup(
        "CKD stage 3b", chunk.chunk_id, [chunk], simulate_timeout=True
    )

    assert result.status == "timeout"
    assert result.code is None
    assert "no code was fabricated" in result.warning


def test_code_claim_requires_successful_tool(curated_corpus):
    _, chunks = curated_corpus
    chunk = _retrieved(next(item for item in chunks if item.patient_id == "SYN-P004"))
    candidate = ModelCandidate(
        status="answered",
        intent="icd_lookup",
        answer=f"The code is N18.32 [{chunk.chunk_id}].",
        citations=[
            Citation(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                quote=chunk.content.splitlines()[1],
                supports="condition",
            )
        ],
        missing_information=[],
    )
    with pytest.raises(ValueError, match="without a successful"):
        validate_grounding(candidate, [chunk], None)


def test_confidence_penalizes_fallback_and_repair(curated_corpus):
    _, chunks = curated_corpus
    chunk = _retrieved(next(item for item in chunks if item.patient_id == "SYN-P004"))
    candidate = ModelCandidate(
        status="answered",
        intent="clinical_evidence",
        answer=f"Supported [{chunk.chunk_id}].",
        citations=[
            Citation(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                quote=chunk.content.splitlines()[1],
                supports="fact",
            )
        ],
        missing_information=[],
    )
    score, level, reason = workflow_confidence(
        candidate, [chunk], None, repaired=True, fallback=True
    )
    assert score == 0.6
    assert level == "medium"
    assert "fallback" in reason and "repair" in reason
