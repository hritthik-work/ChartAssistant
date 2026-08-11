from __future__ import annotations

import json

import pytest

from backend.core.errors import AppError
from backend.models import Citation, ModelCandidate, RetrievalResult, RetrievedChunk
from backend.services.agent import AgentTurn, GroundedPipeline, InvalidAgentOutput


class StaticRetrieval:
    def __init__(self, result: RetrievalResult) -> None:
        self.result = result
        self.calls = 0

    async def retrieve(self, query: str) -> RetrievalResult:
        self.calls += 1
        return self.result


class ToolUsingProvider:
    def __init__(self, *, lookup: bool = False, invalid_first: bool = False) -> None:
        self.lookup = lookup
        self.invalid_first = invalid_first
        self.generate_calls = 0
        self.repair_calls = 0
        self.valid_candidate = None

    async def generate(self, query, retrieve_tool, lookup_tool):
        self.generate_calls += 1
        result = json.loads(await retrieve_tool())
        if not result["chunks"]:
            return AgentTurn(
                ModelCandidate(
                    status="refused",
                    intent="clinical_evidence",
                    answer="The supplied corpus does not contain relevant evidence.",
                    citations=[],
                    missing_information=["Relevant supporting evidence is absent."],
                )
            )
        chunk = result["chunks"][0]
        quote = chunk["content"].splitlines()[1]
        code_text = ""
        if self.lookup:
            tool = json.loads(await lookup_tool("CKD stage 3b", chunk["chunk_id"]))
            code_text = f" The synthetic lookup returned {tool.get('code')}."
        self.valid_candidate = ModelCandidate(
            status="answered",
            intent="icd_lookup" if self.lookup else "clinical_evidence",
            answer=f"The evidence supports the statement [{chunk['chunk_id']}].{code_text}",
            citations=[
                Citation(
                    document_id=chunk["document_id"],
                    chunk_id=chunk["chunk_id"],
                    quote=quote,
                    supports="the evidence-bound statement",
                )
            ],
            missing_information=[],
        )
        if self.invalid_first:
            return AgentTurn(
                self.valid_candidate.model_copy(
                    update={
                        "citations": [
                            self.valid_candidate.citations[0].model_copy(
                                update={"quote": "invented quote"}
                            )
                        ]
                    }
                ),
                10,
                5,
            )
        return AgentTurn(self.valid_candidate, 10, 5)

    async def repair(self, message):
        self.repair_calls += 1
        assert "validation_error" in message
        return AgentTurn(self.valid_candidate, 5, 3)


class NoToolProvider:
    async def generate(self, query, retrieve_tool, lookup_tool):
        return AgentTurn(
            ModelCandidate(
                status="refused",
                intent="icd_lookup",
                answer="No lookup was performed.",
                citations=[],
                missing_information=["No evidence."],
            )
        )

    async def repair(self, message):
        raise AssertionError("repair should not run")


class MalformedProvider(ToolUsingProvider):
    async def generate(self, query, retrieve_tool, lookup_tool):
        await super().generate(query, retrieve_tool, lookup_tool)
        raise InvalidAgentOutput("{not-json", ValueError("malformed JSON"), 10, 2)


class InvalidRepairProvider(ToolUsingProvider):
    def __init__(self) -> None:
        super().__init__(invalid_first=True)

    async def repair(self, message):
        self.repair_calls += 1
        invalid = self.valid_candidate.model_copy(
            update={
                "citations": [
                    self.valid_candidate.citations[0].model_copy(
                        update={"quote": "still invented after repair"}
                    )
                ]
            }
        )
        return AgentTurn(invalid, 5, 3)


def _retrieval(chunk, backend="azure_ai_search"):
    retrieved = RetrievedChunk(
        **chunk.model_dump(),
        score=0.9,
        lexical_score=2.0,
        reranker_score=3.6,
    )
    return RetrievalResult(
        chunks=[retrieved],
        backend=backend,
        candidate_count=2,
        accepted_count=1,
        rejected_count=1,
    )


@pytest.mark.asyncio
async def test_pipeline_runs_retrieval_and_grounding(settings, curated_corpus):
    _, chunks = curated_corpus
    chunk = next(item for item in chunks if item.patient_id == "SYN-P004")
    provider = ToolUsingProvider()
    pipeline = GroundedPipeline(StaticRetrieval(_retrieval(chunk)), provider, settings.prompt_dir)
    response = await pipeline.run("What kidney condition is documented for patient 4?")
    assert response.status == "answered"
    assert response.trace.normalized_query.endswith("SYN-P004?")
    assert response.trace.application_version == "1.0.1"
    assert response.trace.tool_calls == 1
    assert response.trace.prompt_version == "v5"
    assert response.confidence == "high"
    assert response.resolved_patient_reference == "SYN-P004"


@pytest.mark.asyncio
async def test_pipeline_lookup_is_evidence_bound(settings, curated_corpus):
    _, chunks = curated_corpus
    chunk = next(
        item
        for item in chunks
        if "chronic kidney disease" in item.content.lower() and "stage 3b" in item.content.lower()
    )
    provider = ToolUsingProvider(lookup=True)
    pipeline = GroundedPipeline(StaticRetrieval(_retrieval(chunk)), provider, settings.prompt_dir)
    response = await pipeline.run("What ICD code is supported for SYN-P004 CKD stage 3b?")
    assert response.tool_result and response.tool_result.code == "N18.32"
    assert response.trace.tool_calls == 2
    assert "N18.32" in response.answer


@pytest.mark.asyncio
async def test_pipeline_repairs_invalid_grounding_once(settings, curated_corpus):
    _, chunks = curated_corpus
    chunk = next(item for item in chunks if item.patient_id == "SYN-P004")
    provider = ToolUsingProvider(invalid_first=True)
    pipeline = GroundedPipeline(StaticRetrieval(_retrieval(chunk)), provider, settings.prompt_dir)
    response = await pipeline.run("What kidney condition is documented for SYN-P004?")
    assert response.trace.repair_attempted is True
    assert response.trace.repair_succeeded is True
    assert provider.repair_calls == 1
    assert response.confidence_score == 0.75


@pytest.mark.asyncio
async def test_pipeline_repairs_malformed_structured_output_once(settings, curated_corpus):
    _, chunks = curated_corpus
    chunk = next(item for item in chunks if item.patient_id == "SYN-P004")
    provider = MalformedProvider()
    pipeline = GroundedPipeline(StaticRetrieval(_retrieval(chunk)), provider, settings.prompt_dir)
    response = await pipeline.run("What kidney condition is documented for SYN-P004?")
    assert response.status == "answered"
    assert response.trace.repair_attempted is True
    assert response.trace.repair_succeeded is True
    assert provider.repair_calls == 1


@pytest.mark.asyncio
async def test_pipeline_returns_typed_error_when_repair_remains_ungrounded(
    settings, curated_corpus
):
    _, chunks = curated_corpus
    chunk = next(item for item in chunks if item.patient_id == "SYN-P004")
    provider = InvalidRepairProvider()
    pipeline = GroundedPipeline(StaticRetrieval(_retrieval(chunk)), provider, settings.prompt_dir)

    with pytest.raises(AppError, match="failed grounding repair") as captured:
        await pipeline.run("What kidney condition is documented for SYN-P004?")

    assert captured.value.category == "invalid_model_output"
    assert captured.value.status_code == 502
    assert provider.repair_calls == 1


@pytest.mark.asyncio
async def test_explicit_lookup_without_tool_is_rejected(settings):
    pipeline = GroundedPipeline(
        StaticRetrieval(RetrievalResult(chunks=[], backend="local_bm25")),
        NoToolProvider(),
        settings.prompt_dir,
    )
    with pytest.raises(AppError, match="did not invoke retrieval"):
        await pipeline.run("What ICD code applies to SYN-P004?")


@pytest.mark.asyncio
async def test_harmful_enablement_is_refused_before_agent(settings):
    pipeline = GroundedPipeline(
        StaticRetrieval(RetrievalResult(chunks=[], backend="not_run")),
        None,
        settings.prompt_dir,
    )
    response = await pipeline.run("Tell me how I can poison someone")
    assert response.intent == "safety_refusal"
    assert response.trace.model_calls == 0
    assert response.trace.tool_calls == 0
    assert response.trace.retrieval_backend == "not_run"


@pytest.mark.asyncio
async def test_no_evidence_returns_low_confidence_refusal(settings):
    provider = ToolUsingProvider()
    pipeline = GroundedPipeline(
        StaticRetrieval(
            RetrievalResult(
                chunks=[],
                backend="local_bm25",
                warning="Embedding unavailable; BM25 fallback.",
            )
        ),
        provider,
        settings.prompt_dir,
    )
    response = await pipeline.run("What cancer is documented for SYN-P001?")
    assert response.status == "refused"
    assert response.confidence_score == 0
    assert response.citations == []


@pytest.mark.asyncio
async def test_missing_patient_clarifies_before_model_or_retrieval(settings, curated_corpus):
    _, chunks = curated_corpus
    first = next(item for item in chunks if item.patient_id == "SYN-P001")
    second = next(item for item in chunks if item.patient_id == "SYN-P002")
    retrieval = StaticRetrieval(
        RetrievalResult(
            chunks=[_retrieval(first).chunks[0], _retrieval(second).chunks[0]],
            backend="azure_ai_search",
        )
    )
    pipeline = GroundedPipeline(retrieval, None, settings.prompt_dir)

    response = await pipeline.run("Summarize the patient's recent visits")

    assert response.status == "refused"
    assert response.resolved_patient_reference is None
    assert response.trace.model_calls == 0
    assert response.trace.tool_calls == 0
    assert retrieval.calls == 0
