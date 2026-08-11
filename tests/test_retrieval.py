from __future__ import annotations

import pytest

from backend.services.retrieval import RetrievalService

from .conftest import FakeEmbeddingService, hosted_item


class FailingHosted:
    index_name = "hosted-index"

    async def hybrid_search(self, *args, **kwargs):
        raise RuntimeError("search unavailable")


class Hosted:
    index_name = "hosted-index"

    def __init__(self, items):
        self.items = items
        self.last_filter = None

    async def hybrid_search(self, query, vector, *, filter_text, top, candidates):
        self.last_filter = filter_text
        return self.items


@pytest.mark.asyncio
async def test_bm25_fallback_enforces_patient_isolation(settings, artifacts, curated_corpus):
    service = RetrievalService(settings, artifacts)
    result = await service.retrieve("What kidney disease is documented for patient 4?")
    assert result.backend == "local_bm25"
    assert result.chunks
    assert {chunk.patient_id for chunk in result.chunks} == {"SYN-P004"}
    assert "BM25 fallback" in (result.warning or "")


@pytest.mark.asyncio
async def test_bm25_restricts_guide_questions(settings, artifacts, curated_corpus):
    service = RetrievalService(settings, artifacts)
    result = await service.retrieve("What does the documentation guide say about family history?")
    assert result.chunks
    assert {chunk.source_role for chunk in result.chunks} == {"documentation_guide"}


@pytest.mark.asyncio
async def test_in_memory_hybrid_is_first_fallback(settings, artifacts, curated_corpus):
    service = RetrievalService(settings, artifacts, FakeEmbeddingService())
    result = await service.retrieve("What kidney disease is documented for SYN-P004?")
    assert result.backend == "in_memory_hybrid"
    assert result.warning and "not configured" in result.warning
    assert all(chunk.patient_id == "SYN-P004" for chunk in result.chunks)


@pytest.mark.asyncio
async def test_search_failure_uses_visible_hybrid_fallback(settings, artifacts, curated_corpus):
    service = RetrievalService(
        settings,
        artifacts,
        FakeEmbeddingService(),
        FailingHosted(),
    )
    result = await service.retrieve("What kidney disease is documented for SYN-P004?")
    assert result.backend == "in_memory_hybrid"
    assert "Azure AI Search unavailable" in (result.warning or "")


@pytest.mark.asyncio
async def test_embedding_failure_uses_bm25(settings, artifacts, curated_corpus):
    service = RetrievalService(
        settings, artifacts, FakeEmbeddingService(fail=True), FailingHosted()
    )
    result = await service.retrieve("What kidney disease is documented for SYN-P004?")
    assert result.backend == "local_bm25"
    assert "Embedding retrieval unavailable" in (result.warning or "")


@pytest.mark.asyncio
async def test_hosted_semantic_threshold_and_filter(settings, artifacts, curated_corpus):
    _, chunks = curated_corpus
    relevant = next(chunk for chunk in chunks if chunk.patient_id == "SYN-P004")
    unrelated = next(chunk for chunk in chunks if chunk.patient_id == "SYN-P001")
    hosted = Hosted([hosted_item(relevant), hosted_item(unrelated, reranker=1.0)])
    service = RetrievalService(settings, artifacts, FakeEmbeddingService(), hosted)
    result = await service.retrieve("What kidney disease is documented for SYN-P004?")
    assert result.backend == "azure_ai_search"
    assert [chunk.chunk_id for chunk in result.chunks] == [relevant.chunk_id]
    assert "SYN-P004" in (hosted.last_filter or "")
    assert result.rejected_count == 1


@pytest.mark.asyncio
async def test_hosted_retrieval_rejects_semantic_only_unknown_diagnosis(
    settings, artifacts, curated_corpus
):
    _, chunks = curated_corpus
    unrelated = next(chunk for chunk in chunks if chunk.patient_id == "SYN-P001")
    hosted = Hosted([hosted_item(unrelated, reranker=3.5)])
    service = RetrievalService(settings, artifacts, FakeEmbeddingService(), hosted)

    result = await service.retrieve("What schizophrenia is documented for SYN-P001?")

    assert result.backend == "azure_ai_search"
    assert result.chunks == []
    assert result.accepted_count == 0
    assert result.rejected_count == 1
