from __future__ import annotations

import pytest

from backend.core.errors import ValidationAppError
from backend.services.ingestion import (
    ArtifactStore,
    IngestionService,
    UploadPayload,
    chunk_document,
    parse_upload,
)

from .conftest import FakeEmbeddingService, FakeSearchIndex


def test_curated_corpus_parses_and_excludes_boilerplate(curated_uploads):
    documents = [parse_upload(upload) for upload in curated_uploads]
    assert len(documents) == 25
    assert sum(document.source_role == "clinical_note" for document in documents) == 15
    assert sum(document.source_role == "documentation_guide" for document in documents) == 5
    assert sum(document.source_role == "payer_policy" for document in documents) == 5
    assert all("CURATION REVIEW NOTES" not in document.normalized_text for document in documents)
    assert all(document.sections for document in documents)


def test_rejects_missing_notice_and_non_synthetic(curated_uploads):
    source = curated_uploads[0]
    missing_notice = source.content.replace(b"SYNTHETIC MOCK DOCUMENT", b"REMOVED")
    with pytest.raises(ValidationAppError, match="notice"):
        parse_upload(UploadPayload(source.filename, missing_notice))
    non_synthetic = source.content.replace(
        b"DATA_CLASSIFICATION: SYNTHETIC", b"DATA_CLASSIFICATION: REAL"
    )
    with pytest.raises(ValidationAppError, match="SYNTHETIC"):
        parse_upload(UploadPayload(source.filename, non_synthetic))


def test_markdown_frontmatter_is_supported(curated_uploads):
    source = curated_uploads[0].content.decode()
    header, body = source.split("\n\n", 1)
    markdown = f"---\n{header}\n---\n\n{body}".encode()
    document = parse_upload(UploadPayload("sample.md", markdown, "text/markdown"))
    assert document.document_id == "DOC-CLIN-P001-001"


def test_pdf_parser_preserves_page_provenance(monkeypatch, curated_uploads):
    text = curated_uploads[0].content.decode()

    class Page:
        def extract_text(self):
            return text

    class Reader:
        is_encrypted = False
        pages = [Page()]

    monkeypatch.setattr("backend.services.ingestion.PdfReader", lambda _: Reader())
    document = parse_upload(UploadPayload("sample.pdf", b"%PDF fake", "application/pdf"))
    assert document.sections[0].page_start == 1
    assert document.sections[0].page_end == 1


def test_image_only_pdf_is_rejected(monkeypatch):
    class Page:
        def extract_text(self):
            return ""

    class Reader:
        is_encrypted = False
        pages = [Page()]

    monkeypatch.setattr("backend.services.ingestion.PdfReader", lambda _: Reader())
    with pytest.raises(ValidationAppError, match="OCR"):
        parse_upload(UploadPayload("empty.pdf", b"%PDF fake"))


def test_chunking_is_stable_bounded_and_section_aware(settings, curated_uploads):
    document = parse_upload(curated_uploads[0])
    first = chunk_document(document, settings)
    second = chunk_document(document, settings)
    assert first == second
    assert [chunk.chunk_id for chunk in first] == [
        f"{document.document_id}-chunk-{index:03d}" for index in range(1, len(first) + 1)
    ]
    assert all(chunk.token_count <= settings.max_chunk_tokens for chunk in first)
    assert all(chunk.section for chunk in first)


@pytest.mark.asyncio
async def test_ingestion_is_idempotent_and_persists_artifacts(
    settings, artifacts: ArtifactStore, curated_uploads
):
    embeddings = FakeEmbeddingService()
    search = FakeSearchIndex()
    service = IngestionService(settings, embeddings, search, artifacts)
    first = await service.ingest(curated_uploads[:2])
    second = await service.ingest(curated_uploads[:2])
    assert first.status == "indexed"
    assert first.documents_indexed == 2
    assert second.status == "unchanged"
    assert second.chunks_indexed == 0
    assert len(artifacts.load_chunks()) == first.chunks_indexed
    assert artifacts.manifest_path.exists()


@pytest.mark.asyncio
async def test_ingestion_removes_stale_chunks_for_uploaded_document(
    settings, artifacts: ArtifactStore, curated_uploads
):
    embeddings = FakeEmbeddingService()
    search = FakeSearchIndex()
    stale_id = "DOC-CLIN-P001-001-chunk-999"
    search.hashes[stale_id] = "old"
    search.documents[stale_id] = {"chunk_id": stale_id}
    service = IngestionService(settings, embeddings, search, artifacts)
    await service.ingest([curated_uploads[0]])
    assert stale_id in search.stale_deleted
    assert stale_id not in search.documents


def test_upload_limits_are_enforced(settings, artifacts):
    service = IngestionService(settings, None, None, artifacts)
    oversized = UploadPayload("large.txt", b"x" * (settings.max_file_bytes + 1))
    with pytest.raises(ValidationAppError, match="size"):
        service.validate_uploads([oversized])
