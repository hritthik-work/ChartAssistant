from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from backend.core.config import ROOT, Settings
from backend.services.ingestion import (
    ArtifactStore,
    IndexedBatch,
    UploadPayload,
    chunk_document,
    parse_upload,
)


class FakeEmbeddingService:
    deployment = "fake-embedding"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    async def embed(self, texts: list[str]) -> tuple[list[list[float]], int]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("embedding unavailable")
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vectors.append([1.0, digest[0] / 255.0, digest[1] / 255.0])
        return vectors, sum(len(text.split()) for text in texts)


class FakeSearchIndex:
    index_name = "fake-healthchat-index"

    def __init__(self) -> None:
        self.hashes: dict[str, str] = {}
        self.documents: dict[str, dict] = {}
        self.dimensions: int | None = None
        self.stale_deleted: set[str] = set()

    async def existing_hashes(self, document_ids: list[str]) -> dict[str, str]:
        prefixes = tuple(f"{document_id}-chunk-" for document_id in document_ids)
        return {key: value for key, value in self.hashes.items() if key.startswith(prefixes)}

    async def ensure_schema(self, vector_dimensions: int) -> None:
        self.dimensions = vector_dimensions

    async def replace_documents(self, documents, chunks, vectors, existing_hashes):
        changed = {
            chunk.chunk_id
            for chunk in chunks
            if existing_hashes.get(chunk.chunk_id) != chunk.content_hash
        }
        desired = {chunk.chunk_id for chunk in chunks}
        stale = set(existing_hashes) - desired
        self.stale_deleted.update(stale)
        for chunk_id in stale:
            self.hashes.pop(chunk_id, None)
            self.documents.pop(chunk_id, None)
        for chunk in chunks:
            self.hashes[chunk.chunk_id] = chunk.content_hash
            self.documents[chunk.chunk_id] = chunk.model_dump(mode="json")
            if chunk.chunk_id in changed:
                assert chunk.chunk_id in vectors
        return IndexedBatch(
            changed_chunk_ids=changed,
            unchanged_chunk_ids=desired - changed,
        )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        artifact_dir=tmp_path / "artifacts",
        azure_openai_endpoint=None,
        azure_openai_api_key=None,
        azure_openai_chat_deployment=None,
        azure_openai_embedding_deployment=None,
        azure_search_endpoint=None,
        azure_search_admin_key=None,
        azure_search_query_key=None,
    )


@pytest.fixture
def artifacts(settings: Settings) -> ArtifactStore:
    return ArtifactStore(settings.artifact_dir)


@pytest.fixture
def curated_uploads() -> list[UploadPayload]:
    return [
        UploadPayload(path.name, path.read_bytes(), "text/plain")
        for path in sorted((ROOT / "data" / "synthetic-charts").glob("DOC-*.txt"))
    ]


@pytest.fixture
def curated_corpus(settings: Settings, artifacts: ArtifactStore, curated_uploads):
    documents = [parse_upload(upload) for upload in curated_uploads]
    chunks = [chunk for document in documents for chunk in chunk_document(document, settings)]
    artifacts.save(documents, chunks)
    return documents, chunks


def hosted_item(chunk, *, reranker: float = 3.2, score: float = 2.0) -> dict:
    return {
        **chunk.model_dump(mode="json"),
        "@search.score": score,
        "@search.reranker_score": reranker,
    }


def tool_payload(value: str) -> dict:
    return json.loads(value)
