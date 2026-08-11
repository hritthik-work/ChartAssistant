from __future__ import annotations

import asyncio
import logging
import time as clock
from datetime import UTC, date, datetime, time
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncAzureOpenAI,
    RateLimitError,
)

from ..core.config import Settings
from ..core.diagnostics import log_event
from ..core.errors import AppError, ConfigurationAppError, ProviderAppError
from ..models import Chunk, SourceDocument
from .ingestion import IndexedBatch

VECTOR_PROFILE = "healthchat-vector-profile"
VECTOR_ALGORITHM = "healthchat-hnsw"
SEMANTIC_CONFIGURATION = "healthchat-semantic"
INDEX_FIELDS = {
    "chunk_id",
    "document_id",
    "title",
    "source_filename",
    "source_role",
    "document_type",
    "document_version",
    "patient_id",
    "encounter_id",
    "encounter_date",
    "author_role",
    "organization",
    "document_owner",
    "effective_date",
    "section",
    "page_start",
    "page_end",
    "order",
    "token_count",
    "content",
    "content_hash",
    "content_vector",
}
LOGGER = logging.getLogger("healthchat.azure")


def _date_time(value: date | None) -> datetime | None:
    return datetime.combine(value, time.min, tzinfo=UTC) if value else None


class AzureEmbeddingService:
    def __init__(self, settings: Settings) -> None:
        if not settings.embeddings_ready:
            raise ConfigurationAppError("Azure OpenAI embeddings are not configured")
        self.deployment = str(settings.azure_openai_embedding_deployment)
        self.client = AsyncAzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            timeout=30.0,
            max_retries=0,
        )

    async def embed(self, texts: list[str]) -> tuple[list[list[float]], int]:
        started = clock.perf_counter()
        log_event(
            LOGGER,
            logging.INFO,
            "azure_openai.embeddings.started",
            deployment=self.deployment,
            input_count=len(texts),
            input_characters=sum(len(text) for text in texts),
        )
        try:
            response = await self.client.embeddings.create(model=self.deployment, input=texts)
        except APITimeoutError as exc:
            log_event(
                LOGGER,
                logging.ERROR,
                "azure_openai.embeddings.failed",
                error=exc,
                deployment=self.deployment,
                latency_ms=round((clock.perf_counter() - started) * 1000, 2),
            )
            raise AppError(
                "Azure OpenAI embeddings timed out",
                category="provider_timeout",
                status_code=503,
                retryable=True,
            ) from exc
        except RateLimitError as exc:
            log_event(
                LOGGER,
                logging.WARNING,
                "azure_openai.embeddings.failed",
                error=exc,
                deployment=self.deployment,
                latency_ms=round((clock.perf_counter() - started) * 1000, 2),
            )
            raise AppError(
                "Azure OpenAI embedding rate limit reached",
                category="rate_limit",
                status_code=429,
                retryable=True,
            ) from exc
        except APIConnectionError as exc:
            log_event(
                LOGGER,
                logging.ERROR,
                "azure_openai.embeddings.failed",
                error=exc,
                deployment=self.deployment,
                latency_ms=round((clock.perf_counter() - started) * 1000, 2),
            )
            raise ProviderAppError("Could not connect to Azure OpenAI embeddings") from exc
        except Exception as exc:
            log_event(
                LOGGER,
                logging.ERROR,
                "azure_openai.embeddings.failed",
                error=exc,
                include_traceback=True,
                deployment=self.deployment,
                latency_ms=round((clock.perf_counter() - started) * 1000, 2),
            )
            raise ProviderAppError(
                f"Azure OpenAI embedding request failed ({type(exc).__name__})",
                retryable=False,
            ) from exc
        ordered = sorted(response.data, key=lambda item: item.index)
        usage = getattr(response, "usage", None)
        tokens = getattr(usage, "prompt_tokens", 0) or getattr(usage, "total_tokens", 0) or 0
        vectors = [item.embedding for item in ordered]
        log_event(
            LOGGER,
            logging.INFO,
            "azure_openai.embeddings.completed",
            deployment=self.deployment,
            vector_count=len(vectors),
            vector_dimensions=len(vectors[0]) if vectors else 0,
            input_tokens=tokens,
            latency_ms=round((clock.perf_counter() - started) * 1000, 2),
        )
        return vectors, tokens


class AzureSearchService:
    def __init__(self, settings: Settings, *, require_admin: bool = False) -> None:
        ready = settings.search_admin_ready if require_admin else settings.search_query_ready
        if not ready:
            key_name = "admin" if require_admin else "query"
            raise ConfigurationAppError(f"Azure AI Search {key_name} access is not configured")
        self.settings = settings
        self.endpoint = str(settings.azure_search_endpoint).rstrip("/")
        self.index_name = settings.azure_search_index_name
        query_key = (
            settings.azure_search_admin_key if require_admin else settings.azure_search_query_key
        )
        self.query_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=AzureKeyCredential(str(query_key)),
        )
        self.admin_client = (
            SearchIndexClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(str(settings.azure_search_admin_key)),
            )
            if settings.search_admin_ready
            else None
        )
        self.index_client = (
            SearchClient(
                endpoint=self.endpoint,
                index_name=self.index_name,
                credential=AzureKeyCredential(str(settings.azure_search_admin_key)),
            )
            if settings.search_admin_ready
            else None
        )

    @staticmethod
    def _schema_mismatch_detail(
        index: SearchIndex, vector_dimensions: int | None = None
    ) -> str | None:
        fields = {field.name: field for field in index.fields}
        missing = INDEX_FIELDS - fields.keys()
        if missing:
            return f"missing fields {sorted(missing)}"
        if vector_dimensions is not None:
            vector_field = fields.get("content_vector")
            existing_dimensions = getattr(vector_field, "vector_search_dimensions", None)
            if existing_dimensions != vector_dimensions:
                return (
                    f"expected {vector_dimensions} vector dimensions, "
                    f"found {existing_dimensions}"
                )
        return None

    def _raise_schema_mismatch(self, detail: str) -> None:
        raise AppError(
            f"Existing index '{self.index_name}' is incompatible: {detail}. "
            "Configure a new AZURE_SEARCH_INDEX_NAME or recreate the index with this app's schema.",
            category="index_schema_mismatch",
            status_code=409,
        )

    async def existing_hashes(self, document_ids: list[str]) -> dict[str, str]:
        if self.index_client is None:
            raise ConfigurationAppError("Azure AI Search admin access is required for ingestion")

        started = clock.perf_counter()
        log_event(
            LOGGER,
            logging.INFO,
            "azure_search.existing_hashes.started",
            index_name=self.index_name,
            document_count=len(document_ids),
        )

        def run() -> dict[str, str]:
            if self.admin_client is None:
                raise ConfigurationAppError(
                    "Azure AI Search admin access is required for ingestion"
                )
            try:
                index = self.admin_client.get_index(self.index_name)
            except ResourceNotFoundError:
                return {}
            mismatch = self._schema_mismatch_detail(index)
            if mismatch:
                self._raise_schema_mismatch(mismatch)
            escaped = [value.replace("'", "''") for value in document_ids]
            filter_text = " or ".join(f"document_id eq '{value}'" for value in escaped)
            results = self.index_client.search(
                search_text="*",
                filter=filter_text,
                select=["chunk_id", "content_hash"],
                top=1000,
            )
            return {str(item["chunk_id"]): str(item["content_hash"]) for item in results}

        try:
            hashes = await asyncio.to_thread(run)
        except Exception as exc:
            log_event(
                LOGGER,
                logging.WARNING if isinstance(exc, AppError) else logging.ERROR,
                "azure_search.existing_hashes.failed",
                error=exc,
                include_traceback=not isinstance(exc, AppError),
                index_name=self.index_name,
                latency_ms=round((clock.perf_counter() - started) * 1000, 2),
            )
            raise
        log_event(
            LOGGER,
            logging.INFO,
            "azure_search.existing_hashes.completed",
            index_name=self.index_name,
            existing_chunk_count=len(hashes),
            latency_ms=round((clock.perf_counter() - started) * 1000, 2),
        )
        return hashes

    def _schema(self, dimensions: int) -> SearchIndex:
        fields = [
            SimpleField(
                name="chunk_id", type=SearchFieldDataType.String, key=True, filterable=True
            ),
            SimpleField(name="document_id", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SimpleField(name="source_filename", type=SearchFieldDataType.String, filterable=True),
            SimpleField(
                name="source_role",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
            ),
            SimpleField(
                name="document_type",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
            ),
            SimpleField(name="document_version", type=SearchFieldDataType.String, filterable=True),
            SimpleField(
                name="patient_id",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
            ),
            SimpleField(name="encounter_id", type=SearchFieldDataType.String, filterable=True),
            SimpleField(
                name="encounter_date",
                type=SearchFieldDataType.DateTimeOffset,
                filterable=True,
                sortable=True,
            ),
            SimpleField(name="author_role", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="organization", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="document_owner", type=SearchFieldDataType.String, filterable=True),
            SimpleField(
                name="effective_date",
                type=SearchFieldDataType.DateTimeOffset,
                filterable=True,
                sortable=True,
            ),
            SearchableField(name="section", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="page_start", type=SearchFieldDataType.Int32, filterable=True),
            SimpleField(name="page_end", type=SearchFieldDataType.Int32, filterable=True),
            SimpleField(
                name="order", type=SearchFieldDataType.Int32, filterable=True, sortable=True
            ),
            SimpleField(name="token_count", type=SearchFieldDataType.Int32, filterable=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(name="content_hash", type=SearchFieldDataType.String, filterable=True),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=dimensions,
                vector_search_profile_name=VECTOR_PROFILE,
                hidden=True,
            ),
        ]
        return SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=VectorSearch(
                algorithms=[HnswAlgorithmConfiguration(name=VECTOR_ALGORITHM)],
                profiles=[
                    VectorSearchProfile(
                        name=VECTOR_PROFILE,
                        algorithm_configuration_name=VECTOR_ALGORITHM,
                    )
                ],
            ),
            semantic_search=SemanticSearch(
                configurations=[
                    SemanticConfiguration(
                        name=SEMANTIC_CONFIGURATION,
                        prioritized_fields=SemanticPrioritizedFields(
                            title_field=SemanticField(field_name="title"),
                            content_fields=[SemanticField(field_name="content")],
                            keywords_fields=[
                                SemanticField(field_name="section"),
                                SemanticField(field_name="document_type"),
                            ],
                        ),
                    )
                ]
            ),
        )

    async def ensure_schema(self, vector_dimensions: int) -> None:
        if self.admin_client is None:
            raise ConfigurationAppError("Azure AI Search admin access is required for ingestion")

        started = clock.perf_counter()
        log_event(
            LOGGER,
            logging.INFO,
            "azure_search.schema_validation.started",
            index_name=self.index_name,
            vector_dimensions=vector_dimensions,
        )

        def run() -> str:
            try:
                existing = self.admin_client.get_index(self.index_name)
            except ResourceNotFoundError:
                self.admin_client.create_index(self._schema(vector_dimensions))
                return "created"
            mismatch = self._schema_mismatch_detail(existing, vector_dimensions)
            if mismatch:
                self._raise_schema_mismatch(mismatch)
            return "validated"

        try:
            outcome = await asyncio.to_thread(run)
        except Exception as exc:
            log_event(
                LOGGER,
                logging.WARNING if isinstance(exc, AppError) else logging.ERROR,
                "azure_search.schema_validation.failed",
                error=exc,
                include_traceback=not isinstance(exc, AppError),
                index_name=self.index_name,
                vector_dimensions=vector_dimensions,
                latency_ms=round((clock.perf_counter() - started) * 1000, 2),
            )
            raise
        log_event(
            LOGGER,
            logging.INFO,
            "azure_search.schema_validation.completed",
            index_name=self.index_name,
            outcome=outcome,
            vector_dimensions=vector_dimensions,
            latency_ms=round((clock.perf_counter() - started) * 1000, 2),
        )

    @staticmethod
    def _index_document(chunk: Chunk, vector: list[float]) -> dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "title": chunk.title,
            "source_filename": chunk.source_filename,
            "source_role": chunk.source_role,
            "document_type": chunk.document_type,
            "document_version": chunk.document_version,
            "patient_id": chunk.patient_id,
            "encounter_id": chunk.encounter_id,
            "encounter_date": _date_time(chunk.encounter_date),
            "author_role": chunk.author_role,
            "organization": chunk.organization,
            "document_owner": chunk.document_owner,
            "effective_date": _date_time(chunk.effective_date),
            "section": chunk.section,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "order": chunk.order,
            "token_count": chunk.token_count,
            "content": chunk.content,
            "content_hash": chunk.content_hash,
            "content_vector": vector,
        }

    async def replace_documents(
        self,
        documents: list[SourceDocument],
        chunks: list[Chunk],
        vectors: dict[str, list[float]],
        existing_hashes: dict[str, str],
    ) -> IndexedBatch:
        if self.index_client is None:
            raise ConfigurationAppError("Azure AI Search admin access is required for ingestion")
        changed = {
            chunk.chunk_id
            for chunk in chunks
            if existing_hashes.get(chunk.chunk_id) != chunk.content_hash
        }
        unchanged = {chunk.chunk_id for chunk in chunks} - changed
        desired = {chunk.chunk_id for chunk in chunks}
        stale = set(existing_hashes) - desired
        upload_payload = [
            self._index_document(chunk, vectors[chunk.chunk_id])
            for chunk in chunks
            if chunk.chunk_id in changed
        ]
        started = clock.perf_counter()
        log_event(
            LOGGER,
            logging.INFO,
            "azure_search.replace_documents.started",
            index_name=self.index_name,
            document_count=len(documents),
            upload_count=len(upload_payload),
            stale_count=len(stale),
        )

        def run() -> None:
            if upload_payload:
                results = self.index_client.merge_or_upload_documents(upload_payload)
                failures = [result.key for result in results if not result.succeeded]
                if failures:
                    raise ProviderAppError(f"Azure AI Search rejected {len(failures)} chunk(s)")
            if stale:
                results = self.index_client.delete_documents(
                    [{"chunk_id": chunk_id} for chunk_id in sorted(stale)]
                )
                failures = [result.key for result in results if not result.succeeded]
                if failures:
                    raise ProviderAppError(
                        f"Azure AI Search failed to remove {len(failures)} stale chunk(s)"
                    )

        try:
            await asyncio.to_thread(run)
        except Exception as exc:
            log_event(
                LOGGER,
                logging.ERROR,
                "azure_search.replace_documents.failed",
                error=exc,
                include_traceback=True,
                index_name=self.index_name,
                upload_count=len(upload_payload),
                stale_count=len(stale),
                latency_ms=round((clock.perf_counter() - started) * 1000, 2),
            )
            raise
        log_event(
            LOGGER,
            logging.INFO,
            "azure_search.replace_documents.completed",
            index_name=self.index_name,
            upload_count=len(upload_payload),
            stale_count=len(stale),
            latency_ms=round((clock.perf_counter() - started) * 1000, 2),
        )
        return IndexedBatch(changed_chunk_ids=changed, unchanged_chunk_ids=unchanged)

    async def hybrid_search(
        self,
        query: str,
        vector: list[float],
        *,
        filter_text: str | None,
        top: int,
        candidates: int,
    ) -> list[dict[str, Any]]:
        select = sorted(INDEX_FIELDS - {"content_vector"})

        def run() -> list[dict[str, Any]]:
            results = self.query_client.search(
                search_text=query,
                vector_queries=[
                    VectorizedQuery(
                        vector=vector,
                        k_nearest_neighbors=candidates,
                        fields="content_vector",
                    )
                ],
                filter=filter_text,
                query_type="semantic",
                semantic_configuration_name=SEMANTIC_CONFIGURATION,
                select=select,
                top=top,
            )
            return [dict(item) for item in results]

        started = clock.perf_counter()
        log_event(
            LOGGER,
            logging.INFO,
            "azure_search.hybrid_search.started",
            index_name=self.index_name,
            query_length=len(query),
            has_filter=filter_text is not None,
            top=top,
            candidates=candidates,
        )
        try:
            result = await asyncio.to_thread(run)
        except HttpResponseError as exc:
            log_event(
                LOGGER,
                logging.ERROR,
                "azure_search.hybrid_search.failed",
                error=exc,
                index_name=self.index_name,
                latency_ms=round((clock.perf_counter() - started) * 1000, 2),
            )
            raise
        log_event(
            LOGGER,
            logging.INFO,
            "azure_search.hybrid_search.completed",
            index_name=self.index_name,
            result_count=len(result),
            latency_ms=round((clock.perf_counter() - started) * 1000, 2),
        )
        return result

    async def list_documents(self) -> list[dict[str, Any]]:
        def run() -> list[dict[str, Any]]:
            results = self.query_client.search(
                search_text="*",
                select=[
                    "document_id",
                    "title",
                    "source_role",
                    "document_type",
                    "patient_id",
                    "encounter_date",
                    "effective_date",
                ],
                top=1000,
            )
            return [dict(item) for item in results]

        return await asyncio.to_thread(run)

    async def probe(self) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            results = self.query_client.search(search_text="*", select=["chunk_id"], top=1)
            first = next(iter(results), None)
            return {"index_name": self.index_name, "has_documents": first is not None}

        return await asyncio.to_thread(run)

    async def probe_admin(self) -> dict[str, Any]:
        if self.admin_client is None:
            raise ConfigurationAppError("Azure AI Search admin access is not configured")

        def run() -> dict[str, Any]:
            names = list(self.admin_client.list_index_names())
            return {
                "index_name": self.index_name,
                "index_exists": self.index_name in names,
                "can_create_index": True,
            }

        return await asyncio.to_thread(run)
