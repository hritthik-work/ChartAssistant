from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from typing import Protocol

from ..core.config import Settings
from ..core.diagnostics import log_event
from ..models import Chunk, RetrievalResult, RetrievedChunk, SourceRole
from .azure import AzureSearchService
from .ingestion import ArtifactStore, EmbeddingService
from .policy import normalize_query, patient_ids

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "which",
    "with",
    "synthetic",
    "patient",
    "document",
    "documents",
    "record",
    "records",
}
EXPANSIONS = {
    "ckd": ("chronic", "kidney", "disease"),
    "copd": ("chronic", "obstructive", "pulmonary", "disease"),
    "t2dm": ("type", "2", "diabetes", "mellitus"),
    "icd": ("icd", "code", "diagnosis"),
    "cardiovascular": ("heart", "cardiac"),
    "lab": ("laboratory",),
    "laboratory": ("lab",),
    "medications": ("medication", "medicine"),
    "renal": ("kidney",),
}
GENERIC_QUERY_TERMS = {
    "condition",
    "diagnosis",
    "documented",
    "evidence",
    "lookup",
    "record",
    "supported",
}
GUIDE = re.compile(r"\b(guidance|guide|documentation rule|coding rule)\b", re.IGNORECASE)
POLICY = re.compile(r"\b(payer|policy|coverage rule|review policy)\b", re.IGNORECASE)
CLINICAL = re.compile(r"\b(clinical note|patient note|encounter|chart|record)\b", re.IGNORECASE)
LOGGER = logging.getLogger("healthchat.retrieval")


class HostedSearch(Protocol):
    index_name: str

    async def hybrid_search(
        self,
        query: str,
        vector: list[float],
        *,
        filter_text: str | None,
        top: int,
        candidates: int,
    ) -> list[dict]: ...


def tokenize(text: str) -> list[str]:
    original = [item for item in TOKEN.findall(text.lower()) if item not in STOPWORDS]
    expanded: list[str] = []
    for item in original:
        expanded.append(item)
        expanded.extend(EXPANSIONS.get(item, ()))
    return expanded


def requested_roles(query: str) -> set[SourceRole]:
    roles: set[SourceRole] = set()
    if GUIDE.search(query):
        roles.add("documentation_guide")
    if POLICY.search(query):
        roles.add("payer_policy")
    if CLINICAL.search(query):
        roles.add("clinical_note")
    return roles


def _subjects(query: str, patient_reference: str | None = None) -> set[str]:
    return {patient_reference} if patient_reference else set(patient_ids(query))


def _allowed(query: str, chunk: Chunk, patient_reference: str | None = None) -> bool:
    subjects = _subjects(query, patient_reference)
    roles = requested_roles(query)
    if subjects:
        if roles - {"clinical_note"}:
            patient_clinical = chunk.source_role == "clinical_note" and chunk.patient_id in subjects
            requested_general = chunk.source_role in (roles - {"clinical_note"})
            return patient_clinical or requested_general
        return chunk.source_role == "clinical_note" and chunk.patient_id in subjects
    if roles:
        return chunk.source_role in roles
    return True


def _search_text(query: str, patient_reference: str | None = None) -> str:
    cleaned = query
    for patient in sorted(_subjects(query, patient_reference), key=len, reverse=True):
        cleaned = re.sub(re.escape(patient), " ", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.split()) or query


def _has_lexical_support(
    query: str, chunk: Chunk, patient_reference: str | None = None
) -> bool:
    """Reject semantic-only matches when the query contains a specific medical concept."""
    query_terms = set(tokenize(_search_text(query, patient_reference))) - GENERIC_QUERY_TERMS
    if not query_terms:
        return True
    evidence_terms = set(tokenize(f"{chunk.title} {chunk.section} {chunk.content}"))
    return bool(query_terms & evidence_terms)


def _odata_filter(query: str, patient_reference: str | None = None) -> str | None:
    subjects = sorted(_subjects(query, patient_reference))
    roles = requested_roles(query)
    if subjects:
        escaped_subjects = [subject.replace("'", "''") for subject in subjects]
        subject_filter = " or ".join(f"patient_id eq '{subject}'" for subject in escaped_subjects)
        clinical = f"(source_role eq 'clinical_note' and ({subject_filter}))"
        general_roles = roles - {"clinical_note"}
        if general_roles:
            general = " or ".join(f"source_role eq '{role}'" for role in sorted(general_roles))
            return f"({clinical} or ({general}))"
        return clinical
    if roles:
        return " or ".join(f"source_role eq '{role}'" for role in sorted(roles))
    return None


def _normalize(vector: list[float]) -> tuple[float, ...]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if not magnitude:
        raise ValueError("embedding vector has zero magnitude")
    return tuple(value / magnitude for value in vector)


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    return sum(a * b for a, b in zip(left, right, strict=True))


class BM25Index:
    def __init__(self, chunks: list[Chunk], settings: Settings) -> None:
        self.chunks = chunks
        self.settings = settings
        self.term_frequencies = [Counter(tokenize(chunk.content)) for chunk in chunks]
        self.lengths = [sum(value.values()) for value in self.term_frequencies]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        frequency = Counter(term for values in self.term_frequencies for term in values)
        size = len(chunks)
        self.idf = {
            term: math.log(1 + (size - count + 0.5) / (count + 0.5))
            for term, count in frequency.items()
        }

    def scores(self, query: str, patient_reference: str | None = None) -> list[float]:
        terms = set(tokenize(_search_text(query, patient_reference)))
        scores: list[float] = []
        for index, frequencies in enumerate(self.term_frequencies):
            length = self.lengths[index]
            score = 0.0
            for term in terms:
                count = frequencies.get(term, 0)
                if not count:
                    continue
                denominator = count + 1.5 * (
                    1 - 0.75 + 0.75 * length / max(1.0, self.average_length)
                )
                score += self.idf.get(term, 0.0) * count * 2.5 / denominator
            scores.append(score)
        return scores

    def threshold(self, query: str, patient_reference: str | None = None) -> float:
        return self.settings.bm25_min_score * (
            0.25 if _subjects(query, patient_reference) else 1.0
        )

    def bounded(self, ranked: list[RetrievedChunk]) -> list[RetrievedChunk]:
        selected: list[RetrievedChunk] = []
        tokens = 0
        for chunk in ranked[: self.settings.top_k]:
            if selected and tokens + chunk.token_count > self.settings.context_token_budget:
                continue
            selected.append(chunk)
            tokens += chunk.token_count
        return selected

    def retrieve(self, query: str, patient_reference: str | None = None) -> RetrievalResult:
        started = time.perf_counter()
        scores = self.scores(query, patient_reference)
        candidates = [
            index
            for index, chunk in enumerate(self.chunks)
            if _allowed(query, chunk, patient_reference)
        ]
        accepted = [
            RetrievedChunk(
                **self.chunks[index].model_dump(),
                score=round(scores[index], 4),
                lexical_score=round(scores[index], 4),
            )
            for index in candidates
            if scores[index] >= self.threshold(query, patient_reference)
        ]
        accepted.sort(key=lambda item: (-item.score, item.chunk_id))
        return RetrievalResult(
            chunks=self.bounded(accepted),
            backend="local_bm25",
            candidate_count=len(candidates),
            accepted_count=len(accepted),
            rejected_count=len(candidates) - len(accepted),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )


class RetrievalService:
    def __init__(
        self,
        settings: Settings,
        artifacts: ArtifactStore,
        embeddings: EmbeddingService | None = None,
        hosted: HostedSearch | AzureSearchService | None = None,
    ) -> None:
        self.settings = settings
        self.artifacts = artifacts
        self.embeddings = embeddings
        self.hosted = hosted
        self.chunks: list[Chunk] = []
        self.bm25 = BM25Index([], settings)
        self.chunk_vectors: list[tuple[float, ...]] | None = None
        self.last_backend = "not_run"
        self.last_warning: str | None = None
        self.refresh()

    def refresh(self) -> None:
        self.chunks = self.artifacts.load_chunks()
        self.bm25 = BM25Index(self.chunks, self.settings)
        self.chunk_vectors = None
        log_event(
            LOGGER,
            logging.INFO,
            "retrieval.artifacts.refreshed",
            chunk_count=len(self.chunks),
            document_count=len({chunk.document_id for chunk in self.chunks}),
        )

    async def _ensure_chunk_vectors(self) -> int:
        if self.chunk_vectors is not None:
            return 0
        if self.embeddings is None or not self.chunks:
            raise RuntimeError("local vector retrieval is unavailable")
        vectors, tokens = await self.embeddings.embed(
            [
                f"Title: {chunk.title}\nSource role: {chunk.source_role}\n"
                f"Section: {chunk.section}\n{chunk.content}"
                for chunk in self.chunks
            ]
        )
        if len(vectors) != len(self.chunks):
            raise RuntimeError("embedding response count does not match local chunks")
        self.chunk_vectors = [_normalize(vector) for vector in vectors]
        return tokens

    async def _in_memory(
        self,
        query: str,
        query_vector: list[float],
        warning: str,
        query_tokens: int,
        patient_reference: str | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        index_tokens = await self._ensure_chunk_vectors()
        normalized_query = _normalize(query_vector)
        lexical_scores = self.bm25.scores(query, patient_reference)
        candidates = [
            index
            for index, chunk in enumerate(self.chunks)
            if _allowed(query, chunk, patient_reference)
        ]
        accepted: list[RetrievedChunk] = []
        for index in candidates:
            lexical = lexical_scores[index]
            vector_score = _cosine(normalized_query, (self.chunk_vectors or [])[index])
            passes = vector_score >= self.settings.vector_min_similarity or (
                lexical >= self.bm25.threshold(query, patient_reference) and vector_score >= 0.30
            )
            if not passes:
                continue
            lexical_normalized = lexical / (lexical + 3.0)
            semantic_normalized = max(0.0, min(1.0, (vector_score - 0.2) / 0.6))
            fused = 0.55 * lexical_normalized + 0.45 * semantic_normalized
            accepted.append(
                RetrievedChunk(
                    **self.chunks[index].model_dump(),
                    score=round(fused, 4),
                    lexical_score=round(lexical, 4),
                    vector_score=round(vector_score, 4),
                )
            )
        accepted.sort(key=lambda item: (-item.score, item.chunk_id))
        return RetrievalResult(
            chunks=self.bm25.bounded(accepted),
            backend="in_memory_hybrid",
            candidate_count=len(candidates),
            accepted_count=len(accepted),
            rejected_count=len(candidates) - len(accepted),
            warning=warning,
            embedding_deployment=self.embeddings.deployment if self.embeddings else None,
            embedding_input_tokens=query_tokens + index_tokens,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    async def retrieve(
        self, query: str, patient_reference: str | None = None
    ) -> RetrievalResult:
        started = time.perf_counter()
        query = normalize_query(query)
        log_event(
            LOGGER,
            logging.INFO,
            "retrieval.started",
            query_length=len(query),
            local_chunk_count=len(self.chunks),
            hosted_configured=self.hosted is not None,
            embeddings_configured=self.embeddings is not None,
        )
        if not self.chunks and (self.hosted is None or self.embeddings is None):
            result = RetrievalResult(
                chunks=[],
                backend="local_bm25",
                warning="No successfully ingested local artifacts are available.",
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            self.last_backend, self.last_warning = result.backend, result.warning
            log_event(
                LOGGER,
                logging.WARNING,
                "retrieval.completed",
                backend=result.backend,
                result_count=0,
                warning=result.warning,
                latency_ms=result.latency_ms,
            )
            return result

        query_vector: list[float] | None = None
        query_tokens = 0
        embedding_error: Exception | None = None
        if self.embeddings is not None:
            try:
                vectors, query_tokens = await self.embeddings.embed(
                    [_search_text(query, patient_reference)]
                )
                query_vector = vectors[0]
            except Exception as exc:
                embedding_error = exc
                log_event(
                    LOGGER,
                    logging.ERROR,
                    "retrieval.query_embedding.failed",
                    error=exc,
                    latency_ms=round((time.perf_counter() - started) * 1000, 2),
                )

        if self.hosted is not None and query_vector is not None:
            try:
                hosted = await self.hosted.hybrid_search(
                    _search_text(query, patient_reference),
                    query_vector,
                    filter_text=_odata_filter(query, patient_reference),
                    top=self.settings.candidate_limit,
                    candidates=self.settings.candidate_limit,
                )
                accepted: list[RetrievedChunk] = []
                for item in hosted:
                    reranker = item.get("@search.reranker_score")
                    if (
                        reranker is None
                        or float(reranker) < self.settings.semantic_reranker_threshold
                    ):
                        continue
                    chunk_payload = {
                        key: value for key, value in item.items() if not key.startswith("@")
                    }
                    chunk = Chunk.model_validate(chunk_payload)
                    if not _allowed(query, chunk, patient_reference) or not _has_lexical_support(
                        query, chunk, patient_reference
                    ):
                        continue
                    accepted.append(
                        RetrievedChunk(
                            **chunk.model_dump(),
                            score=round(min(1.0, float(reranker) / 4.0), 4),
                            lexical_score=round(float(item.get("@search.score") or 0.0), 4),
                            reranker_score=round(float(reranker), 4),
                        )
                    )
                accepted.sort(key=lambda item: (-item.score, item.chunk_id))
                result = RetrievalResult(
                    chunks=self.bm25.bounded(accepted),
                    backend="azure_ai_search",
                    candidate_count=len(hosted),
                    accepted_count=len(accepted),
                    rejected_count=len(hosted) - len(accepted),
                    search_index_name=self.hosted.index_name,
                    embedding_deployment=self.embeddings.deployment if self.embeddings else None,
                    embedding_input_tokens=query_tokens,
                    latency_ms=round((time.perf_counter() - started) * 1000, 2),
                )
                self.last_backend, self.last_warning = result.backend, result.warning
                log_event(
                    LOGGER,
                    logging.INFO,
                    "retrieval.completed",
                    backend=result.backend,
                    candidate_count=result.candidate_count,
                    accepted_count=result.accepted_count,
                    result_count=len(result.chunks),
                    latency_ms=result.latency_ms,
                )
                return result
            except Exception as exc:
                log_event(
                    LOGGER,
                    logging.WARNING,
                    "retrieval.hosted.failed",
                    error=exc,
                    backend="azure_ai_search",
                    latency_ms=round((time.perf_counter() - started) * 1000, 2),
                )
                warning = (
                    "Azure AI Search unavailable; used in-memory hybrid fallback "
                    f"({type(exc).__name__})."
                )
                try:
                    result = await self._in_memory(
                        query, query_vector, warning, query_tokens, patient_reference
                    )
                    self.last_backend, self.last_warning = result.backend, result.warning
                    log_event(
                        LOGGER,
                        logging.WARNING,
                        "retrieval.completed",
                        backend=result.backend,
                        result_count=len(result.chunks),
                        warning=result.warning,
                        latency_ms=result.latency_ms,
                    )
                    return result
                except Exception as vector_exc:
                    embedding_error = vector_exc
                    log_event(
                        LOGGER,
                        logging.WARNING,
                        "retrieval.in_memory.failed",
                        error=vector_exc,
                    )

        if query_vector is not None:
            warning = "Hosted Search is not configured; used in-memory hybrid fallback."
            try:
                result = await self._in_memory(
                    query, query_vector, warning, query_tokens, patient_reference
                )
                self.last_backend, self.last_warning = result.backend, result.warning
                log_event(
                    LOGGER,
                    logging.WARNING,
                    "retrieval.completed",
                    backend=result.backend,
                    result_count=len(result.chunks),
                    warning=result.warning,
                    latency_ms=result.latency_ms,
                )
                return result
            except Exception as exc:
                embedding_error = exc
                log_event(
                    LOGGER,
                    logging.WARNING,
                    "retrieval.in_memory.failed",
                    error=exc,
                )

        result = self.bm25.retrieve(query, patient_reference)
        reason = type(embedding_error).__name__ if embedding_error else "not_configured"
        result.warning = f"Embedding retrieval unavailable; used BM25 fallback ({reason})."
        result.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        self.last_backend, self.last_warning = result.backend, result.warning
        log_event(
            LOGGER,
            logging.WARNING,
            "retrieval.completed",
            backend=result.backend,
            candidate_count=result.candidate_count,
            accepted_count=result.accepted_count,
            result_count=len(result.chunks),
            warning=result.warning,
            latency_ms=result.latency_ms,
        )
        return result
