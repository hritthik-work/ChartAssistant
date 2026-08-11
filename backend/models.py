from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import __version__

SourceRole = Literal["clinical_note", "documentation_guide", "payer_policy"]
Intent = Literal[
    "clinical_evidence",
    "documentation_guidance",
    "payer_policy",
    "mixed_evidence",
    "icd_lookup",
    "out_of_scope",
    "safety_refusal",
]
AnswerStatus = Literal["answered", "partial", "refused"]
RetrievalBackend = Literal["not_run", "azure_ai_search", "in_memory_hybrid", "local_bm25"]
PromptVersion = Literal["v1", "v2", "v3", "v4", "v5"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceDocument(StrictModel):
    document_id: str
    title: str
    document_type: str
    data_classification: Literal["SYNTHETIC"]
    document_version: str
    source_role: SourceRole
    patient_id: str | None = None
    encounter_id: str | None = None
    encounter_date: date | None = None
    author_role: str | None = None
    organization: str | None = None
    document_owner: str | None = None
    effective_date: date | None = None
    source_filename: str
    normalized_text: str
    sections: list[DocumentSection]


class DocumentSection(StrictModel):
    heading: str
    content: str
    page_start: int | None = None
    page_end: int | None = None


class Chunk(StrictModel):
    chunk_id: str
    document_id: str
    title: str
    source_filename: str
    source_role: SourceRole
    document_type: str
    document_version: str
    patient_id: str | None = None
    encounter_id: str | None = None
    encounter_date: date | None = None
    author_role: str | None = None
    organization: str | None = None
    document_owner: str | None = None
    effective_date: date | None = None
    section: str
    page_start: int | None = None
    page_end: int | None = None
    order: int
    token_count: int
    content: str
    content_hash: str


class RetrievedChunk(Chunk):
    score: float
    lexical_score: float = 0.0
    vector_score: float | None = None
    reranker_score: float | None = None


class RetrievalResult(StrictModel):
    chunks: list[RetrievedChunk]
    backend: RetrievalBackend
    candidate_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    warning: str | None = None
    search_index_name: str | None = None
    embedding_deployment: str | None = None
    embedding_input_tokens: int = 0
    latency_ms: float = 0.0


class IngestionFileResult(StrictModel):
    filename: str
    document_id: str
    status: Literal["indexed", "unchanged"]
    chunks: int


class IngestionResponse(StrictModel):
    ingestion_id: str
    status: Literal["indexed", "unchanged"]
    documents_received: int
    documents_indexed: int
    documents_unchanged: int
    chunks_indexed: int
    chunks_unchanged: int
    index_name: str
    embedding_deployment: str
    files: list[IngestionFileResult]
    latency_ms: float


IngestionStage = Literal[
    "queued",
    "validating",
    "parsing",
    "chunking",
    "checking_changes",
    "embedding",
    "updating_index",
    "saving",
    "completed",
    "failed",
]


class IngestionJobCreated(StrictModel):
    job_id: str
    status: Literal["queued"] = "queued"
    status_url: str


class IngestionJobError(StrictModel):
    category: str
    message: str
    retryable: bool


class IngestionJobStatus(StrictModel):
    job_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    stage: IngestionStage
    progress: int = Field(ge=0, le=100)
    message: str
    created_at: str
    updated_at: str
    result: IngestionResponse | None = None
    error: IngestionJobError | None = None


class QueryRequest(StrictModel):
    query: str = Field(min_length=3, max_length=1000)

    @field_validator("query", mode="before")
    @classmethod
    def normalize_query_input(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class Citation(StrictModel):
    document_id: str
    chunk_id: str
    quote: str = Field(min_length=1)
    supports: str = Field(min_length=1)


class ToolResult(StrictModel):
    tool: Literal["mock_icd10_lookup"]
    query: str
    evidence_chunk_id: str
    status: Literal["matched", "no_match", "timeout", "rejected"]
    code: str | None = None
    description: str | None = None
    warning: str


class ModelCandidate(StrictModel):
    status: AnswerStatus
    intent: Intent
    answer: str
    citations: list[Citation]
    missing_information: list[str]


class RetrievedEvidence(StrictModel):
    document_id: str
    chunk_id: str
    source_role: SourceRole
    section: str
    score: float
    lexical_score: float
    vector_score: float | None = None
    reranker_score: float | None = None


class Trace(StrictModel):
    request_route: Literal["rag", "safety_refusal"]
    original_query: str
    normalized_query: str
    application_version: str = __version__
    prompt_version: PromptVersion = "v5"
    retrieval_backend: RetrievalBackend
    search_index_name: str | None = None
    retrieval_candidate_count: int = 0
    retrieval_accepted_count: int = 0
    retrieval_rejected_count: int = 0
    retrieval_warning: str | None = None
    retrieved: list[RetrievedEvidence]
    embedding_deployment: str | None = None
    embedding_input_tokens: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    repair_attempted: bool = False
    repair_succeeded: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    retrieval_latency_ms: float = 0.0
    latency_ms: float = 0.0


class QueryResponse(StrictModel):
    request_id: str
    status: AnswerStatus
    intent: Intent
    answer: str
    resolved_patient_reference: str | None = None
    citations: list[Citation]
    tool_result: ToolResult | None = None
    confidence_score: float = Field(ge=0, le=1)
    confidence: Literal["high", "medium", "low"]
    confidence_reason: str
    missing_information: list[str]
    human_review_required: Literal[True] = True
    trace: Trace

    @model_validator(mode="after")
    def validate_status(self) -> QueryResponse:
        if self.status == "answered" and not self.citations:
            raise ValueError("answered responses require citations")
        if self.status in {"partial", "refused"} and not self.missing_information:
            raise ValueError("partial/refused responses require missing information")
        return self


class DocumentSummary(StrictModel):
    document_id: str
    title: str
    source_role: SourceRole
    document_type: str
    patient_id: str | None = None
    encounter_date: date | None = None
    effective_date: date | None = None
    chunks: int


class ErrorResponse(StrictModel):
    error: Literal["request_error", "service_error"]
    request_id: str | None = None
    category: Literal[
        "validation",
        "configuration",
        "index_schema_mismatch",
        "provider_timeout",
        "rate_limit",
        "provider_error",
        "invalid_model_output",
        "internal_error",
    ]
    message: str
    retryable: bool
    human_review_required: Literal[True] = True


SourceDocument.model_rebuild()
