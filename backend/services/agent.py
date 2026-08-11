from __future__ import annotations

import inspect
import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Protocol

from pydantic import Field, ValidationError

from ..core.config import Settings
from ..core.diagnostics import log_event
from ..core.errors import AppError, ConfigurationAppError
from ..models import (
    ModelCandidate,
    QueryResponse,
    RetrievalResult,
    RetrievedChunk,
    RetrievedEvidence,
    ToolResult,
    Trace,
)
from .grounding import validate_grounding, workflow_confidence
from .patients import patient_references, resolve_patient
from .policy import evaluate_query_policy
from .retrieval import RetrievalService, requested_roles
from .tools import execute_icd_lookup

LOOKUP_TERMS = re_lookup = ("icd", "code", "lookup")
EVIDENCE_INTENTS = {
    "clinical_evidence",
    "documentation_guidance",
    "payer_policy",
    "mixed_evidence",
    "icd_lookup",
}
LOGGER = logging.getLogger("healthchat.agent")


@dataclass
class AgentTurn:
    candidate: ModelCandidate
    input_tokens: int = 0
    output_tokens: int = 0


class InvalidAgentOutput(RuntimeError):
    def __init__(self, raw: str, error: Exception, input_tokens: int, output_tokens: int) -> None:
        super().__init__(str(error))
        self.raw = raw
        self.error = error
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class AgentProvider(Protocol):
    async def generate(
        self,
        query: str,
        retrieve_tool: Any,
        lookup_tool: Any,
    ) -> AgentTurn: ...

    async def repair(self, message: str) -> AgentTurn: ...


def _usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage_details", None)
    if usage is None:
        return 0, 0
    if isinstance(usage, dict):
        return (
            int(usage.get("input_token_count") or usage.get("prompt_tokens") or 0),
            int(usage.get("output_token_count") or usage.get("completion_tokens") or 0),
        )
    return (
        int(getattr(usage, "input_token_count", 0) or getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "output_token_count", 0) or getattr(usage, "completion_tokens", 0) or 0),
    )


class MafAgentProvider:
    def __init__(self, settings: Settings) -> None:
        if not settings.chat_ready:
            raise ConfigurationAppError("Azure OpenAI chat is not configured")
        from agent_framework import Agent
        from agent_framework.openai import OpenAIChatCompletionClient

        self.agent_class = Agent
        self.deployment = str(settings.azure_openai_chat_deployment)
        self.client = OpenAIChatCompletionClient(
            model=self.deployment,
            azure_endpoint=str(settings.azure_openai_endpoint),
            api_key=str(settings.azure_openai_api_key),
            api_version=settings.azure_openai_api_version,
        )
        self.system_prompt = (settings.prompt_dir / "system_v2.txt").read_text(encoding="utf-8")
        self.repair_prompt = (settings.prompt_dir / "repair_v1.txt").read_text(encoding="utf-8")

    async def generate(self, query: str, retrieve_tool: Any, lookup_tool: Any) -> AgentTurn:
        started = time.perf_counter()
        log_event(
            LOGGER,
            logging.INFO,
            "maf.generate.started",
            deployment=self.deployment,
            query_length=len(query),
            tool_count=2,
        )
        agent = self.agent_class(
            client=self.client,
            name="healthchat_evidence_agent",
            description="Answers synthetic patient-chart questions through bounded tools.",
            instructions=self.system_prompt,
            tools=[retrieve_tool, lookup_tool],
            default_options={
                "allow_multiple_tool_calls": False,
                "response_format": ModelCandidate,
            },
        )
        try:
            response = await agent.run(
                query,
                options={
                    "allow_multiple_tool_calls": False,
                    "response_format": ModelCandidate,
                },
            )
            candidate = response.value
            if not isinstance(candidate, ModelCandidate):
                candidate = ModelCandidate.model_validate_json(response.text)
            input_tokens, output_tokens = _usage(response)
            log_event(
                LOGGER,
                logging.INFO,
                "maf.generate.completed",
                deployment=self.deployment,
                intent=candidate.intent,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            return AgentTurn(candidate, input_tokens, output_tokens)
        except (ValidationError, ValueError) as exc:
            raw = response.text if "response" in locals() else ""
            input_tokens, output_tokens = _usage(response) if "response" in locals() else (0, 0)
            log_event(
                LOGGER,
                logging.ERROR,
                "maf.generate.invalid_output",
                error=exc,
                deployment=self.deployment,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                raw_output_length=len(raw),
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            raise InvalidAgentOutput(raw, exc, input_tokens, output_tokens) from exc
        except AppError:
            raise
        except Exception as exc:
            log_event(
                LOGGER,
                logging.ERROR,
                "maf.generate.failed",
                error=exc,
                include_traceback=True,
                deployment=self.deployment,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            raise AppError(
                f"Microsoft Agent Framework request failed ({type(exc).__name__})",
                category="provider_error",
                status_code=503,
                retryable=True,
            ) from exc

    async def repair(self, message: str) -> AgentTurn:
        started = time.perf_counter()
        log_event(
            LOGGER,
            logging.INFO,
            "maf.repair.started",
            deployment=self.deployment,
            input_length=len(message),
        )
        agent = self.agent_class(
            client=self.client,
            name="healthchat_grounding_repair",
            instructions=self.repair_prompt,
            tools=[],
            default_options={"response_format": ModelCandidate},
        )
        try:
            response = await agent.run(
                message,
                options={"response_format": ModelCandidate},
            )
            candidate = response.value
            if not isinstance(candidate, ModelCandidate):
                candidate = ModelCandidate.model_validate_json(response.text)
            input_tokens, output_tokens = _usage(response)
            log_event(
                LOGGER,
                logging.INFO,
                "maf.repair.completed",
                deployment=self.deployment,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            return AgentTurn(candidate, input_tokens, output_tokens)
        except Exception as exc:
            log_event(
                LOGGER,
                logging.ERROR,
                "maf.repair.failed",
                error=exc,
                include_traceback=True,
                deployment=self.deployment,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            raise AppError(
                f"Model output failed repair ({type(exc).__name__})",
                category="invalid_model_output",
                status_code=502,
            ) from exc


class GroundedPipeline:
    def __init__(
        self,
        retrieval: RetrievalService,
        provider: AgentProvider | None,
        prompt_dir: Path,
    ) -> None:
        self.retrieval = retrieval
        self.provider = provider
        self.prompt_dir = prompt_dir

    async def run(self, query: str) -> QueryResponse:
        started = time.perf_counter()
        request_id = str(uuid.uuid4())
        policy = evaluate_query_policy(query)
        log_event(
            LOGGER,
            logging.INFO,
            "pipeline.started",
            pipeline_request_id=request_id,
            query_length=len(query),
            normalized_query_length=len(policy.normalized_query),
            policy_route=policy.route,
        )
        if policy.route == "safety_refusal":
            log_event(
                LOGGER,
                logging.WARNING,
                "pipeline.safety_refusal",
                pipeline_request_id=request_id,
            )
            return self._safety_response(
                request_id, policy.original_query, policy.normalized_query, started
            )

        available_chunks = getattr(self.retrieval, "chunks", None)
        if available_chunks is None:
            available_chunks = getattr(getattr(self.retrieval, "result", None), "chunks", [])
        references = patient_references(list(available_chunks))
        resolution = resolve_patient(policy.normalized_query, references)
        roles = requested_roles(policy.normalized_query)
        general_only = bool(roles) and roles <= {"documentation_guide", "payer_policy"}
        resolved_patient = resolution.reference if resolution.status == "resolved" else None
        if not general_only and resolution.status != "resolved":
            return self._patient_clarification_response(
                request_id,
                policy.original_query,
                policy.normalized_query,
                resolution.message or "Please identify one patient chart.",
                started,
            )
        if self.provider is None:
            raise ConfigurationAppError(
                "Azure OpenAI chat is required for agentic query processing"
            )

        retrieval_result: RetrievalResult | None = None
        tool_result: ToolResult | None = None
        retrieval_calls = 0
        lookup_calls = 0

        async def retrieve_evidence() -> str:
            """Retrieve bounded evidence for the current normalized user question exactly once."""
            nonlocal retrieval_result, retrieval_calls
            retrieval_calls += 1
            log_event(
                LOGGER,
                logging.INFO,
                "pipeline.retrieve_tool.called",
                pipeline_request_id=request_id,
                call_number=retrieval_calls,
            )
            if retrieval_calls > 1:
                return json.dumps({"status": "rejected", "reason": "retrieval already called"})
            retrieve_parameters = inspect.signature(self.retrieval.retrieve).parameters
            if len(retrieve_parameters) >= 2:
                retrieval_result = await self.retrieval.retrieve(
                    policy.normalized_query,
                    resolved_patient,
                )
            else:
                retrieval_result = await self.retrieval.retrieve(policy.normalized_query)
            log_event(
                LOGGER,
                logging.INFO,
                "pipeline.retrieve_tool.completed",
                pipeline_request_id=request_id,
                backend=retrieval_result.backend,
                result_count=len(retrieval_result.chunks),
                warning=retrieval_result.warning,
            )
            return json.dumps(
                {
                    "status": "evidence" if retrieval_result.chunks else "no_evidence",
                    "backend": retrieval_result.backend,
                    "warning": retrieval_result.warning,
                    "chunks": [
                        {
                            "document_id": chunk.document_id,
                            "chunk_id": chunk.chunk_id,
                            "source_role": chunk.source_role,
                            "section": chunk.section,
                            "content": chunk.content,
                        }
                        for chunk in retrieval_result.chunks
                    ],
                }
            )

        async def mock_icd10_lookup(
            condition: Annotated[
                str, Field(description="Exact condition documented by retrieved clinical evidence.")
            ],
            evidence_chunk_id: Annotated[
                str,
                Field(description="Retrieved clinical chunk explicitly documenting the condition."),
            ],
        ) -> str:
            """Look up one evidence-supported condition in a small non-authoritative ICD table."""
            nonlocal tool_result, lookup_calls
            lookup_calls += 1
            log_event(
                LOGGER,
                logging.INFO,
                "pipeline.icd_tool.called",
                pipeline_request_id=request_id,
                call_number=lookup_calls,
                evidence_chunk_id=evidence_chunk_id,
            )
            if lookup_calls > 1:
                return json.dumps({"status": "rejected", "reason": "lookup already called"})
            tool_result = execute_icd_lookup(
                condition,
                evidence_chunk_id,
                retrieval_result.chunks if retrieval_result else [],
            )
            return tool_result.model_dump_json()

        repair_attempted = False
        repaired = False
        try:
            turn = await self.provider.generate(
                policy.normalized_query,
                retrieve_evidence,
                mock_icd10_lookup,
            )
            candidate = turn.candidate
            input_tokens = turn.input_tokens
            output_tokens = turn.output_tokens
        except InvalidAgentOutput as invalid:
            repair_attempted = True
            retrieval_for_repair = retrieval_result or RetrievalResult(chunks=[], backend="not_run")
            repair_turn = await self.provider.repair(
                self._repair_message(
                    invalid.error,
                    invalid.raw,
                    retrieval_for_repair.chunks,
                    tool_result,
                )
            )
            candidate = repair_turn.candidate
            input_tokens = invalid.input_tokens + repair_turn.input_tokens
            output_tokens = invalid.output_tokens + repair_turn.output_tokens
            repaired = True
        if candidate.intent in EVIDENCE_INTENTS and retrieval_calls != 1:
            raise AppError(
                "Evidence-bound intent did not invoke retrieval exactly once",
                category="invalid_model_output",
                status_code=502,
            )
        explicit_lookup = any(term in policy.normalized_query.lower() for term in LOOKUP_TERMS)
        if explicit_lookup and lookup_calls != 1:
            raise AppError(
                "Explicit lookup request did not invoke the ICD tool exactly once",
                category="invalid_model_output",
                status_code=502,
            )
        retrieval_result = retrieval_result or RetrievalResult(chunks=[], backend="not_run")
        try:
            validate_grounding(candidate, retrieval_result.chunks, tool_result)
        except ValueError as error:
            if repair_attempted:
                raise AppError(
                    f"Model output failed grounding repair: {error}",
                    category="invalid_model_output",
                    status_code=502,
                ) from error
            repair_attempted = True
            repair_turn = await self.provider.repair(
                self._repair_message(error, candidate, retrieval_result.chunks, tool_result)
            )
            input_tokens += repair_turn.input_tokens
            output_tokens += repair_turn.output_tokens
            candidate = repair_turn.candidate
            try:
                validate_grounding(candidate, retrieval_result.chunks, tool_result)
            except ValueError as repair_error:
                raise AppError(
                    f"Model output failed grounding repair: {repair_error}",
                    category="invalid_model_output",
                    status_code=502,
                ) from repair_error
            repaired = True

        status = candidate.status
        missing = list(candidate.missing_information)
        if tool_result and tool_result.status != "matched":
            status = "partial" if candidate.citations else "refused"
            missing.append(f"ICD lookup did not complete with a match: {tool_result.status}.")
        final_candidate = candidate.model_copy(
            update={"status": status, "missing_information": sorted(set(missing))}
        )
        fallback = retrieval_result.backend in {"in_memory_hybrid", "local_bm25"}
        score, confidence, reason = workflow_confidence(
            final_candidate,
            retrieval_result.chunks,
            tool_result,
            repaired=repaired,
            fallback=fallback,
        )
        response = QueryResponse(
            request_id=request_id,
            status=final_candidate.status,
            intent=final_candidate.intent,
            answer=final_candidate.answer,
            resolved_patient_reference=resolved_patient,
            citations=final_candidate.citations,
            tool_result=tool_result,
            confidence_score=score,
            confidence=confidence,
            confidence_reason=reason,
            missing_information=final_candidate.missing_information,
            human_review_required=True,
            trace=self._trace(
                retrieval_result,
                original_query=policy.original_query,
                normalized_query=policy.normalized_query,
                route="rag",
                model_calls=1 + retrieval_calls + lookup_calls + int(repair_attempted),
                tool_calls=retrieval_calls + lookup_calls,
                repair_attempted=repair_attempted,
                repair_succeeded=repaired,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                started=started,
            ),
        )
        log_event(
            LOGGER,
            logging.INFO,
            "pipeline.completed",
            pipeline_request_id=request_id,
            status=response.status,
            intent=response.intent,
            confidence=response.confidence,
            model_calls=response.trace.model_calls,
            tool_calls=response.trace.tool_calls,
            retrieval_backend=response.trace.retrieval_backend,
            latency_ms=response.trace.latency_ms,
        )
        return response

    def _patient_clarification_response(
        self,
        request_id: str,
        original_query: str,
        normalized_query: str,
        message: str,
        started: float,
    ) -> QueryResponse:
        retrieval = RetrievalResult(chunks=[], backend="not_run")
        return QueryResponse(
            request_id=request_id,
            status="refused",
            intent="clinical_evidence",
            answer=message,
            resolved_patient_reference=None,
            citations=[],
            confidence_score=0.0,
            confidence="low",
            confidence_reason="A unique patient chart could not be selected from the question.",
            missing_information=["A unique patient ID, name, or chart number is required."],
            human_review_required=True,
            trace=self._trace(
                retrieval,
                original_query=original_query,
                normalized_query=normalized_query,
                route="rag",
                model_calls=0,
                tool_calls=0,
                repair_attempted=False,
                repair_succeeded=False,
                input_tokens=0,
                output_tokens=0,
                started=started,
            ),
        )

    @staticmethod
    def _repair_message(
        error: ValueError,
        candidate: ModelCandidate | str,
        chunks: list[RetrievedChunk],
        tool_result: ToolResult | None,
    ) -> str:
        context = [
            {
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "source_role": chunk.source_role,
                "content": chunk.content,
            }
            for chunk in chunks
        ]
        return json.dumps(
            {
                "validation_error": str(error),
                "candidate": (
                    candidate.model_dump(mode="json")
                    if isinstance(candidate, ModelCandidate)
                    else candidate
                ),
                "allowed_evidence": context,
                "allowed_tool_result": tool_result.model_dump(mode="json") if tool_result else None,
                "required_schema": ModelCandidate.model_json_schema(),
            }
        )

    @staticmethod
    def _trace(
        retrieval: RetrievalResult,
        *,
        original_query: str,
        normalized_query: str,
        route: str,
        model_calls: int,
        tool_calls: int,
        repair_attempted: bool,
        repair_succeeded: bool,
        input_tokens: int,
        output_tokens: int,
        started: float,
    ) -> Trace:
        return Trace(
            request_route=route,
            original_query=original_query,
            normalized_query=normalized_query,
            retrieval_backend=retrieval.backend,
            search_index_name=retrieval.search_index_name,
            retrieval_candidate_count=retrieval.candidate_count,
            retrieval_accepted_count=retrieval.accepted_count,
            retrieval_rejected_count=retrieval.rejected_count,
            retrieval_warning=retrieval.warning,
            retrieved=[
                RetrievedEvidence(
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    source_role=chunk.source_role,
                    section=chunk.section,
                    score=chunk.score,
                    lexical_score=chunk.lexical_score,
                    vector_score=chunk.vector_score,
                    reranker_score=chunk.reranker_score,
                )
                for chunk in retrieval.chunks
            ],
            embedding_deployment=retrieval.embedding_deployment,
            embedding_input_tokens=retrieval.embedding_input_tokens,
            model_calls=model_calls,
            tool_calls=tool_calls,
            repair_attempted=repair_attempted,
            repair_succeeded=repair_succeeded,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            retrieval_latency_ms=retrieval.latency_ms,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    def _safety_response(
        self,
        request_id: str,
        original_query: str,
        normalized_query: str,
        started: float,
    ) -> QueryResponse:
        retrieval = RetrievalResult(chunks=[], backend="not_run")
        return QueryResponse(
            request_id=request_id,
            status="refused",
            intent="safety_refusal",
            answer=(
                "This service cannot assist with harming a person. Retrieval, the model, "
                "and tools were not run."
            ),
            citations=[],
            confidence_score=0.0,
            confidence="low",
            confidence_reason=(
                "A deterministic pre-retrieval safety policy refused the request; this score "
                "describes evidence support, not confidence in the refusal."
            ),
            missing_information=["The request is outside this patient-chart assistant's scope."],
            human_review_required=True,
            trace=self._trace(
                retrieval,
                original_query=original_query,
                normalized_query=normalized_query,
                route="safety_refusal",
                model_calls=0,
                tool_calls=0,
                repair_attempted=False,
                repair_succeeded=False,
                input_tokens=0,
                output_tokens=0,
                started=started,
            ),
        )
